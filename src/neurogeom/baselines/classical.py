"""Baseline B1: classical fit-every-family + geometric-AIC model selection (no learning).

Global context: the strongest non-learned alternative to the network. Every family is fitted to the
cloud by the same two-sided refinement objective the method uses (infer/refine.py), from heuristic
multi-start initialisations instead of a network guess, and the family is chosen by Kanatani's
geometric AIC (infer/select.py). Inits exploit the PCA frame (the input is already PCA-aligned):
  - posed families: the unit default shape with the local frame set to the identity and to the two
    axis permutations that put the shape's main axis e3 on PC1 / PC2 (helix/torus/cone axes and circle
    normals can lie along any PC), plus one random rotation;
  - clusters: Lloyd's k-means (Lloyd 1982) for every k in 2..6, each a separate candidate;
  - polytope: every subtype with two rotations.
"""

from __future__ import annotations

import torch

from neurogeom.geometry.rotation6d import matrix_to_rot6d, random_rotation
from neurogeom.infer.refine import refine, sq_residual_sum
from neurogeom.infer.select import gaic_scores, n_free
from neurogeom.model.net import _default_raw
from neurogeom.shapes.pointsets import K_VALUES, SUBTYPES
from neurogeom.shapes.registry import FAMILIES, INDEX

DENSE = {"curve": 512, "surface_side": 40, "blob": 256}
# frames whose third column (the shape's main axis / normal) is PC3, PC1, PC2 respectively
_PERMS = [torch.eye(3), torch.tensor([[0., 0, 1], [0, 1, 0], [-1, 0, 0]]), torch.tensor([[1., 0, 0], [0, 0, 1], [0, -1, 0]])]


def kmeans(x: torch.Tensor, mask: torch.Tensor, k: int, iters: int = 20, gen=None) -> torch.Tensor:
    """Batched masked Lloyd's k-means; init = k distinct random valid points. Returns (B, k, 3)."""
    b, n, _ = x.shape
    scores = torch.rand(b, n, generator=gen, device=x.device).masked_fill(~mask, -1.0)
    idx = scores.topk(k, -1).indices
    c = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(-1)                                     # (B, N)
        onehot = torch.nn.functional.one_hot(assign, k).to(x.dtype) * mask.unsqueeze(-1)
        cnt = onehot.sum(1)                                                        # (B, k)
        newc = onehot.transpose(1, 2) @ x / cnt.clamp_min(1.0).unsqueeze(-1)
        c = torch.where(cnt.unsqueeze(-1) > 0, newc, c)
    return c


def _init_raws(f, b: int, device, gen) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """List of (raw (B, n_raw), select) initialisations for family f."""
    base = _default_raw(f).to(device)
    out = []
    if f.name == "clusters":
        return out  # handled separately with k-means
    if f.name == "polytope":
        ns = len(SUBTYPES)
        for si in range(ns):
            for r in (torch.eye(3, device=device).expand(b, 3, 3), random_rotation(b, generator=gen, device=device)):
                raw = base.expand(b, -1).clone()
                raw[:, ns + 3:ns + 9] = matrix_to_rot6d(r)
                out.append((raw, torch.full((b,), si, device=device)))
        return out
    rots = [p.to(device).expand(b, 3, 3) for p in _PERMS] + [random_rotation(b, generator=gen, device=device)]
    for r in rots:
        raw = base.expand(b, -1).clone()
        raw[:, 3:9] = matrix_to_rot6d(r)
        out.append((raw, None))
    return out


def fit_all_families(x: torch.Tensor, mask: torch.Tensor, steps: int, back_w: float, gen=None,
                     families: list[int] | None = None, restarts: int | None = None) -> list[dict]:
    """Classical fit of every (or the given) family. Returns candidate dicts with best raw per sample."""
    b, device = x.shape[0], x.device
    cands = []
    for fi in (families if families is not None else range(len(FAMILIES))):
        f = FAMILIES[fi]
        if f.name == "clusters":
            n_min = int(mask.sum(-1).min())
            for j, k in enumerate(K_VALUES):
                if k >= n_min:   # need more points than clusters (k-means init draws k distinct points)
                    continue
                cent = kmeans(x, mask, k, gen=gen)
                p = {"k_idx": torch.full((b,), j, device=device), "centroids": torch.zeros(b, max(K_VALUES), 3, device=device),
                     "spread": torch.full((b,), 0.05, device=device)}
                p["centroids"][:, :k] = cent
                sel = p["k_idx"]
                raw = f.encode(p)
                raw, obj = refine(f, raw, x, mask, sel, steps, 1e-2, DENSE, back_w)
                pd = f.decode(raw, sel)
                cands.append({"fi": fi, "raw": raw, "select": sel, "obj": obj, "J": sq_residual_sum(f, raw, x, mask, sel, DENSE),
                              "p": n_free(f, pd)})
            continue
        inits = _init_raws(f, b, device, gen)
        if restarts is not None and f.name != "polytope":
            inits = inits[:restarts]
        groups = {}
        for raw0, sel in inits:  # polytope: best rotation per subtype is a separate candidate
            key = None if sel is None else int(sel[0])
            raw, obj = refine(f, raw0, x, mask, sel, steps, 1e-2, DENSE, back_w)
            if key not in groups:
                groups[key] = (raw, obj, sel)
            else:
                braw, bobj, _ = groups[key]
                better = obj < bobj
                groups[key] = (torch.where(better.unsqueeze(-1), raw, braw), torch.where(better, obj, bobj), sel)
        for raw, obj, sel in groups.values():
            if sel is not None:  # write the forced sub-choice into the raw logits so decode(raw) reproduces it
                raw = raw.clone()
                raw[:, :len(SUBTYPES)] = -5.0
                raw[torch.arange(b, device=device), sel] = 5.0
            cands.append({"fi": fi, "raw": raw, "select": sel, "obj": obj, "J": sq_residual_sum(f, raw, x, mask, sel, DENSE),
                          "p": n_free(f, f.decode(raw, sel))})
    return cands


def classical_select(cands: list[dict], mask: torch.Tensor) -> dict:
    """Pick, per sample, the candidate with the lowest geometric AIC."""
    aic = gaic_scores(cands, mask.sum(-1))                                   # (B, n_cands)
    best = aic.argmin(-1)
    return {"cand_index": best, "family": torch.tensor([c["fi"] for c in cands], device=aic.device)[best], "aic": aic}


def fit_classical(x, mask, steps: int, back_w: float, gen=None, restarts=None) -> tuple[list[dict], dict]:
    cands = fit_all_families(x, mask, steps, back_w, gen, restarts=restarts)
    return cands, classical_select(cands, mask)


def fit_one_family(fi: int, x, mask, steps: int, back_w: float, gen=None) -> dict:
    """Best classical fit of a single family (clusters: best k by geometric AIC among k candidates)."""
    cands = fit_all_families(x, mask, steps, back_w, gen, families=[fi])
    if len(cands) == 1:
        return cands[0]
    aic = gaic_scores(cands, mask.sum(-1))
    best = aic.argmin(-1)
    out = {"fi": fi, "obj": torch.stack([c["obj"] for c in cands], -1).gather(-1, best.unsqueeze(-1)).squeeze(-1)}
    out["raw"] = torch.stack([c["raw"] for c in cands], 1)[torch.arange(len(best)), best]
    sels = [c["select"] for c in cands]
    out["select"] = None if sels[0] is None else torch.stack(sels, 1)[torch.arange(len(best)), best]
    return out


ARC = INDEX["arc"]
BLOB = INDEX["blob"]
