"""End-to-end inference: normalised clouds -> network -> top-k refinement -> family selection -> output.

Global context: this is "the deep learning system" the user asked for at inference time. For a batch
of normalised clouds (padded + mask) it returns, per cloud, the selected family, its refined raw
parameters, class probabilities and fit diagnostics. `describe_fit` converts one result into the
unique, canonical parameter description in top-3 PCA coordinates (what a user plots).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neurogeom.geometry.preprocess import PCAFrame, fit_frame
from neurogeom.infer.refine import objective, refine, sq_residual_sum
from neurogeom.infer.select import gaic_scores, n_free, select
from neurogeom.shapes.registry import FAMILIES, NAMES

INFER_DENSE = {"curve": 512, "surface_side": 40, "blob": 256}


@dataclass
class Fit:
    family: int
    raw: torch.Tensor          # refined raw parameters of the selected family (normalised frame)
    probs: np.ndarray          # calibrated class probabilities (13,)
    objective: float           # two-sided refinement objective of the selected fit
    cand: list                 # [(family index, objective, AIC_G)] for the refined top-k candidates
    cand_raws: list | None = None   # refined raw parameters of every candidate (for alternative selection rules)


@torch.no_grad()
def _forward(model, x, mask, temperature):
    out = model(x, mask)
    return out, torch.softmax(out["logits"] / temperature, -1)


def fit_batch(model, x: torch.Tensor, mask: torch.Tensor, infer_cfg: dict, temperature: float = 1.0,
              rule: str | None = None, families: list[int] | None = None) -> list[Fit]:
    """Fit a padded batch. `families` restricts candidates (e.g. a baseline that forces one family)."""
    rule = rule or infer_cfg["selection"]
    out, probs = _forward(model, x, mask, temperature)
    b, k = x.shape[0], infer_cfg["top_k"]
    if families is not None:
        topk = torch.tensor(families, device=x.device).unsqueeze(0).expand(b, -1)
    else:
        topk = probs.topk(k, -1).indices                                   # (B, k) candidate families
    k = topk.shape[1]
    raws = [[None] * k for _ in range(b)]
    objs = torch.full((b, k), float("inf"), device=x.device)
    js = torch.zeros(b, k, device=x.device)
    ps = torch.zeros(b, k, device=x.device)
    for fi in topk.unique().tolist():
        f = FAMILIES[fi]
        rows, slots = torch.nonzero(topk == fi, as_tuple=True)
        raw0 = out["raw"][fi][rows]
        xs, ms = x[rows], mask[rows]
        if infer_cfg.get("refine", True) and infer_cfg["refine_steps"] > 0:
            with torch.enable_grad():
                raw, obj = refine(f, raw0, xs, ms, None, infer_cfg["refine_steps"], infer_cfg["refine_lr"],
                                  INFER_DENSE, infer_cfg["refine_back_weight"])
        else:  # ablation A3: the network's raw output as is
            raw, obj = raw0, objective(f, raw0, xs, ms, None, INFER_DENSE, infer_cfg["refine_back_weight"])
        j = sq_residual_sum(f, raw, xs, ms, None, INFER_DENSE)
        pdec = f.decode(raw)
        objs[rows, slots], js[rows, slots], ps[rows, slots] = obj.detach(), j, n_free(f, pdec)
        for r, s, rr in zip(rows.tolist(), slots.tolist(), raw.detach()):
            raws[r][s] = rr
    # Columns hold different families in different rows, so AIC_G is computed row by row.
    aic = torch.stack([_row_gaic(topk[i], js[i], ps[i], mask[i].sum()) for i in range(b)])
    pc = torch.gather(probs, 1, topk)                                     # probabilities of the candidates
    fits = []
    for i in range(b):
        s = int(select(pc[i:i + 1], list(range(k)), aic[i:i + 1], rule)[0])
        fits.append(Fit(family=int(topk[i, s]), raw=raws[i][s], probs=probs[i].cpu().numpy(), objective=float(objs[i, s]),
                        cand=[(int(topk[i, j]), float(objs[i, j]), float(aic[i, j])) for j in range(k)],
                        cand_raws=raws[i]))
    return fits


def _row_gaic(fams: torch.Tensor, j: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    cands = [{"fi": int(fi), "J": j[c:c + 1], "p": p[c:c + 1]} for c, fi in enumerate(fams.tolist())]
    return gaic_scores(cands, n.view(1))[0]


def describe_fit(fit: Fit, frame: PCAFrame | None = None) -> dict:
    """Canonical, unique parameters of the fitted shape in top-3 PCA coordinates (unnormalised)."""
    f = FAMILIES[fit.family]
    p = f.decode(fit.raw.unsqueeze(0).cpu())
    if frame is not None:  # normalised frame -> PCA coordinates: x_pca = scale * x_norm
        eye = torch.eye(3).unsqueeze(0)
        p = f.transform(p, eye, torch.zeros(1, 3), torch.tensor([frame.scale], dtype=torch.float32))
    d = f.describe(f.canonicalize(p), 0)
    d["probability"] = float(fit.probs[fit.family])
    d["top_probs"] = {NAMES[i]: float(fit.probs[i]) for i in np.argsort(-fit.probs)[:3]}
    d["objective_normalised"] = fit.objective
    return d


def fit_cloud(model, x_raw: np.ndarray, infer_cfg: dict, temperature: float = 1.0, device=None) -> tuple[Fit, PCAFrame, np.ndarray]:
    """Convenience API for one real cloud (N, D): PCA-normalise, fit, return (fit, frame, normalised points)."""
    y, frame = fit_frame(x_raw)
    device = device or next(model.parameters()).device
    xt = torch.as_tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.ones(1, y.shape[0], dtype=torch.bool, device=device)
    return fit_batch(model, xt, m, infer_cfg, temperature)[0], frame, y
