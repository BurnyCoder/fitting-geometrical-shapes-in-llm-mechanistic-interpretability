"""On-the-fly synthetic training/evaluation data: ground-truth shape -> noisy point cloud -> top-3 PCA.

Global context: the fitter is trained only on synthetic shapes. For one family, `generate_family`
draws ground-truth parameters (respecting label margins), samples N observed points with nuisances,
optionally embeds them in R^D with noise in all dimensions, and applies *exactly* the preprocessing
used for real activations (geometry/preprocess.py). It returns the normalised input cloud plus the
ground-truth shape in the same normalised frame, both as parameters (for auxiliary losses and
parameter-error metrics) and as dense weighted samples (for the shape-to-shape Chamfer loss).
"""

from __future__ import annotations

import torch

from neurogeom.data.nuisance import (add_noise, add_outliers, density_factor, random_embedding, sample_sigma)
from neurogeom.geometry.preprocess import apply_transform, normalize_batch
from neurogeom.geometry.rotation6d import nearest_orthogonal
from neurogeom.shapes.registry import FAMILIES, INDEX


def _observe_pointset(f, p, n: int, gen) -> torch.Tensor:
    """Observed points for clusters/polytopes: every centroid/vertex gets >= 1 point, the rest are
    distributed by random Dirichlet proportions; each point = vertex + spread * N(0, I)."""
    verts, w = f.points(p)
    b, vmax, _ = verts.shape
    valid = w > 0
    k = valid.sum(-1)
    dev = verts.device
    # first k points: one per vertex (vertex order is random already because the pose is random)
    first = torch.arange(n, device=dev).unsqueeze(0).expand(b, -1)
    props = -torch.log(torch.rand(b, vmax, generator=gen, device=dev).clamp_min(1e-9)) * valid   # Dirichlet(1) via Exp(1)
    rest = torch.multinomial(props / props.sum(-1, keepdim=True), n, replacement=True, generator=gen)
    assign = torch.where(first < k.unsqueeze(-1), first, rest)
    pts = torch.gather(verts, 1, assign.unsqueeze(-1).expand(-1, -1, 3))
    return pts + p["spread"].view(-1, 1, 1) * torch.randn(pts.shape, generator=gen, device=dev)


def _observe_manifold(f, p, n: int, cfg: dict, gen) -> torch.Tensor:
    """Observed points on a curve/surface/blob: draw N of G candidates with probability proportional to
    (natural density) x (random density nuisance), without replacement."""
    b = p["center"].shape[0]
    dev = p["center"].device
    u = f.random_u(b, cfg["gen_candidates"], gen, dev)
    pts, w = f.points(p, u)
    if f.name == "blob":
        return pts[:, :n]
    w = f.natural_weights(p, u, w) * density_factor(u, gen, cfg["density_modes"])
    idx = torch.multinomial(w / w.sum(-1, keepdim=True), n, replacement=False, generator=gen)
    return torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def generate_family(fam, b: int, n: int, cfg: dict, cfg_dense: dict, gen: torch.Generator, device,
                    sigma: torch.Tensor | None = None, embed_dim: int | None = -1) -> dict:
    """One batch of b samples of family `fam` (name, index or object) with n points each.

    sigma: optional fixed relative noise per sample (else drawn from cfg). embed_dim: -1 -> random
    per config (probability cfg['embed_prob']), None -> no embedding, int -> embed in R^embed_dim.
    """
    f = FAMILIES[INDEX[fam]] if isinstance(fam, str) else (FAMILIES[fam] if isinstance(fam, int) else fam)
    if sigma is None:
        sigma = sample_sigma(b, cfg, gen, device)
    sigma = sigma.to(device)
    if f.posed:
        p = f.sample_valid(b, gen, sigma, device)
        x0 = _observe_manifold(f, p, n, cfg, gen)
    else:
        p = f.sample_valid(b, gen, sigma, device, k_max=n)
        x0 = _observe_pointset(f, p, n, gen)
    rms = f.rms_radius(p, cfg_dense)
    x = add_noise(x0, sigma * rms, cfg["aniso_noise_prob"], gen)
    x = add_outliers(x, cfg["outlier_prob"], cfg["outlier_frac_max"], gen)

    # --- optional embedding in R^D (mimics activations: the shape spans 3 of D dims, noise spans all D)
    if embed_dim == -1:
        embed = bool(torch.rand(1, generator=gen, device=device) < cfg["embed_prob"])
        embed_dim = cfg["embed_dims"][int(torch.randint(0, len(cfg["embed_dims"]), (1,), generator=gen, device=device))] if embed else None
    gt_pts, gt_w = f.dense(p, cfg_dense)
    if embed_dim:
        e = random_embedding(b, embed_dim, gen, device)                         # (B, D, 3)
        eta = 0.3 + 0.7 * torch.rand(b, 1, 1, generator=gen, device=device)
        x = x @ e.transpose(-1, -2) + (sigma * rms).view(-1, 1, 1) * eta * torch.randn(b, n, embed_dim, generator=gen, device=device)
        gt_pts = gt_pts @ e.transpose(-1, -2)

    # --- the same preprocessing as for real activations
    y, t = normalize_batch(x)
    gt_dense = apply_transform(gt_pts, t)
    # ground-truth parameters in the normalised frame: x_norm = (V^T E x3 - V^T mean) / scale
    v = t["basis"]                                                              # (B, D, 3)
    m = v.transpose(-1, -2) @ e if embed_dim else v.transpose(-1, -2)           # (B, 3, 3), ~orthogonal
    q = nearest_orthogonal(m)        # may be improper after a D > 3 embedding; transform() handles mirrors
    # The embedding and the PCA projection are isometries on the shape's subspace, so the scale is exactly
    # 1/scale. (m can be rank-deficient when the shape is flat and the 3rd PC is a pure-noise direction;
    # the polar factor q then maps the unused local axis to an arbitrary orthogonal direction, which does
    # not move any point of the flat shape.)
    s = 1.0 / t["scale"]
    tr = -(t["mean"].unsqueeze(1) @ v).squeeze(1) / t["scale"].unsqueeze(-1)
    gt = f.transform(p, q, tr, s)
    return {"x": y, "mask": torch.ones(b, n, dtype=torch.bool, device=device), "family": torch.full((b,), INDEX[f.name], device=device),
            "sigma": sigma, "gt": gt, "gt_dense": gt_dense, "gt_w": gt_w, "embed_dim": embed_dim or 3,
            "eig": t["eig"]}


def generate_mixed(b: int, n: int, cfg: dict, cfg_dense: dict, gen: torch.Generator, device) -> list[dict]:
    """A training step's data: all 13 families, ~b/13 samples each (remainder spread at random)."""
    k = len(FAMILIES)
    counts = torch.full((k,), b // k)
    counts[torch.randperm(k, generator=gen, device=device)[: b % k].cpu()] += 1
    return [generate_family(i, int(c), n, cfg, cfg_dense, gen, device) for i, c in enumerate(counts) if c > 0]
