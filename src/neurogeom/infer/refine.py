"""Test-time refinement: gradient-based fitting of one family's parameters to a point cloud.

Global context: the network gives an amortised first guess (as in SPIN, Kolotouros et al. 2019,
arXiv:1909.12828, where a regressor initialises an optimisation-based fit); refinement then makes the
shape pass as close to the input points as possible. The objective is two-sided:
    J(theta) = mean_i d(x_i, S_theta) + lambda * sum_j w_j d(s_j, X)
The first term alone (points -> shape) is degenerate: a sphere's radius or an arc's extent can grow
to cover every point, and cluster centroids can multiply; the second term (shape -> points, weighted
by length/area) penalises shape regions with no supporting points. The same routine, started from
heuristic initialisations instead of the network, is the classical baseline B1.
"""

from __future__ import annotations

import torch

from neurogeom.geometry.chamfer import nearest_dist, one_sided, residual


def objective(f, raw: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, select, dense: dict, back_w: float) -> torch.Tensor:
    p = f.decode(raw, select)
    pts, w = f.dense(p, dense)
    fwd = residual(x, mask, pts, w)                                   # input points -> shape
    back = one_sided(pts, w, x, mask.to(x.dtype))                     # shape -> input points
    return fwd + back_w * back


def refine(f, raw0: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, select=None, steps: int = 300,
           lr: float = 1e-2, dense: dict | None = None, back_w: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    """Adam on the raw (unconstrained) parameters, independently per sample. Returns (raw, objective)."""
    dense = dense or {"curve": 512, "surface_side": 40, "blob": 256}
    raw = raw0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    best_raw, best_obj = raw.detach().clone(), torch.full((raw.shape[0],), float("inf"), device=raw.device)
    for _ in range(steps):
        obj = objective(f, raw, x, mask, select, dense, back_w)
        with torch.no_grad():                                       # keep the best iterate per sample
            better = obj < best_obj
            best_obj = torch.where(better, obj.detach(), best_obj)
            best_raw[better] = raw.detach()[better]
        opt.zero_grad(set_to_none=True)
        obj.sum().backward()                                        # samples are independent: sum = per-sample steps
        opt.step()
    with torch.no_grad():
        obj = objective(f, raw, x, mask, select, dense, back_w)
        better = obj < best_obj
        best_obj = torch.where(better, obj, best_obj)
        best_raw[better] = raw.detach()[better]
    return best_raw, best_obj


@torch.no_grad()
def sq_residual_sum(f, raw: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, select, dense: dict) -> torch.Tensor:
    """J = sum_i d(x_i, S)^2 over valid points (used by geometric AIC)."""
    p = f.decode(raw, select)
    pts, w = f.dense(p, dense)
    d = nearest_dist(x, pts, w > 0)
    return (d ** 2 * mask.to(d.dtype)).sum(-1)
