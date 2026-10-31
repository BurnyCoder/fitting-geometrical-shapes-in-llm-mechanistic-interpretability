"""Masked, weighted point-set distances: Chamfer distance and point-to-shape residual.

Global context: the training loss compares the predicted shape with the ground-truth shape through
dense samples of both (the user's requirement: "evaluate how close the output shape is to the ground
truth shape"), and the residual term measures how close the input points lie to the predicted shape.
pytorch3d's chamfer_distance is not practical to install on Windows with CUDA 13 wheels, so this is a
small torch.cdist-style implementation (pairwise distances are cheap at N <= 2048, M <= 1024).
Distances are unsquared, sqrt(d^2 + eps), so the metric is in the same units as the coordinates
(the RMS radius of the normalised cloud) and robust to a few far points (Fan et al. 2017 used squared
Chamfer; we prefer the unsquared, metric version as in many shape-fitting evaluations).
"""

from __future__ import annotations

import torch

_EPS = 1e-8   # keeps sqrt differentiable when two points coincide (d = 0)
_BIG = 1e6    # distance assigned to masked-out (padding) points so they never win a min


def pairwise_dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(B, M, 3) x (B, K, 3) -> (B, M, K) Euclidean distances, sqrt(||p - q||^2 + eps)."""
    # ||p||^2 + ||q||^2 - 2 p.q is the standard expansion; clamp at 0 against round-off.
    d2 = (p * p).sum(-1, keepdim=True) + (q * q).sum(-1).unsqueeze(-2) - 2.0 * p @ q.transpose(-1, -2)
    return torch.sqrt(d2.clamp_min(0.0) + _EPS)


def nearest_dist(p: torch.Tensor, q: torch.Tensor, q_valid: torch.Tensor | None = None) -> torch.Tensor:
    """For each point of p, the distance to the nearest *valid* point of q. Returns (B, M)."""
    d = pairwise_dist(p, q)
    if q_valid is not None:
        d = d.masked_fill(~q_valid.unsqueeze(-2), _BIG)   # padding columns can never be nearest
    return d.min(dim=-1).values


def _normalise(w: torch.Tensor) -> torch.Tensor:
    return w / w.sum(-1, keepdim=True).clamp_min(1e-12)


def chamfer(p: torch.Tensor, wp: torch.Tensor, q: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """Symmetric weighted Chamfer distance, (B,).

    CD(P, Q) = sum_i wp_i min_j |p_i - q_j| + sum_j wq_j min_i |q_j - p_i|, with weights normalised
    per sample. A weight of exactly 0 marks a padding point (excluded from both sums and mins).
    The weights encode the length/area element of each dense sample, so the distance approximates the
    integral over the shape rather than over the sampling parameter.
    """
    vp, vq = wp > 0, wq > 0
    d = pairwise_dist(p, q)
    d_pq = d.masked_fill(~vq.unsqueeze(-2), _BIG).min(-1).values        # p -> nearest q
    d_qp = d.masked_fill(~vp.unsqueeze(-1), _BIG).min(-2).values        # q -> nearest p
    return (_normalise(wp) * d_pq).sum(-1) + (_normalise(wq) * d_qp).sum(-1)


def one_sided(p: torch.Tensor, wp: torch.Tensor, q: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """Weighted mean distance from the points of p to the shape sampled by q, (B,)."""
    d = nearest_dist(p, q, wq > 0)
    return (_normalise(wp) * d).sum(-1)


def huber(d: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber penalty of non-negative distances (quadratic below delta, linear above)."""
    return torch.where(d < delta, 0.5 * d * d / delta, d - 0.5 * delta)


def residual(x: torch.Tensor, x_mask: torch.Tensor, s: torch.Tensor, ws: torch.Tensor, delta: float | None = None) -> torch.Tensor:
    """Mean (Huber-)distance from the valid input points x to the shape samples s, (B,).

    This is the "goes through the input points" term. With delta=None it is the plain mean distance.
    """
    d = nearest_dist(x, s, ws > 0)
    if delta is not None:
        d = huber(d, delta)
    m = x_mask.to(d.dtype)
    return (d * m).sum(-1) / m.sum(-1).clamp_min(1.0)
