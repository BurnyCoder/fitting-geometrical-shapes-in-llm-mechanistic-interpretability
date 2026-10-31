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
_CHUNK = 1 << 26   # max distances per transient block in nearest_index (256 MB in float32)


def pairwise_dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(B, M, 3) x (B, K, 3) -> (B, M, K) Euclidean distances, sqrt(||p - q||^2 + eps)."""
    # ||p||^2 + ||q||^2 - 2 p.q is the standard expansion; clamp at 0 against round-off.
    d2 = (p * p).sum(-1, keepdim=True) + (q * q).sum(-1).unsqueeze(-2) - 2.0 * p @ q.transpose(-1, -2)
    return torch.sqrt(d2.clamp_min(0.0) + _EPS)


@torch.no_grad()
def nearest_index(p: torch.Tensor, q: torch.Tensor, q_valid: torch.Tensor | None = None) -> torch.Tensor:
    """(B, M) index of the nearest valid point of q for each point of p.

    Computed without autograd and in batch chunks of at most _CHUNK distances, so the (B, M, K) matrix
    is never stored for the backward pass (the same split as pytorch3d's knn_points + knn_gather,
    https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.knn_points).
    """
    step = max(1, _CHUNK // max(1, p.shape[1] * q.shape[1]))
    out = []
    for s in range(0, p.shape[0], step):
        pp, qq = p[s:s + step], q[s:s + step]
        # squared distances in one fused kernel, |p|^2 + |q|^2 - 2 p.q (argmin of d^2 = argmin of d, no sqrt needed)
        d2 = torch.baddbmm((pp * pp).sum(-1, keepdim=True) + (qq * qq).sum(-1).unsqueeze(-2), pp, qq.transpose(-1, -2), alpha=-2.0)
        if q_valid is not None:
            d2.masked_fill_(~q_valid[s:s + step].unsqueeze(-2), _BIG)   # padding columns can never be nearest
        out.append(d2.argmin(-1))
    return torch.cat(out)


def nearest_dist(p: torch.Tensor, q: torch.Tensor, q_valid: torch.Tensor | None = None) -> torch.Tensor:
    """For each point of p, the distance to the nearest *valid* point of q. Returns (B, M).

    The gradient of min_j d_ij is the gradient of d_ij at the minimiser, so differentiating only the
    gathered nearest pair gives the same value and gradient as a min over the full matrix (up to ties)
    with O(B M) instead of O(B M K) autograd memory. Test-time refinement of 3 candidate surfaces for
    512 clouds of 256 points against 1600 shape samples needed > 8 GB the old way.
    """
    idx = nearest_index(p, q, q_valid)
    qn = torch.gather(q, 1, idx.unsqueeze(-1).expand(-1, -1, q.shape[-1]))    # (B, M, 3) nearest q per p
    d = torch.sqrt(((p - qn) ** 2).sum(-1) + _EPS)
    if q_valid is not None:   # a sample whose q has no valid point at all keeps the old _BIG sentinel
        d = torch.where(q_valid.any(-1, keepdim=True), d, torch.full_like(d, _BIG))
    return d


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
