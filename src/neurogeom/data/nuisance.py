"""Nuisance factors of the synthetic generator: noise, sampling density, outliers, high-dim embedding.

Global context: real top-3 PCA clouds are noisy, sampled non-uniformly along the manifold (e.g. 7
weekdays, clumped number tokens), contain outliers, and come from activations in R^D (D = 768..4096)
where noise lives in every dimension. Training with these nuisances (domain randomisation) is what
lets a synthetic-only fitter transfer to real activations (Tobin et al. 2017, "Domain Randomization
for Transferring Deep Neural Networks from Simulation to the Real World", arXiv:1703.06907).
"""

from __future__ import annotations

import math

import torch

from neurogeom.geometry.rotation6d import random_rotation


def log_uniform(b: int, lo: float, hi: float, gen: torch.Generator, device) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(b, generator=gen, device=device))


def sample_sigma(b: int, cfg: dict, gen: torch.Generator, device) -> torch.Tensor:
    """Relative noise level per sample: log-uniform, with a fraction of exactly noise-free samples."""
    s = log_uniform(b, cfg["noise_min"], cfg["noise_max"], gen, device)
    free = torch.rand(b, generator=gen, device=device) < cfg["noise_free_frac"]
    return torch.where(free, torch.zeros_like(s), s)


def density_factor(u: torch.Tensor, gen: torch.Generator, modes: list[str]) -> torch.Tensor:
    """Multiplicative sampling density over the shape's parameter domain u in [0,1]^d, (B, G).

    'uniform' -> 1; 'beta' -> Beta-like profile along u_0; 'clumped' -> 1-3 Gaussian bumps.
    Floored at 0.2 of the maximum so every part of the shape keeps some probability mass.
    """
    b, g, d = u.shape
    dev = u.device
    mode = torch.randint(0, len(modes), (b,), generator=gen, device=dev)
    out = torch.ones(b, g, device=dev)
    if "beta" in modes:
        a = 0.7 + 2.3 * torch.rand(b, 1, generator=gen, device=dev)
        bb = 0.7 + 2.3 * torch.rand(b, 1, generator=gen, device=dev)
        x = u[..., 0].clamp(1e-3, 1 - 1e-3)
        f = torch.exp((a - 1) * torch.log(x) + (bb - 1) * torch.log1p(-x))
        f = f / f.amax(-1, keepdim=True)
        out = torch.where((mode == modes.index("beta")).unsqueeze(-1), f, out)
    if "clumped" in modes:
        nb = 3
        mu = torch.rand(b, nb, d, generator=gen, device=dev)
        width = 0.05 + 0.15 * torch.rand(b, nb, 1, generator=gen, device=dev)
        on = torch.rand(b, nb, generator=gen, device=dev) < 0.67
        on[:, 0] = True
        d2 = ((u.unsqueeze(1) - mu.unsqueeze(2)) ** 2).sum(-1)                       # (B, nb, G)
        f = (torch.exp(-d2 / (2 * width ** 2)) * on.unsqueeze(-1)).sum(1)
        f = f / f.amax(-1, keepdim=True).clamp_min(1e-12)
        out = torch.where((mode == modes.index("clumped")).unsqueeze(-1), f, out)
    return out.clamp_min(0.2)


def add_noise(x: torch.Tensor, sigma_abs: torch.Tensor, aniso_prob: float, gen: torch.Generator) -> torch.Tensor:
    """Gaussian noise with per-sample std sigma_abs; optionally anisotropic in a random frame."""
    b = x.shape[0]
    eps = torch.randn(x.shape, generator=gen, device=x.device)
    aniso = torch.rand(b, generator=gen, device=x.device) < aniso_prob
    scales = 0.3 + 0.7 * torch.rand(b, 1, 3, generator=gen, device=x.device)
    scales = torch.where(aniso.view(-1, 1, 1), scales, torch.ones_like(scales))
    rot = random_rotation(b, generator=gen, device=x.device)
    eps = (eps * scales) @ rot.transpose(-1, -2)
    return x + sigma_abs.view(-1, 1, 1) * eps


def add_outliers(x: torch.Tensor, prob: float, frac_max: float, gen: torch.Generator) -> torch.Tensor:
    """Replace a random fraction of points by uniform draws from the padded bounding box."""
    b, n, _ = x.shape
    dev = x.device
    has = torch.rand(b, generator=gen, device=dev) < prob
    frac = frac_max * torch.rand(b, generator=gen, device=dev) * has
    is_out = torch.rand(b, n, generator=gen, device=dev) < frac.unsqueeze(-1)
    lo, hi = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    pad = 0.2 * (hi - lo)
    box = lo - pad + (hi - lo + 2 * pad) * torch.rand(x.shape, generator=gen, device=dev)
    return torch.where(is_out.unsqueeze(-1), box, x)


def random_embedding(b: int, d: int, gen: torch.Generator, device) -> torch.Tensor:
    """Random orthonormal 3-frames in R^D, (B, D, 3), to embed 3D shapes into activation-like spaces.
    Gram-Schmidt on 3 Gaussian vectors (= Haar-distributed frame), written out to avoid batched QR."""
    a = torch.randn(b, d, 3, generator=gen, device=device)
    e1 = torch.nn.functional.normalize(a[..., 0], dim=-1)
    v2 = a[..., 1] - (a[..., 1] * e1).sum(-1, keepdim=True) * e1
    e2 = torch.nn.functional.normalize(v2, dim=-1)
    v3 = a[..., 2] - (a[..., 2] * e1).sum(-1, keepdim=True) * e1 - (a[..., 2] * e2).sum(-1, keepdim=True) * e2
    return torch.stack([e1, e2, torch.nn.functional.normalize(v3, dim=-1)], -1)
