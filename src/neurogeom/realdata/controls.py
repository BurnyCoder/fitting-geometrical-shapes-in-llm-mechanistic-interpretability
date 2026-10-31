"""Controls C1 (ordinal processes -> PCA ringing) and C2 (matched-covariance Gaussians).

Global context:
- C1: high-dimensional trajectories over an ordinal index with no designed curvature. PCA of a smooth
  Gaussian process or of a random walk yields principal components ~ cos(k pi s) (Novembre & Stephens
  2008; Antognini & Sohl-Dickstein 2018, arXiv:1806.08805), i.e. an arch / Lissajous curve in the
  top-3 PCs. The fitter should call these `ringing`, not `helix` (preregistered H4).
- C2: for every real dataset, a Gaussian with the same mean and covariance and the same N. It has no
  low-dimensional structure beyond its covariance, so the fitter should call it `blob` (H4).
"""

from __future__ import annotations

import numpy as np

from neurogeom.realdata.common import save_cloud


def ordinal_controls(cfg: dict, n_instances: int = 10, n: int = 200, d: int = 768) -> None:
    rng = np.random.default_rng(1234)
    s = np.linspace(0, 1, n)
    for i in range(n_instances):
        ell = rng.uniform(0.08, 0.3)                                      # RBF length scale
        k = np.exp(-0.5 * (s[:, None] - s[None]) ** 2 / ell ** 2) + 1e-6 * np.eye(n)
        x_gp = np.linalg.cholesky(k) @ rng.normal(size=(n, d))          # each of the d dims is an independent GP draw
        save_cloud(cfg, f"C1_gp_{i}", x_gp, np.arange(n), {"source": f"RBF Gaussian process over an ordinal index, length scale {ell:.3f}, R^{d}",
                                                          "expected_coarse": "1d", "expected_fine": ["ringing"], "prereg": True})
        x_rw = np.cumsum(rng.normal(size=(n, d)), 0)                    # random walk (Antognini & Sohl-Dickstein 2018)
        save_cloud(cfg, f"C1_walk_{i}", x_rw, np.arange(n), {"source": f"Gaussian random walk, {n} steps in R^{d}",
                                                            "expected_coarse": "1d", "expected_fine": ["ringing"], "prereg": True})


def matched_gaussian(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample N points from N(mean(x), cov(x)) without forming the D x D covariance:
    mu + Xc^T g / sqrt(N) with g ~ N(0, I_N) has covariance Xc^T Xc / N."""
    mu = x.mean(0)
    xc = x - mu
    g = rng.normal(size=(len(x), len(x)))
    return mu + g @ xc / np.sqrt(len(x))
