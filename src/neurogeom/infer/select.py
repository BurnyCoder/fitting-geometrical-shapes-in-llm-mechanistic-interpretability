"""Model selection among candidate shape families: calibrated classifier, geometric AIC, or hybrid.

Global context: after refinement, several families may fit a cloud (nested families: a full circle is
also an ellipse, a helix with tiny pitch is almost a circle). Selection rules (ablation A5):
- classifier: argmax of temperature-scaled softmax probabilities (Guo et al. 2017, "On Calibration of
  Modern Neural Networks", arXiv:1706.04599).
- gaic: Kanatani's geometric AIC, AIC_G = J + 2 (d N + p) eps^2, with J the sum of squared point-to-shape
  distances, d the manifold dimension, p the number of shape parameters and eps^2 the noise variance,
  estimated from the most general candidate as eps^2 = J_gen / ((3 - d_gen) N - p_gen)
  (Kanatani 2004, "Uncertainty Modeling and Model Selection for Geometric Inference", IEEE TPAMI 26(10);
  https://iim.cs.tut.ac.jp/member/kanatani/papers/acmodel.pdf). The volumetric blob has d = 3, J = 0.
- hybrid: among candidates whose calibrated probability is >= 0.1 x the top probability, the lowest AIC_G.
"""

from __future__ import annotations

import torch

from neurogeom.shapes.registry import FAMILIES


def n_free(f, p: dict) -> torch.Tensor:
    if f.name == "clusters":
        return (3 * p["k"] + 1).to(torch.float32)
    return torch.full_like(p["center"][:, 0] if "center" in p else p["spread"], float(f.n_free))


def gaic_scores(cands: list[dict], n_points: torch.Tensor) -> torch.Tensor:
    """cands: list over families of {"fi", "J" (B,), "p" (B,)}; returns AIC_G (B, n_cands)."""
    n = n_points.to(torch.float32)
    js = torch.stack([c["J"] for c in cands], -1)
    ds = torch.tensor([FAMILIES[c["fi"]].manifold_dim for c in cands], dtype=torch.float32, device=js.device)
    ps = torch.stack([c["p"] for c in cands], -1)
    # noise estimate from the most general non-volumetric candidate (highest d < 3; ties: smallest J)
    gen_mask = ds < 3
    dmax = ds[gen_mask].max() if bool(gen_mask.any()) else torch.tensor(2.0)
    use = gen_mask & (ds == dmax)
    jg = js.masked_fill(~use, float("inf")).min(-1)
    p_g = torch.gather(ps, -1, jg.indices.unsqueeze(-1)).squeeze(-1)
    dof = ((3 - dmax) * n - p_g).clamp_min(1.0)
    eps2 = (jg.values / dof).clamp_min(1e-8)
    return js + 2 * (ds.unsqueeze(0) * n.unsqueeze(-1) + ps) * eps2.unsqueeze(-1)


def select(probs: torch.Tensor, cand_fi: list[int], aic: torch.Tensor | None, rule: str) -> torch.Tensor:
    """Index into cand_fi chosen per sample. probs: (B, 13) calibrated; aic: (B, n_cands)."""
    pc = probs[:, cand_fi]
    if rule == "classifier" or aic is None:
        return pc.argmax(-1)
    if rule == "gaic":
        return aic.argmin(-1)
    if rule == "hybrid":
        ok = pc >= 0.1 * pc.max(-1, keepdim=True).values
        return aic.masked_fill(~ok, float("inf")).argmin(-1)
    raise ValueError(rule)
