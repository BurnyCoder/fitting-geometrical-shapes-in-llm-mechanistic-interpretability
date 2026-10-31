"""PCA-to-3D normalisation shared by synthetic and real point clouds.

Global context: the fitter's input is "points in the top-3 PCA directions of activations". To make
synthetic training data and real activations statistically identical at the input, both go through
exactly this transform: centre -> project onto the top-3 principal directions -> fix axis signs ->
(3D only) force a right-handed frame -> divide by the RMS radius. We never whiten, because whitening
would erase the aspect ratios that distinguish e.g. a circle from an ellipse or a line.
The transform is a similarity in 3D (rotation + translation + uniform scale), so shapes fitted in the
normalised frame map back exactly to the original coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class PCAFrame:
    """Everything needed to map between original coordinates (R^D) and the normalised 3D frame."""

    mean: np.ndarray            # (D,) centroid of the input cloud
    basis: np.ndarray           # (D, 3) principal directions (columns), sign-fixed
    scale: float                # RMS radius of the projected, centred cloud
    eigvals: np.ndarray         # (k,) top eigenvalues of the covariance (k = min(6, D, N)) for diagnostics
    total_var: float            # trace of the covariance (to report explained-variance ratios)
    extra: dict = field(default_factory=dict)

    @property
    def explained_ratio(self) -> np.ndarray:
        return self.eigvals / max(self.total_var, 1e-12)

    def to_dict(self) -> dict:
        return {"scale": float(self.scale), "eigvals": self.eigvals.tolist(), "total_var": float(self.total_var),
                "explained_ratio": self.explained_ratio.tolist()}


def _sign_fix(scores: torch.Tensor, basis: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip each principal axis so the third moment (skewness) of its scores is >= 0.

    Unlike sklearn's svd_flip (sign by largest loading), this rule depends only on the cloud's shape,
    so the normalised output is invariant to rotations of the input (up to measure-zero ties).
    """
    m3 = scores.pow(3)
    if mask is not None:
        m3 = m3 * mask.unsqueeze(-1)
    s = torch.sign(m3.sum(-2))                       # (..., 3)
    s = torch.where(s == 0, torch.ones_like(s), s)
    return scores * s.unsqueeze(-2), basis * s.unsqueeze(-2)


def normalize_batch(x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Batched PCA normalisation (torch, differentiable-free use). x: (B, N, D), mask: (B, N) bool.

    Returns y (B, N, 3) and a transform dict {mean (B,D), basis (B,D,3), scale (B,), eig (B,k)}, with
    y = ((x - mean) @ basis) / scale for valid points (padding rows are zeroed).
    """
    b, n, d = x.shape
    if mask is None:
        mask = torch.ones(b, n, dtype=torch.bool, device=x.device)
    mf = mask.to(x.dtype).unsqueeze(-1)                                   # (B, N, 1)
    cnt = mf.sum(1).clamp_min(1.0)                                         # (B, 1)
    mean = (x * mf).sum(1) / cnt                                           # masked centroid
    xc = (x - mean.unsqueeze(1)) * mf                                      # centred, padding zeroed
    cov = xc.transpose(1, 2) @ xc / cnt.unsqueeze(-1)                      # (B, D, D) population covariance
    if d <= 512:
        # float64 on CPU (real data, exactness); the input dtype on GPU, because consumer GPUs run FP64
        # at ~1/64 of FP32 throughput (an eigh of 160 x 256 x 256 took 0.25 s in FP64 on the RTX 5070).
        # Small matrices (D <= 16) are faster on the CPU than through cuSOLVER's batched eigh.
        small = d <= 16
        work = cov.detach().cpu().double() if (cov.device.type == "cpu" or small) else cov
        evals, evecs = torch.linalg.eigh(work)                             # ascending eigenvalues
        evals, evecs = evals.flip(-1).to(x.device, x.dtype), evecs.flip(-1).to(x.device, x.dtype)
    else:  # large D: SVD of the centred data is cheaper than eigh of a D x D covariance
        _, s, vh = torch.linalg.svd(xc, full_matrices=False)
        evals, evecs = (s ** 2) / cnt, vh.transpose(1, 2)
    k3 = min(3, d)
    basis = evecs[..., :k3]
    if k3 < 3:  # degenerate D < 3: pad with zero directions so the output is always 3D
        basis = torch.cat([basis, torch.zeros(b, d, 3 - k3, dtype=x.dtype, device=x.device)], -1)
    scores = xc @ basis
    scores, basis = _sign_fix(scores, basis, mask.to(x.dtype))
    if d == 3:  # force a right-handed frame so handedness (e.g. of a helix) is preserved
        det = (torch.cross(basis[..., 0], basis[..., 1], dim=-1) * basis[..., 2]).sum(-1)
        flip = torch.where(det < 0, -1.0, 1.0).to(x.dtype)
        basis = basis.clone()
        basis[..., 2] = basis[..., 2] * flip.unsqueeze(-1)
        scores = scores.clone()
        scores[..., 2] = scores[..., 2] * flip.unsqueeze(-1)
    rms = torch.sqrt(((scores ** 2).sum(-1) * mask).sum(-1) / cnt.squeeze(-1)).clamp_min(1e-12)
    y = scores / rms.view(b, 1, 1) * mask.unsqueeze(-1)
    kk = min(6, d)
    return y, {"mean": mean, "basis": basis, "scale": rms, "eig": evals[..., :kk], "total_var": evals.sum(-1)}


def apply_transform(x: torch.Tensor, t: dict) -> torch.Tensor:
    """Map other points in R^D (e.g. ground-truth dense samples) into the same normalised frame."""
    return ((x - t["mean"].unsqueeze(1)) @ t["basis"]) / t["scale"].view(-1, 1, 1)


def fit_frame(x: np.ndarray) -> tuple[np.ndarray, PCAFrame]:
    """NumPy entry point for one real point cloud (N, D) -> normalised (N, 3) and its PCAFrame."""
    xt = torch.as_tensor(np.asarray(x, dtype=np.float64)).unsqueeze(0)
    y, t = normalize_batch(xt)
    frame = PCAFrame(mean=t["mean"][0].numpy(), basis=t["basis"][0].numpy(), scale=float(t["scale"][0]),
                     eigvals=t["eig"][0].numpy(), total_var=float(t["total_var"][0]))
    return y[0].numpy(), frame


def to_pca_coords(y: np.ndarray, frame: PCAFrame) -> np.ndarray:
    """Normalised 3D coordinates -> un-normalised top-3 PCA coordinates (what the user plots)."""
    return np.asarray(y) * frame.scale


def to_original(y: np.ndarray, frame: PCAFrame) -> np.ndarray:
    """Normalised 3D coordinates -> original activation space R^D (inside the top-3 PCA subspace)."""
    return np.asarray(y) * frame.scale @ frame.basis.T + frame.mean
