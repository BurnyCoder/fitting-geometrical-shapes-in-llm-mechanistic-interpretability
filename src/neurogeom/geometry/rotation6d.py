"""Continuous 6D rotation representation and random rotations.

Global context: every shape family is posed by a rotation R in SO(3) whose columns (e1, e2, e3) are
the shape's local axes. Networks regress rotations poorly with Euler angles or quaternions because
those representations are discontinuous; Zhou et al. 2019 show that the 6D representation (first two
columns, orthonormalised by Gram-Schmidt) is continuous and regresses well.
Source: Zhou, Barnes, Lu, Yang, Li, "On the Continuity of Rotation Representations in Neural
Networks", CVPR 2019, Appendix B, https://arxiv.org/abs/1812.07035
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Map (..., 6) -> (..., 3, 3) rotation matrices; columns are the frame axes (e1, e2, e3).

    Gram-Schmidt as in Zhou et al. 2019, Appendix B: b1 = N(a1), b2 = N(a2 - (b1.a2) b1), b3 = b1 x b2.
    """
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)                      # first axis: normalised a1
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1              # remove the b1 component of a2
    b2 = F.normalize(b2, dim=-1, eps=1e-8)                      # second axis: orthonormal to b1
    b3 = torch.cross(b1, b2, dim=-1)                            # third axis: right-handed => det = +1
    return torch.stack([b1, b2, b3], dim=-1)                    # stack as columns


def matrix_to_rot6d(r: torch.Tensor) -> torch.Tensor:
    """Inverse map used to initialise raw parameters from a known rotation: first two columns."""
    return torch.cat([r[..., :, 0], r[..., :, 1]], dim=-1)


def random_rotation(n: int, generator: torch.Generator | None = None, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Haar-uniform random rotations from normalised 4D Gaussians = uniform unit quaternions
    (Shoemake 1992, "Uniform random rotations", Graphics Gems III). Closed form, no batched QR: batched
    torch.linalg.qr on CUDA launches one kernel per matrix and dominated the generator's run time."""
    q = torch.randn(n, 4, generator=generator, device=device, dtype=dtype)
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    # standard unit-quaternion -> rotation-matrix formula
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def det3(m: torch.Tensor) -> torch.Tensor:
    """Determinant of (..., 3, 3) matrices via the scalar triple product (no LU / cusolver call)."""
    return (torch.cross(m[..., :, 0], m[..., :, 1], dim=-1) * m[..., :, 2]).sum(-1)


def nearest_orthogonal(m: torch.Tensor) -> torch.Tensor:
    """Project (..., 3, 3) matrices onto O(3) (polar factor U V^T; may be a reflection).
    Tiny batched SVDs are much faster on the CPU than through cuSOLVER, so they run there."""
    u, _, vh = torch.linalg.svd(m.detach().cpu().double())
    return (u @ vh).to(m.device, m.dtype)


def nearest_rotation(m: torch.Tensor) -> torch.Tensor:
    """Project (..., 3, 3) matrices onto SO(3) (orthogonal Procrustes via SVD with det correction)."""
    u, _, vh = torch.linalg.svd(m)
    d = torch.sign(torch.linalg.det(u @ vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    corr = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    return u @ corr @ vh
