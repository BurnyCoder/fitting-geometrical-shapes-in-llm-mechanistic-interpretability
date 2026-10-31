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
    """Haar-uniform random rotations via QR of a Gaussian matrix (Mezzadri 2007, sign-corrected)."""
    a = torch.randn(n, 3, 3, generator=generator, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    # Multiply columns by sign(diag(R)) so Q is Haar-distributed (Mezzadri, arXiv math-ph/0609050).
    d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d.unsqueeze(-2)
    # Flip the last column where det = -1 to land in SO(3) (proper rotations only).
    det = torch.linalg.det(q)
    q[..., :, 2] = q[..., :, 2] * det.sign().unsqueeze(-1)
    return q


def nearest_orthogonal(m: torch.Tensor) -> torch.Tensor:
    """Project (..., 3, 3) matrices onto O(3) (polar factor U V^T; may be a reflection)."""
    u, _, vh = torch.linalg.svd(m)
    return u @ vh


def nearest_rotation(m: torch.Tensor) -> torch.Tensor:
    """Project (..., 3, 3) matrices onto SO(3) (orthogonal Procrustes via SVD with det correction)."""
    u, _, vh = torch.linalg.svd(m)
    d = torch.sign(torch.linalg.det(u @ vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    corr = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    return u @ corr @ vh
