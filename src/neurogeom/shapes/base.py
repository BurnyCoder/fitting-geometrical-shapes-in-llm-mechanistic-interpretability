"""ShapeFamily: the common interface every shape family implements.

Global context: a shape family turns an unconstrained raw vector (what a network head or an optimiser
produces) into constrained parameters (`decode`), parameters into dense differentiable samples
(`points`, used by the Chamfer loss, the residual and refinement), draws random ground-truth
parameters with label margins (`sample_valid`, used by the synthetic generator), maps parameters
through similarity transforms (`transform`), and picks a unique representative among equivalent
parameterisations (`canonicalize`, so the output "specifies the shape uniquely").

Posed families (all except clusters/polytope) share the layout raw = [center(3), rot6d(6), intrinsic]
and place a local shape (defined around the origin in the frame e1, e2, e3) at `center` with rotation
`rot` (columns = e1, e2, e3). Intrinsic parameters are declared by a small spec so encode/decode and
similarity transforms are generic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from neurogeom.geometry.rotation6d import det3, matrix_to_rot6d, random_rotation, rot6d_to_matrix

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Spec:
    """One intrinsic parameter: name, constraint kind, physical dimension, bounds.

    kind: 'pos' -> softplus(x) + lo (lengths, curvatures); 'unit' -> lo + (hi - lo) * sigmoid(x)
          (ratios, fractions of a full angle); 'real' -> x (signed lengths such as helix pitch).
    dim:  'L' length (scales with s), 'K' curvature (scales with 1/s), '1' dimensionless.
    """

    name: str
    kind: str
    dim: str
    lo: float = 0.0
    hi: float = 1.0


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    # log(exp(y) - 1), computed stably; inverse of F.softplus for y > 0.
    y = y.clamp_min(1e-6)
    return y + torch.log(-torch.expm1(-y))


def dominant_positive(v: torch.Tensor) -> torch.Tensor:
    """+1/-1 per row so that the largest-|.| component of v becomes positive (deterministic sign rule)."""
    idx = v.abs().argmax(-1, keepdim=True)
    s = torch.sign(torch.gather(v, -1, idx)).squeeze(-1)
    return torch.where(s == 0, torch.ones_like(s), s)


def frame_from_axis(e3: torch.Tensor) -> torch.Tensor:
    """Canonical right-handed frame with given third axis: e1 = projection of the x-axis (or y if
    e3 is nearly parallel to x) onto the plane orthogonal to e3. Used where the shape is symmetric
    under rotations about e3 (full circle, sphere, cone, torus), to make the output unique."""
    ref = torch.zeros_like(e3)
    use_y = e3[..., 0].abs() > 0.9
    ref[..., 0] = torch.where(use_y, 0.0, 1.0)
    ref[..., 1] = torch.where(use_y, 1.0, 0.0)
    e1 = F.normalize(ref - (ref * e3).sum(-1, keepdim=True) * e3, dim=-1)
    e2 = torch.cross(e3, e1, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def merge_where(mask: torch.Tensor, a: dict, b: dict) -> dict:
    """Per-sample select between two parameter dicts (b where mask, else a)."""
    out = {}
    for k in a:
        m = mask.view(-1, *([1] * (a[k].dim() - 1)))
        out[k] = torch.where(m, b[k], a[k])
    return out


class ShapeFamily:
    """Base class for posed shape families. Subclasses define specs, local geometry and margins."""

    name: str = "base"
    coarse: str = "1d"        # '0d' point set, '1d' curve, '2d' surface, '3d' volume
    manifold_dim: int = 1     # used by geometric AIC
    u_dim: int = 1            # dimension of the sampling-parameter domain
    specs: tuple[Spec, ...] = ()
    posed: bool = True

    # ------------------------------------------------------------------ sizes
    @property
    def n_raw(self) -> int:
        return 9 + len(self.specs)

    @property
    def n_free(self) -> int:
        """Number of free shape parameters (for geometric AIC); override where the pose is degenerate."""
        return 6 + len(self.specs)

    # ------------------------------------------------------------------ raw <-> constrained
    def decode(self, raw: torch.Tensor, select: torch.Tensor | None = None) -> dict:
        p = {"center": raw[:, 0:3], "rot": rot6d_to_matrix(raw[:, 3:9])}
        for i, s in enumerate(self.specs):
            x = raw[:, 9 + i]
            if s.kind == "pos":
                p[s.name] = F.softplus(x) + s.lo
            elif s.kind == "unit":
                p[s.name] = s.lo + (s.hi - s.lo) * torch.sigmoid(x)
            else:
                p[s.name] = x
        return p

    def encode(self, p: dict) -> torch.Tensor:
        cols = [p["center"], matrix_to_rot6d(p["rot"])]
        for s in self.specs:
            v = p[s.name]
            if s.kind == "pos":
                x = _inv_softplus(v - s.lo)
            elif s.kind == "unit":
                t = ((v - s.lo) / (s.hi - s.lo)).clamp(1e-4, 1 - 1e-4)
                x = torch.log(t) - torch.log1p(-t)
            else:
                x = v
            cols.append(x.unsqueeze(-1))
        return torch.cat(cols, -1)

    # ------------------------------------------------------------------ sampling domain
    def grid(self, m_curve: int, side: int, device) -> torch.Tensor:
        """Fixed sampling grid in [0,1]^u_dim (midpoint rule), shared by all samples of a batch."""
        if self.u_dim == 1:
            return ((torch.arange(m_curve, device=device) + 0.5) / m_curve).unsqueeze(-1)
        g = (torch.arange(side, device=device) + 0.5) / side
        a, b = torch.meshgrid(g, g, indexing="ij")
        return torch.stack([a.reshape(-1), b.reshape(-1)], -1)

    def random_u(self, b: int, g: int, gen: torch.Generator, device) -> torch.Tensor:
        """Random sampling parameters (B, G, u_dim) for generating observed points."""
        return torch.rand(b, g, self.u_dim, generator=gen, device=device)

    # ------------------------------------------------------------------ geometry
    def local(self, p: dict, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Local-frame points (B, M, 3) and geometric weights (B, M) ~ length/area element."""
        raise NotImplementedError

    def natural_weights(self, p: dict, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """How points are naturally distributed on the shape when generating data (default: uniform
        w.r.t. length/area). Families whose real-world sampling is uniform in a parameter override."""
        return w

    def points(self, p: dict, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame points: center + R @ local. u is (M, d) or (B, M, d)."""
        if u.dim() == 2:
            u = u.unsqueeze(0).expand(p["center"].shape[0], -1, -1)
        loc, w = self.local(p, u)
        pts = p["center"].unsqueeze(1) + loc @ p["rot"].transpose(-1, -2)
        return pts, w

    def dense(self, p: dict, cfg_dense: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense samples on the fixed grid (used for losses, residuals and refinement)."""
        u = self.grid(cfg_dense["curve"], cfg_dense["surface_side"], p["center"].device)
        return self.points(p, u)

    def rms_radius(self, p: dict, cfg_dense: dict | None = None) -> torch.Tensor:
        """Weighted RMS distance of the shape's points from their weighted centroid, (B,)."""
        pts, w = self.dense(p, cfg_dense or {"curve": 128, "surface_side": 16, "blob": 256})
        w = w / w.sum(-1, keepdim=True)
        mu = (pts * w.unsqueeze(-1)).sum(1, keepdim=True)
        return torch.sqrt((((pts - mu) ** 2).sum(-1) * w).sum(-1))

    # ------------------------------------------------------------------ random ground truth
    def sample_local(self, b: int, gen: torch.Generator, device) -> dict:
        """Random intrinsic parameters (subclass); pose is added by sample_params."""
        raise NotImplementedError

    def margin_ok(self, p: dict, sigma: torch.Tensor) -> torch.Tensor:
        """Label margin: True where the shape is not (nearly) a member of a simpler family at noise sigma."""
        return torch.ones(p["center"].shape[0], dtype=torch.bool, device=p["center"].device)

    def sample_params(self, b: int, gen: torch.Generator, device) -> dict:
        p = self.sample_local(b, gen, device)
        p["center"] = torch.zeros(b, 3, device=device)
        p["rot"] = random_rotation(b, generator=gen, device=device)
        return p

    def sample_valid(self, b: int, gen: torch.Generator, sigma: torch.Tensor, device, max_iter: int = 50) -> dict:
        """Rejection sampling until every sample satisfies its margin (sigma: (B,) relative noise)."""
        out = self.sample_params(b, gen, device)
        need = ~self.margin_ok(out, sigma)
        for _ in range(max_iter):
            if not bool(need.any()):
                break
            cand = self.sample_params(b, gen, device)
            take = need & self.margin_ok(cand, sigma)
            out = merge_where(take, out, cand)
            need = need & ~take
        return out

    # ------------------------------------------------------------------ similarity transforms
    # A local mirror F (det -1) that maps the local shape onto itself up to `mirror_params`. Used when the
    # transform q is a reflection (possible after PCA of a 3D subspace of R^D, where handedness is not
    # defined): q R local = (q R F)(F local), and q R F is a proper rotation again.
    MIRROR = (1.0, -1.0, 1.0)

    def mirror_params(self, p: dict) -> dict:
        """Intrinsic parameters of F(local shape); identity for shapes symmetric under F."""
        return p

    def transform(self, p: dict, q: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> dict:
        """Parameters of the image of the shape under x -> s * q @ x + t (q orthogonal, s > 0)."""
        out = dict(p)
        improper = det3(q) < 0
        if bool(improper.any()):
            f = torch.tensor(self.MIRROR, dtype=q.dtype, device=q.device)
            out = merge_where(improper, out, self.mirror_params(dict(p)))
            out["rot"] = torch.where(improper.view(-1, 1, 1), p["rot"] * f, p["rot"])   # R F (F diagonal: scale columns)
        out["center"] = s.unsqueeze(-1) * (q @ p["center"].unsqueeze(-1)).squeeze(-1) + t
        out["rot"] = q @ out["rot"]
        for spec in self.specs:
            if spec.dim == "L":
                out[spec.name] = out[spec.name] * s
            elif spec.dim == "K":
                out[spec.name] = out[spec.name] / s
        return out

    # ------------------------------------------------------------------ symmetries & canonical form
    def symmetries(self, p: dict) -> list[dict]:
        """Equivalent parameterisations of the same point set (used by tests of `canonicalize`)."""
        return []

    def canonicalize(self, p: dict) -> dict:
        return p

    def aux(self, p: dict) -> tuple[torch.Tensor | None, torch.Tensor]:
        """(unsigned axis or None, log of a characteristic length) for the auxiliary loss."""
        raise NotImplementedError

    def describe(self, p: dict, i: int) -> dict:
        """JSON-friendly canonical parameters of sample i (call on canonicalised params)."""
        d = {"family": self.name, "center": p["center"][i].tolist(), "axes": p["rot"][i].T.tolist()}
        for s in self.specs:
            d[s.name] = float(p[s.name][i])
        return d

    # ------------------------------------------------------------------ helpers for subclasses
    @staticmethod
    def flip_frame(p: dict, mask: torch.Tensor, signs: tuple[float, float, float]) -> dict:
        """Where mask, multiply frame columns by the given signs (product must be +1)."""
        out = dict(p)
        sv = torch.tensor(signs, dtype=p["rot"].dtype, device=p["rot"].device)
        flipped = p["rot"] * sv
        out["rot"] = torch.where(mask.view(-1, 1, 1), flipped, p["rot"])
        return out

    @staticmethod
    def uniform(b: int, lo: float, hi: float, gen: torch.Generator, device) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(b, generator=gen, device=device)
