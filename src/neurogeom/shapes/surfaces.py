"""2D (and volumetric) shape families: disc, sphere/cap, paraboloid sheet, cone/frustum, torus, blob.

Global context: surface-like neural geometries are rarer but reported - a paraboloid for colors
(Bhalla et al. 2026), a sphere for cities (Tiblias et al. 2025), discs/tori in modular addition, a cone
from anisotropy/intensity (Ethayarajh 2019; Engels et al. 2024). The blob is the null class: a
Gaussian cloud without low-dimensional structure (what matched-covariance controls should map to).
Surfaces use a (side x side) grid on [0,1]^2 mapped so that equal grid cells have (nearly) equal area,
with residual non-uniformity absorbed by the weights (area element).
"""

from __future__ import annotations

import math

import torch

from neurogeom.shapes.base import TWO_PI, ShapeFamily, Spec, dominant_positive, frame_from_axis
from neurogeom.shapes.curves import Ellipse


class Disc(ShapeFamily):
    """Filled ellipse c + rho (a cos t e1 + b sin t e2), rho in [0,1], b = a * ratio.
    Anchors: 'pizza' discs in modular addition (Zhong et al. 2023), planar grids (Park et al. 2025)."""

    name, coarse, manifold_dim, u_dim = "disc", "2d", 2, 2
    specs = (Spec("a", "pos", "L", 1e-3), Spec("ratio", "unit", "1", 0.0, 1.0))

    @property
    def n_free(self):
        return 3 + 3 + 2

    def local(self, p, u):
        rho, t = torch.sqrt(u[..., 0]), TWO_PI * u[..., 1]          # sqrt => uniform area density
        a, b = p["a"].unsqueeze(-1), (p["a"] * p["ratio"]).unsqueeze(-1)
        return torch.stack([a * rho * torch.cos(t), b * rho * torch.sin(t), torch.zeros_like(t)], -1), torch.ones_like(t)

    def sample_local(self, b, gen, device):
        return {"a": torch.ones(b, device=device), "ratio": self.uniform(b, 0.25, 1.0, gen, device)}

    def margin_ok(self, p, sigma):
        return p["a"] * p["ratio"] >= 6 * sigma * self.rms_radius(p)

    # A filled ellipse has the same frame symmetries as its boundary ellipse.
    symmetries = Ellipse.symmetries
    canonicalize = Ellipse.canonicalize

    def aux(self, p):
        return p["rot"][..., 2], torch.log(p["a"])


class Sphere(ShapeFamily):
    """Sphere or polar cap of half-angle alpha around pole e3: points c + r(sin th cos ph, sin th sin ph, cos th),
    th in [0, alpha]. Anchor: cities on a sphere (Tiblias et al. 2025)."""

    name, coarse, manifold_dim, u_dim = "sphere", "2d", 2, 2
    specs = (Spec("radius", "pos", "L", 1e-3), Spec("cap_frac", "unit", "1", 0.0, 1.0))
    FULL = 0.98

    @property
    def n_free(self):
        return 3 + 2 + 2  # center, pole (2 dof), radius, cap angle

    def local(self, p, u):
        alpha = math.pi * p["cap_frac"].unsqueeze(-1)
        z = 1 - (1 - torch.cos(alpha)) * u[..., 0]                    # uniform in z  <=> uniform area (Archimedes)
        sxy = torch.sqrt((1 - z * z).clamp_min(1e-9))
        ph = TWO_PI * u[..., 1]
        r = p["radius"].unsqueeze(-1)
        return torch.stack([r * sxy * torch.cos(ph), r * sxy * torch.sin(ph), r * z], -1), torch.ones_like(z)

    def sample_local(self, b, gen, device):
        frac = self.uniform(b, 70 / 180, 1.0, gen, device)
        full = torch.rand(b, generator=gen, device=device) < 0.3
        return {"radius": torch.ones(b, device=device), "cap_frac": torch.where(full, torch.full_like(frac, 0.999), frac)}

    def canonicalize(self, p):
        out = dict(p)
        closed = p["cap_frac"] >= self.FULL
        eye = torch.eye(3, dtype=p["rot"].dtype, device=p["rot"].device).expand_as(p["rot"])
        out["rot"] = torch.where(closed.view(-1, 1, 1), eye, frame_from_axis(p["rot"][..., 2]))
        return out

    def symmetries(self, p):
        # rotation about the pole by an arbitrary angle (here 1 rad) leaves the cap unchanged
        c, s = math.cos(1.0), math.sin(1.0)
        rz = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]], dtype=p["rot"].dtype, device=p["rot"].device)
        out = dict(p)
        out["rot"] = p["rot"] @ rz
        return [out]

    def aux(self, p):
        return None, torch.log(p["radius"])

    def describe(self, p, i):
        closed = float(p["cap_frac"][i]) >= self.FULL
        return {"family": self.name, "center": p["center"][i].tolist(), "radius": float(p["radius"][i]),
                "closed": closed, "pole": None if closed else p["rot"][i, :, 2].tolist(),
                "cap_half_angle_rad": math.pi if closed else math.pi * float(p["cap_frac"][i])}


class Paraboloid(ShapeFamily):
    """Quadric sheet z = (k1 x^2 + k2 y^2)/2 over the disc x^2 + y^2 <= rho^2 in the frame (e1, e2, e3),
    with k1 > 0 and |k2| <= k1 (bowl: k2 > 0, trough: k2 ~ 0, saddle: k2 < 0).
    Anchor: colors on a paraboloid with circular hue and lightness (Bhalla et al. 2026)."""

    name, coarse, manifold_dim, u_dim = "paraboloid", "2d", 2, 2
    specs = (Spec("k1", "pos", "K", 1e-3), Spec("k_ratio", "unit", "1", -1.0, 1.0), Spec("rho", "pos", "L", 1e-3))

    @property
    def n_free(self):
        return 3 + 3 + 3

    def local(self, p, u):
        rr = p["rho"].unsqueeze(-1) * torch.sqrt(u[..., 0])
        t = TWO_PI * u[..., 1]
        x, y = rr * torch.cos(t), rr * torch.sin(t)
        k1 = p["k1"].unsqueeze(-1)
        k2 = k1 * p["k_ratio"].unsqueeze(-1)
        z = 0.5 * (k1 * x * x + k2 * y * y)
        w = torch.sqrt(1 + (k1 * x) ** 2 + (k2 * y) ** 2)            # area element of a graph surface
        return torch.stack([x, y, z], -1), w

    def sample_local(self, b, gen, device):
        return {"rho": torch.ones(b, device=device), "k1": self.uniform(b, 0.6, 3.0, gen, device),
                "k_ratio": self.uniform(b, -1.0, 1.0, gen, device)}

    def margin_ok(self, p, sigma):
        kr = p["k1"] * p["rho"]
        sag = 0.5 * p["k1"] * p["rho"] ** 2
        ok_sphere = (p["k_ratio"] <= 0.7) | (kr >= 1.5)              # near-umbilic bowls must be deep to differ from a cap
        return (kr >= 0.6) & ok_sphere & (sag >= 3 * sigma * self.rms_radius(p))

    def symmetries(self, p):
        return [self.flip_frame(p, torch.ones_like(p["k1"], dtype=torch.bool), (-1, -1, 1))]

    def canonicalize(self, p):
        return self.flip_frame(p, dominant_positive(p["rot"][..., 0]) < 0, (-1, -1, 1))

    def aux(self, p):
        return p["rot"][..., 2], torch.log(p["rho"])

    def describe(self, p, i):
        d = super().describe(p, i)
        kr = float(p["k_ratio"][i])
        d.update({"k2": float(p["k1"][i]) * kr, "normal": p["rot"][i, :, 2].tolist(),
                  "kind": "bowl" if kr > 0.2 else ("saddle" if kr < -0.2 else "trough")})
        return d


class Cone(ShapeFamily):
    """Lateral surface of a frustum around axis e3: radius r(h) = r_bot + dr (h + H) / (2H), h in [-H, H].
    dr = 0 gives a cylinder, r_bot = 0 a cone. Anchors: anisotropy cone (Ethayarajh 2019), the
    intensity-scaled circle cone of Engels et al. 2024."""

    name, coarse, manifold_dim, u_dim = "cone", "2d", 2, 2
    specs = (Spec("r_bot", "pos", "L", 0.0), Spec("dr", "pos", "L", 0.0), Spec("half_height", "pos", "L", 1e-3))

    @property
    def n_free(self):
        return 3 + 2 + 3

    def local(self, p, u):
        ph, s = TWO_PI * u[..., 0], u[..., 1]
        hh = p["half_height"].unsqueeze(-1)
        r = p["r_bot"].unsqueeze(-1) + p["dr"].unsqueeze(-1) * s
        return torch.stack([r * torch.cos(ph), r * torch.sin(ph), hh * (2 * s - 1)], -1), r + 1e-6

    def sample_local(self, b, gen, device):
        r_bot = torch.where(torch.rand(b, generator=gen, device=device) < 0.3, torch.zeros(b, device=device),
                            self.uniform(b, 0.0, 1.5, gen, device))
        dr = torch.where(torch.rand(b, generator=gen, device=device) < 0.25, torch.full((b,), 1e-3, device=device),
                         self.uniform(b, 0.0, 1.5, gen, device))
        return {"r_bot": r_bot + 1e-3, "dr": dr, "half_height": torch.ones(b, device=device)}

    def margin_ok(self, p, sigma):
        rbar = p["r_bot"] + 0.5 * p["dr"]
        rms = self.rms_radius(p)
        return (rbar >= torch.maximum(0.25 * p["half_height"], 3 * sigma * rms)) & (2 * p["half_height"] >= 0.5 * rbar)

    def canonicalize(self, p):
        out = dict(p)
        e3 = p["rot"][..., 2]
        cyl = p["dr"] < 0.02 * (p["r_bot"] + p["dr"])                  # a cylinder is also symmetric under axis flips
        e3 = torch.where(cyl.unsqueeze(-1), e3 * dominant_positive(e3).unsqueeze(-1), e3)
        out["rot"] = frame_from_axis(e3)
        return out

    def symmetries(self, p):
        return Sphere.symmetries(self, p)

    def aux(self, p):
        return p["rot"][..., 2], torch.log(p["r_bot"] + 0.5 * p["dr"] + 1e-6)


class Torus(ShapeFamily):
    """Torus ((R + r cos v) cos u, (R + r cos v) sin u, r sin v) around axis e3, r = R * ratio.
    Anchors: grid-cell torus (Gardner et al. 2022), modular-addition representations (2512.25060)."""

    name, coarse, manifold_dim, u_dim = "torus", "2d", 2, 2
    specs = (Spec("R", "pos", "L", 1e-3), Spec("ratio", "unit", "1", 0.0, 0.95))

    @property
    def n_free(self):
        return 3 + 2 + 2

    def local(self, p, u):
        a, v = TWO_PI * u[..., 0], TWO_PI * u[..., 1]
        big, small = p["R"].unsqueeze(-1), (p["R"] * p["ratio"]).unsqueeze(-1)
        ring = big + small * torch.cos(v)
        return torch.stack([ring * torch.cos(a), ring * torch.sin(a), small * torch.sin(v)], -1), ring

    def sample_local(self, b, gen, device):
        return {"R": torch.ones(b, device=device), "ratio": self.uniform(b, 0.15, 0.7, gen, device)}

    def margin_ok(self, p, sigma):
        return p["R"] * p["ratio"] >= 3 * sigma * self.rms_radius(p)

    def canonicalize(self, p):
        out = dict(p)
        e3 = p["rot"][..., 2]
        out["rot"] = frame_from_axis(e3 * dominant_positive(e3).unsqueeze(-1))
        return out

    def symmetries(self, p):
        return Sphere.symmetries(self, p) + [self.flip_frame(p, torch.ones_like(p["R"], dtype=torch.bool), (1, -1, -1))]

    def aux(self, p):
        return p["rot"][..., 2], torch.log(p["R"])


class Blob(ShapeFamily):
    """Null class: anisotropic Gaussian c + R diag(s1, s2, s3) z, z ~ N(0, I), s1 >= s2 >= s3.
    Its 'dense samples' are a fixed set of standard-normal draws, so Chamfer between two blobs is a
    smooth function of their parameters. Expected label of matched-covariance controls."""

    name, coarse, manifold_dim, u_dim = "blob", "3d", 3, 3
    specs = (Spec("s1", "pos", "L", 1e-3), Spec("r21", "unit", "1", 0.0, 1.0), Spec("r32", "unit", "1", 0.0, 1.0))
    _grid_cache: dict = {}

    def grid(self, m_curve, side, device):
        key = str(device)
        if key not in self._grid_cache:
            g = torch.Generator().manual_seed(1234)          # fixed normal set, identical for every call
            z = torch.randn(32, 3, generator=g)
            # Close the set under all 8 axis sign flips (the blob's frame symmetries and the mirror used for
            # reflected frames), so equivalent parameterisations give exactly the same sample set: 32 x 8 = 256.
            signs = torch.tensor([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)], dtype=z.dtype)
            self._grid_cache[key] = (z.unsqueeze(0) * signs.unsqueeze(1)).reshape(-1, 3).to(device)
        return self._grid_cache[key]

    def random_u(self, b, g, gen, device):
        return torch.randn(b, g, 3, generator=gen, device=device)

    def local(self, p, u):
        s1 = p["s1"].unsqueeze(-1)
        s2 = s1 * p["r21"].unsqueeze(-1)
        s3 = s2 * p["r32"].unsqueeze(-1)
        return torch.stack([s1 * u[..., 0], s2 * u[..., 1], s3 * u[..., 2]], -1), torch.ones_like(u[..., 0])

    def sample_local(self, b, gen, device):
        return {"s1": torch.ones(b, device=device), "r21": self.uniform(b, 0.25, 1.0, gen, device),
                "r32": self.uniform(b, 0.25, 1.0, gen, device)}

    def symmetries(self, p):
        m = torch.ones_like(p["s1"], dtype=torch.bool)
        return [self.flip_frame(p, m, s) for s in ((-1, -1, 1), (1, -1, -1))]

    def canonicalize(self, p):
        p = self.flip_frame(p, dominant_positive(p["rot"][..., 0]) < 0, (-1, 1, -1))
        p = self.flip_frame(p, dominant_positive(p["rot"][..., 1]) < 0, (1, -1, -1))
        return p

    def aux(self, p):
        return p["rot"][..., 0], torch.log(p["s1"])

    def describe(self, p, i):
        s1 = float(p["s1"][i])
        s2 = s1 * float(p["r21"][i])
        return {"family": self.name, "center": p["center"][i].tolist(), "axes": p["rot"][i].T.tolist(),
                "sigmas": [s1, s2, s2 * float(p["r32"][i])]}
