"""1D shape families: line segment, circle/arc, closed ellipse, helix, ringing (PCA-horseshoe) curve.

Global context: curves are the most frequently reported neural geometries (circles for days/months,
helices for positions/counts, lines for temperature/age). Each family defines its local curve around
the origin in the frame (e1, e2, e3); base.ShapeFamily handles pose, encoding and transforms.
Empirical anchors per family are listed in src/neurogeom/catalog/catalog.yaml.
"""

from __future__ import annotations

import math

import torch

from neurogeom.shapes.base import TWO_PI, ShapeFamily, Spec, dominant_positive, frame_from_axis


class Line(ShapeFamily):
    """Segment c + t*L*e1, t in [-1, 1]. Anchor: temperature/age lines (Bhalla et al. 2026)."""

    name, coarse, manifold_dim = "line", "1d", 1
    specs = (Spec("half_length", "pos", "L", 1e-3),)

    @property
    def n_free(self):
        return 3 + 2 + 1  # center, direction (2 dof), half-length

    def local(self, p, u):
        t = 2.0 * u[..., 0] - 1.0
        z = torch.zeros_like(t)
        return torch.stack([p["half_length"].unsqueeze(-1) * t, z, z], -1), torch.ones_like(t)

    def sample_local(self, b, gen, device):
        return {"half_length": torch.ones(b, device=device)}

    def symmetries(self, p):
        return [self.flip_frame(p, torch.ones_like(p["half_length"], dtype=torch.bool), (-1, -1, 1))]

    def canonicalize(self, p):
        e1 = p["rot"][..., 0] * dominant_positive(p["rot"][..., 0]).unsqueeze(-1)
        # e2/e3 are irrelevant for a line; fix them deterministically via frame_from_axis on e1.
        f = frame_from_axis(e1)                     # columns (a, b, e1)
        out = dict(p)
        out["rot"] = torch.stack([e1, f[..., 0], f[..., 1]], -1)
        return out

    def aux(self, p):
        return p["rot"][..., 0], torch.log(p["half_length"])

    def describe(self, p, i):
        c, d, l = p["center"][i], p["rot"][i, :, 0], float(p["half_length"][i])
        return {"family": self.name, "center": c.tolist(), "direction": d.tolist(), "half_length": l,
                "endpoints": [(c - l * d).tolist(), (c + l * d).tolist()]}


class Arc(ShapeFamily):
    """Circle or arc: c + r(cos t e1 + sin t e2), t in [-D/2, D/2]; e1 bisects the arc, e3 = normal.
    Anchors: days/months (Engels et al. 2024), grokking clock circles (Nanda et al. 2023)."""

    name, coarse, manifold_dim = "arc", "1d", 1
    specs = (Spec("radius", "pos", "L", 1e-3), Spec("extent_frac", "unit", "1", 0.0, 1.0))
    FULL = 0.98  # extent fraction above which the arc is reported as a closed circle

    @property
    def n_free(self):
        return 3 + 3 + 2  # center, frame (3 dof), radius, extent

    def local(self, p, u):
        t = TWO_PI * p["extent_frac"].unsqueeze(-1) * (u[..., 0] - 0.5)
        r = p["radius"].unsqueeze(-1)
        return torch.stack([r * torch.cos(t), r * torch.sin(t), torch.zeros_like(t)], -1), torch.ones_like(t)

    def sample_local(self, b, gen, device):
        frac = self.uniform(b, 1 / 6, 1.0, gen, device)
        full = torch.rand(b, generator=gen, device=device) < 0.4          # full circles are common in real data
        frac = torch.where(full, torch.full_like(frac, 0.999), frac)
        return {"radius": torch.ones(b, device=device), "extent_frac": frac}

    def margin_ok(self, p, sigma):
        half = math.pi * p["extent_frac"]
        sagitta = p["radius"] * (1 - torch.cos(half))
        chord = 2 * p["radius"] * torch.sin(half).abs()
        rms = self.rms_radius(p)
        return sagitta >= torch.maximum(0.1 * chord, 3 * sigma * rms)

    def symmetries(self, p):
        m = torch.ones_like(p["radius"], dtype=torch.bool)
        return [self.flip_frame(p, m, (1, -1, -1))]

    def canonicalize(self, p):
        e3 = p["rot"][..., 2]
        p = self.flip_frame(p, dominant_positive(e3) < 0, (1, -1, -1))   # normal sign via the arc's own symmetry
        closed = p["extent_frac"] >= self.FULL
        out = dict(p)
        out["rot"] = torch.where(closed.view(-1, 1, 1), frame_from_axis(p["rot"][..., 2]), p["rot"])
        return out

    def aux(self, p):
        return p["rot"][..., 2], torch.log(p["radius"])

    def describe(self, p, i):
        frac = float(p["extent_frac"][i])
        closed = frac >= self.FULL
        return {"family": self.name, "center": p["center"][i].tolist(), "normal": p["rot"][i, :, 2].tolist(),
                "radius": float(p["radius"][i]), "extent_rad": TWO_PI if closed else TWO_PI * frac, "closed": closed,
                "bisector": None if closed else p["rot"][i, :, 0].tolist()}


class Ellipse(ShapeFamily):
    """Closed ellipse c + a cos t e1 + b sin t e2, a >= b (b = a * ratio). Anchor: distorted circles."""

    name, coarse, manifold_dim = "ellipse", "1d", 1
    specs = (Spec("a", "pos", "L", 1e-3), Spec("ratio", "unit", "1", 0.0, 1.0))

    @property
    def n_free(self):
        return 3 + 3 + 2

    def local(self, p, u):
        t = TWO_PI * u[..., 0]
        a, b = p["a"].unsqueeze(-1), (p["a"] * p["ratio"]).unsqueeze(-1)
        pts = torch.stack([a * torch.cos(t), b * torch.sin(t), torch.zeros_like(t)], -1)
        speed = torch.sqrt((a * torch.sin(t)) ** 2 + (b * torch.cos(t)) ** 2 + 1e-12)   # arc-length element
        return pts, speed

    def natural_weights(self, p, u, w):
        return torch.ones_like(w)  # a linearly distorted circle is sampled uniformly in its angle

    def sample_local(self, b, gen, device):
        return {"a": torch.ones(b, device=device), "ratio": self.uniform(b, 0.2, 0.85, gen, device)}

    def margin_ok(self, p, sigma):
        return p["ratio"] <= 1 - torch.clamp(3 * sigma, min=0.15)

    def symmetries(self, p):
        m = torch.ones_like(p["a"], dtype=torch.bool)
        return [self.flip_frame(p, m, s) for s in ((-1, -1, 1), (1, -1, -1), (-1, 1, -1))]

    def canonicalize(self, p):
        p = self.flip_frame(p, dominant_positive(p["rot"][..., 2]) < 0, (-1, 1, -1))
        p = self.flip_frame(p, dominant_positive(p["rot"][..., 0]) < 0, (-1, -1, 1))
        return p

    def aux(self, p):
        return p["rot"][..., 2], torch.log(p["a"])

    def describe(self, p, i):
        a = float(p["a"][i])
        return {"family": self.name, "center": p["center"][i].tolist(), "normal": p["rot"][i, :, 2].tolist(),
                "major_axis": p["rot"][i, :, 0].tolist(), "a": a, "b": a * float(p["ratio"][i])}


class Helix(ShapeFamily):
    """Helix c + r cos t e1 + r sin t e2 + (pitch t / 2pi) e3, t in [-pi*turns, pi*turns].
    Signed pitch = rise per full turn along e3 (its sign is the handedness). Anchors: GPT-2 W_pos
    (Yedidia 2023), line-break counting (Gurnee et al. 2025), years (Bhalla et al. 2026)."""

    name, coarse, manifold_dim = "helix", "1d", 1
    specs = (Spec("radius", "pos", "L", 1e-3), Spec("pitch", "real", "L"), Spec("turns", "pos", "1", 0.25))

    @property
    def n_free(self):
        return 3 + 3 + 3

    def local(self, p, u):
        t = math.pi * p["turns"].unsqueeze(-1) * (2 * u[..., 0] - 1)
        r, h = p["radius"].unsqueeze(-1), p["pitch"].unsqueeze(-1)
        return torch.stack([r * torch.cos(t), r * torch.sin(t), h * t / TWO_PI], -1), torch.ones_like(t)

    def sample_local(self, b, gen, device):
        sign = torch.where(torch.rand(b, generator=gen, device=device) < 0.5, -1.0, 1.0).to(device)
        return {"radius": torch.ones(b, device=device), "pitch": sign * self.uniform(b, 0.25, 3.0, gen, device),
                "turns": self.uniform(b, 0.75, 4.0, gen, device)}

    def margin_ok(self, p, sigma):
        height = p["pitch"].abs() * p["turns"]
        rms = self.rms_radius(p)
        return (height >= torch.maximum(0.5 * p["radius"], 3 * sigma * rms)) & (p["radius"] >= torch.maximum(0.1 * height, 3 * sigma * rms))

    def mirror_params(self, p):
        # y-mirror of the local helix (r cos t, -r sin t, h t / 2pi) = local helix with pitch -h at -t.
        out = dict(p)
        out["pitch"] = -p["pitch"]
        return out

    def symmetries(self, p):
        return [self.flip_frame(p, torch.ones_like(p["radius"], dtype=torch.bool), (1, -1, -1))]

    def canonicalize(self, p):
        return self.flip_frame(p, dominant_positive(p["rot"][..., 2]) < 0, (1, -1, -1))

    def aux(self, p):
        return p["rot"][..., 2], torch.log(p["radius"])

    def describe(self, p, i):
        d = super().describe(p, i)
        d.update({"axis": p["rot"][i, :, 2].tolist(), "phase_axis": p["rot"][i, :, 0].tolist(),
                  "handedness": "right" if float(p["pitch"][i]) > 0 else "left",
                  "height": abs(float(p["pitch"][i])) * float(p["turns"][i])})
        return d


class Ringing(ShapeFamily):
    """PCA 'ringing' / horseshoe curve c + sum_k a_k cos(k pi s) e_k, s in [0, 1], k = 1..3.
    This is what PCA of any smooth ordinal 1D variable produces (Novembre & Stephens 2008; Diaconis
    et al. 2008; Antognini & Sohl-Dickstein 2018), so it is the key confounder for helix/arc claims.
    Points are naturally uniform in s (dense near the ends), which is part of its signature."""

    name, coarse, manifold_dim = "ringing", "1d", 1
    specs = (Spec("a1", "pos", "L", 1e-3), Spec("a2", "pos", "L", 1e-4), Spec("a3", "real", "L"))

    @property
    def n_free(self):
        return 3 + 3 + 3

    def local(self, p, u):
        s = u[..., 0]
        a1, a2, a3 = (p[k].unsqueeze(-1) for k in ("a1", "a2", "a3"))
        pts = torch.stack([a1 * torch.cos(math.pi * s), a2 * torch.cos(2 * math.pi * s), a3 * torch.cos(3 * math.pi * s)], -1)
        speed = math.pi * torch.sqrt((a1 * torch.sin(math.pi * s)) ** 2 + (2 * a2 * torch.sin(2 * math.pi * s)) ** 2
                                     + (3 * a3 * torch.sin(3 * math.pi * s)) ** 2 + 1e-12)
        return pts, speed

    MIRROR = (1.0, 1.0, -1.0)   # z-mirror keeps a1, a2 > 0 and flips the sign of the (signed) a3 term

    def mirror_params(self, p):
        out = dict(p)
        out["a3"] = -p["a3"]
        return out

    def natural_weights(self, p, u, w):
        return torch.ones_like(w)

    def sample_local(self, b, gen, device):
        return {"a1": torch.ones(b, device=device), "a2": self.uniform(b, 0.25, 1.0, gen, device),
                "a3": self.uniform(b, -0.7, 0.7, gen, device)}

    def margin_ok(self, p, sigma):
        return p["a2"] >= torch.maximum(0.25 * p["a1"], 3 * sigma * self.rms_radius(p))

    def symmetries(self, p):
        # s -> 1 - s flips the signs of the k = 1 and k = 3 terms: a pi-rotation about e2.
        return [self.flip_frame(p, torch.ones_like(p["a1"], dtype=torch.bool), (-1, 1, -1))]

    def canonicalize(self, p):
        return self.flip_frame(p, dominant_positive(p["rot"][..., 0]) < 0, (-1, 1, -1))

    def aux(self, p):
        return p["rot"][..., 0], torch.log(p["a1"])
