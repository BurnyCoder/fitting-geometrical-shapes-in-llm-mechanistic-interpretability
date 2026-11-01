"""0D shape families: clusters (k = 2..6 centroids) and polytopes (vertex sets of six templates).

Global context: discrete geometries - two clusters for true vs false statements (Marks & Tegmark
2024), the uniform polytopes formed by features in superposition (Elhage et al. 2022: digons, triangles,
tetrahedra, pentagons, square antiprisms). Templates: the regular triangle, square, pentagon, tetrahedron and
octahedron, plus the square antiprism, which is uniform (all edges equal) but not regular. The "dense samples" of a point-set shape are its centroids
or vertices (padded to a fixed maximum with weight 0), so Chamfer between centroid sets is
permutation-invariant and needs no Hungarian matching. The unique output is the vertex/centroid list
sorted lexicographically (by PC1, then PC2, PC3), which removes permutation and symmetry-group
ambiguity in one step.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from neurogeom.geometry.rotation6d import det3, matrix_to_rot6d, random_rotation, rot6d_to_matrix
from neurogeom.shapes.base import ShapeFamily, _inv_softplus

K_VALUES = (2, 3, 4, 5, 6)
K_MAX = max(K_VALUES)


def _sort_points(pts: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort each sample's valid points lexicographically by (x, y, z); padding goes last."""
    key = pts[..., 0] * 1e6 + pts[..., 1] * 1e3 + pts[..., 2]       # coordinates are O(1) after normalisation
    key = key.masked_fill(~valid, float("inf"))
    idx = key.argsort(-1)
    return torch.gather(pts, 1, idx.unsqueeze(-1).expand_as(pts)), torch.gather(valid, 1, idx)


class Clusters(ShapeFamily):
    """k Gaussian clusters with centroids mu_1..mu_k and common spread. Raw layout:
    [k logits (5), centroids for every k (3 * (2+3+4+5+6) = 60), log-spread (1)]."""

    name, coarse, manifold_dim, posed = "clusters", "0d", 0, False
    _offsets = {k: 5 + 3 * sum(K_VALUES[: K_VALUES.index(k)]) for k in K_VALUES}

    @property
    def n_raw(self):
        return 5 + 3 * sum(K_VALUES) + 1

    @property
    def n_free(self):
        return None  # depends on k; see n_free_of

    def n_free_of(self, p):
        return 3 * p["k"] + 1

    def decode(self, raw, select=None):
        logits = raw[:, :5]
        k_idx = logits.argmax(-1) if select is None else select
        b = raw.shape[0]
        cents = torch.zeros(b, K_MAX, 3, dtype=raw.dtype, device=raw.device)
        valid = torch.zeros(b, K_MAX, dtype=torch.bool, device=raw.device)
        for j, k in enumerate(K_VALUES):
            m = k_idx == j
            if bool(m.any()):
                o = self._offsets[k]
                cents[m, :k] = raw[m, o:o + 3 * k].reshape(-1, k, 3)
                valid[m, :k] = True
        return {"logits": logits, "k_idx": k_idx, "k": torch.tensor(K_VALUES, device=raw.device)[k_idx],
                "centroids": cents, "valid": valid, "spread": F.softplus(raw[:, -1]) + 1e-4}

    def encode(self, p):
        b = p["centroids"].shape[0]
        raw = torch.zeros(b, self.n_raw, dtype=p["centroids"].dtype, device=p["centroids"].device)
        raw[torch.arange(b), p["k_idx"]] = 5.0                       # confident logit for the true k
        for j, k in enumerate(K_VALUES):
            m = p["k_idx"] == j
            if bool(m.any()):
                o = self._offsets[k]
                raw[m, o:o + 3 * k] = p["centroids"][m, :k].reshape(-1, 3 * k)
        raw[:, -1] = _inv_softplus(p["spread"] - 1e-4)
        return raw

    def points(self, p, u=None):
        w = p["valid"].to(p["centroids"].dtype)
        return p["centroids"], w / w.sum(-1, keepdim=True)

    def dense(self, p, cfg_dense=None):
        return self.points(p)

    def rms_radius(self, p, cfg_dense=None):
        pts, w = self.points(p)
        mu = (pts * w.unsqueeze(-1)).sum(1, keepdim=True)
        return torch.sqrt((((pts - mu) ** 2).sum(-1) * w).sum(-1))

    def sample_params(self, b, gen, device, k_max: int | None = None):
        k_hi = len(K_VALUES) if k_max is None else max(1, min(len(K_VALUES), k_max - 1))
        k_idx = torch.randint(0, k_hi, (b,), generator=gen, device=device)
        cents = torch.randn(b, K_MAX, 3, generator=gen, device=device)
        cents = cents / cents.norm(dim=-1, keepdim=True) * torch.rand(b, K_MAX, 1, generator=gen, device=device) ** (1 / 3)
        kk = torch.tensor(K_VALUES, device=device)[k_idx]
        valid = torch.arange(K_MAX, device=device).unsqueeze(0) < kk.unsqueeze(1)
        cents = cents * valid.unsqueeze(-1)
        p = {"logits": torch.zeros(b, 5, device=device), "k_idx": k_idx, "k": kk, "centroids": cents, "valid": valid}
        p["spread"] = self.uniform(b, 0.02, 0.15, gen, device) * self._min_pair(p)
        return p

    @staticmethod
    def _min_pair(p):
        d = torch.cdist(p["centroids"], p["centroids"])
        pair = p["valid"].unsqueeze(1) & p["valid"].unsqueeze(2) & ~torch.eye(K_MAX, dtype=torch.bool, device=d.device)
        return d.masked_fill(~pair, float("inf")).amin((1, 2))

    def margin_ok(self, p, sigma):
        d = torch.cdist(p["centroids"], p["centroids"])
        pair = p["valid"].unsqueeze(1) & p["valid"].unsqueeze(2)
        dmax = d.masked_fill(~pair, 0).amax((1, 2))
        well_sep = self._min_pair(p) >= torch.maximum(0.25 * dmax, 8 * sigma * self.rms_radius(p))
        return well_sep & ~near_regular_polytope(p["centroids"], p["valid"])

    def sample_valid(self, b, gen, sigma, device, max_iter=50, k_max=None):
        out = self.sample_params(b, gen, device, k_max)
        need = ~self.margin_ok(out, sigma)
        for _ in range(max_iter):
            if not bool(need.any()):
                break
            cand = self.sample_params(b, gen, device, k_max)
            take = need & self.margin_ok(cand, sigma)
            for key in out:
                m = take.view(-1, *([1] * (out[key].dim() - 1)))
                out[key] = torch.where(m, cand[key], out[key])
            need = need & ~take
        return out

    def transform(self, p, q, t, s):
        out = dict(p)
        out["centroids"] = s.view(-1, 1, 1) * (p["centroids"] @ q.transpose(-1, -2)) + t.unsqueeze(1)   # padding moves too (weight 0)
        out["spread"] = p["spread"] * s
        return out

    def symmetries(self, p):
        perm = torch.randperm(K_MAX, device=p["centroids"].device)
        out = dict(p)
        out["centroids"], out["valid"] = p["centroids"][:, perm], p["valid"][:, perm]
        return [out]

    def canonicalize(self, p):
        out = dict(p)
        out["centroids"], out["valid"] = _sort_points(p["centroids"], p["valid"])
        return out

    def aux(self, p):
        return None, torch.log(self.rms_radius(p) + 1e-6)

    def describe(self, p, i):
        k = int(p["valid"][i].sum())
        return {"family": self.name, "k": k, "centroids": p["centroids"][i, p["valid"][i]].tolist(), "spread": float(p["spread"][i])}


# ------------------------------------------------------------------ polytope templates (unit circumradius)
def _polytope_templates() -> dict[str, torch.Tensor]:
    ang = lambda n, off=0.0: [(math.cos(2 * math.pi * k / n + off), math.sin(2 * math.pi * k / n + off), 0.0) for k in range(n)]
    s3 = 1 / math.sqrt(3)
    # Regular square antiprism (all edges equal): square radius rho at heights +-h with h = rho * 2^(1/4) / 2,
    # scaled to circumradius 1 (rho^2 + h^2 = 1).
    rho = 1 / math.sqrt(1 + math.sqrt(2) / 4)
    h = rho * 2 ** 0.25 / 2
    anti = [(x * rho, y * rho, h) for x, y, _ in ang(4)] + [(x * rho, y * rho, -h) for x, y, _ in ang(4, math.pi / 4)]
    t = {
        "triangle": ang(3), "square": ang(4),
        "tetrahedron": [(s3, s3, s3), (s3, -s3, -s3), (-s3, s3, -s3), (-s3, -s3, s3)],
        "pentagon": ang(5),
        "octahedron": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        "square_antiprism": anti,
    }
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in t.items()}


TEMPLATES = _polytope_templates()
SUBTYPES = tuple(TEMPLATES)
V_MAX = max(len(v) for v in TEMPLATES.values())
N_VERTS = torch.tensor([len(TEMPLATES[s]) for s in SUBTYPES])


def _padded_templates(device, dtype):
    out = torch.zeros(len(SUBTYPES), V_MAX, 3, dtype=dtype, device=device)
    valid = torch.zeros(len(SUBTYPES), V_MAX, dtype=torch.bool, device=device)
    for i, s in enumerate(SUBTYPES):
        n = len(TEMPLATES[s])
        out[i, :n] = TEMPLATES[s].to(device, dtype)
        valid[i, :n] = True
    return out, valid


def _mirrors(device, dtype) -> torch.Tensor:
    """(n_subtypes, 3, 3) mirror symmetries (det -1) of each template vertex set."""
    yflip = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
    swap = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return torch.stack([swap if s == "tetrahedron" else yflip for s in SUBTYPES]).to(device, dtype)


def _dist_profile(pts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Sorted pairwise distances normalised by the maximum (a rotation/scale-invariant signature)."""
    d = torch.cdist(pts, pts)
    pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    d = d.masked_fill(~pair, -1.0)
    d = d.flatten(-2).sort(-1, descending=True).values
    return d / d[..., :1].clamp_min(1e-9)


def near_regular_polytope(cents: torch.Tensor, valid: torch.Tensor, tol: float = 0.08) -> torch.Tensor:
    """True where a centroid set has (nearly) the distance profile of a polytope template with the same
    vertex count, so the 'clusters' label would be ambiguous with 'polytope'."""
    prof = _dist_profile(cents, valid)
    k = valid.sum(-1)
    near = torch.zeros(cents.shape[0], dtype=torch.bool, device=cents.device)
    tpl, tvalid = _padded_templates(cents.device, cents.dtype)
    tprof = _dist_profile(tpl, tvalid)
    for i in range(len(SUBTYPES)):
        n = int(tvalid[i].sum())
        if n > K_MAX:
            continue
        m = k == n
        if bool(m.any()):
            diff = (prof[m][:, : n * n] - tprof[i][: n * n]).abs().amax(-1)   # both profiles sorted descending
            near[m] = diff < tol
    return near


class Polytope(ShapeFamily):
    """Vertices of a polytope template: c + radius * R @ template[subtype] (+ small vertex spread).
    Raw layout: [subtype logits (6), center (3), rot6d (6), log radius (1), log spread (1)]."""

    name, coarse, manifold_dim, posed = "polytope", "0d", 0, False

    @property
    def n_raw(self):
        return len(SUBTYPES) + 3 + 6 + 2

    @property
    def n_free(self):
        return 3 + 3 + 1 + 1

    def decode(self, raw, select=None):
        ns = len(SUBTYPES)
        logits = raw[:, :ns]
        sub = logits.argmax(-1) if select is None else select
        return {"logits": logits, "subtype": sub, "center": raw[:, ns:ns + 3], "rot": rot6d_to_matrix(raw[:, ns + 3:ns + 9]),
                "radius": F.softplus(raw[:, ns + 9]) + 1e-3, "spread": F.softplus(raw[:, ns + 10]) + 1e-4}

    def encode(self, p):
        ns, b = len(SUBTYPES), p["center"].shape[0]
        raw = torch.zeros(b, self.n_raw, dtype=p["center"].dtype, device=p["center"].device)
        raw[torch.arange(b), p["subtype"]] = 5.0
        raw[:, ns:ns + 3] = p["center"]
        raw[:, ns + 3:ns + 9] = matrix_to_rot6d(p["rot"])
        raw[:, ns + 9] = _inv_softplus(p["radius"] - 1e-3)
        raw[:, ns + 10] = _inv_softplus(p["spread"] - 1e-4)
        return raw

    def vertices(self, p):
        tpl, tvalid = _padded_templates(p["center"].device, p["center"].dtype)
        v = tpl[p["subtype"]] * p["radius"].view(-1, 1, 1)
        v = p["center"].unsqueeze(1) + v @ p["rot"].transpose(-1, -2)
        return v, tvalid[p["subtype"]]

    def points(self, p, u=None):
        v, valid = self.vertices(p)
        w = valid.to(v.dtype)
        return v, w / w.sum(-1, keepdim=True)

    def dense(self, p, cfg_dense=None):
        return self.points(p)

    def rms_radius(self, p, cfg_dense=None):
        return p["radius"]

    def sample_params(self, b, gen, device, k_max: int | None = None):
        allowed = torch.ones(len(SUBTYPES), dtype=torch.bool)
        if k_max is not None:
            allowed = N_VERTS <= k_max
        idx = torch.nonzero(allowed).squeeze(-1).to(device)
        sub = idx[torch.randint(0, len(idx), (b,), generator=gen, device=device)]
        edge_frac = self.uniform(b, 0.0, 0.1, gen, device)
        return {"logits": torch.zeros(b, len(SUBTYPES), device=device), "subtype": sub, "center": torch.zeros(b, 3, device=device),
                "rot": random_rotation(b, generator=gen, device=device), "radius": torch.ones(b, device=device),
                "spread": 1e-3 + edge_frac * 0.5}

    def sample_valid(self, b, gen, sigma, device, max_iter=50, k_max=None):
        return self.sample_params(b, gen, device, k_max)

    def transform(self, p, q, t, s):
        out = dict(p)
        out["center"] = s.unsqueeze(-1) * (q @ p["center"].unsqueeze(-1)).squeeze(-1) + t
        rot = p["rot"]
        improper = det3(q) < 0
        if bool(improper.any()):
            # All six templates are achiral: a mirror symmetry F of the template (det -1) turns
            # q R into the proper rotation q R F with the same vertex set (y-flip; x<->y swap for the tetrahedron).
            mir = _mirrors(q.device, q.dtype)[p["subtype"]]
            rot = torch.where(improper.view(-1, 1, 1), rot @ mir, rot)
        out["rot"] = q @ rot
        out["radius"], out["spread"] = p["radius"] * s, p["spread"] * s
        return out

    def symmetries(self, p):
        # A rotation by 2*pi/3 about the tetrahedron's (1,1,1) axis / 2*pi/n about z for planar polygons maps the
        # vertex set to itself; tests use the subtype-appropriate one.
        return []

    def canonicalize(self, p):
        v, valid = self.vertices(p)
        out = dict(p)
        out["vertices"], out["vertex_valid"] = _sort_points(v, valid)
        return out

    def aux(self, p):
        return None, torch.log(p["radius"])

    def describe(self, p, i):
        valid = p["vertex_valid"][i]
        return {"family": self.name, "subtype": SUBTYPES[int(p["subtype"][i])], "center": p["center"][i].tolist(),
                "circumradius": float(p["radius"][i]), "vertices": p["vertices"][i, valid].tolist(), "spread": float(p["spread"][i])}
