"""Tests for the 13 shape families: encoding, sampling, transforms, symmetries/canonical form, margins."""

import math

import numpy as np
import pytest
import torch

from neurogeom.geometry.chamfer import chamfer, one_sided
from neurogeom.geometry.rotation6d import random_rotation
from neurogeom.shapes.pointsets import SUBTYPES
from neurogeom.shapes.registry import FAMILIES, NAMES, family

DENSE = {"curve": 128, "surface_side": 16, "blob": 256}
B = 16


def _params(f, seed=0, sigma=0.02):
    g = torch.Generator().manual_seed(seed)
    p = f.sample_valid(B, g, torch.full((B,), sigma), "cpu")
    # move off the origin so transforms/centers are exercised
    if "center" in p:
        p["center"] = torch.randn(B, 3, generator=g)
    return p


def _select(f, p):
    return p.get("k_idx", p.get("subtype"))


def _flat(d):
    out = []
    for k, v in sorted(d.items()):
        if isinstance(v, (int, float, bool)) and not isinstance(v, str):
            out.append(float(v))
        elif isinstance(v, list):
            out.extend(np.asarray(v, dtype=float).ravel().tolist())
    return np.array(out)


@pytest.mark.parametrize("f", FAMILIES, ids=NAMES)
def test_encode_decode_roundtrip(f):
    p = _params(f)
    p2 = f.decode(f.encode(p), _select(f, p))
    a, wa = f.dense(p, DENSE)
    b, wb = f.dense(p2, DENSE)
    assert torch.allclose(a, b, atol=1e-3), f.name


@pytest.mark.parametrize("f", FAMILIES, ids=NAMES)
def test_dense_samples_finite_and_weighted(f):
    pts, w = f.dense(_params(f), DENSE)
    assert torch.isfinite(pts).all() and torch.isfinite(w).all()
    assert (w >= 0).all() and (w.sum(-1) > 0).all()


@pytest.mark.parametrize("f", FAMILIES, ids=NAMES)
def test_transform_equivariance(f):
    p = _params(f, seed=1)
    g = torch.Generator().manual_seed(7)
    q = random_rotation(B, generator=g)
    t = torch.randn(B, 3, generator=g)
    s = torch.rand(B, generator=g) + 0.5
    a, _ = f.dense(f.transform(p, q, t, s), DENSE)
    b, _ = f.dense(p, DENSE)
    b = s.view(-1, 1, 1) * (b @ q.transpose(-1, -2)) + t.unsqueeze(1)
    assert torch.allclose(a, b, atol=1e-4), f.name


@pytest.mark.parametrize("f", FAMILIES, ids=NAMES)
def test_symmetries_preserve_shape_and_canonical_form(f):
    p = _params(f, seed=2)
    syms = f.symmetries(p)
    if f.name == "polytope":  # build a genuine symmetry per subtype: rotate the template by a group element
        syms = []
        for i, sub in enumerate(SUBTYPES):
            p_i = {k: v.clone() for k, v in p.items()}
            p_i["subtype"] = torch.full_like(p["subtype"], i)
            n = {"triangle": 3, "square": 4, "pentagon": 5, "octahedron": 4, "square_antiprism": 4, "tetrahedron": 2}[sub]
            ang = 2 * math.pi / n
            if sub == "tetrahedron":  # 180 deg about z maps the regular tetrahedron onto itself
                ang = math.pi
            rz = torch.tensor([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1.0]])
            q = dict(p_i)
            q["rot"] = p_i["rot"] @ rz
            ca, cb = f.canonicalize(p_i), f.canonicalize(q)
            assert torch.allclose(ca["vertices"], cb["vertices"], atol=1e-4), sub
        return
    # Compare against a much finer sampling of the original so grid offsets (e.g. a rotated phase grid) do not count.
    base_pts, base_w = f.dense(p, {"curve": 4096, "surface_side": 96, "blob": 256})
    can = f.canonicalize(p)
    for sp in syms:
        pts, w = f.dense(sp, DENSE)
        assert one_sided(pts, w, base_pts, base_w).max() < 0.06, f.name  # ~ half the fine-grid spacing
        cs = f.canonicalize(sp)
        for i in range(B):
            assert np.allclose(_flat(f.describe(cs, i)), _flat(f.describe(can, i)), atol=1e-4), (f.name, i)


@pytest.mark.parametrize("f", FAMILIES, ids=NAMES)
def test_canonicalize_idempotent(f):
    c1 = f.canonicalize(_params(f, seed=3))
    c2 = f.canonicalize(c1)
    for i in range(B):
        assert np.allclose(_flat(f.describe(c1, i)), _flat(f.describe(c2, i)), atol=1e-5)


@pytest.mark.parametrize("f", FAMILIES, ids=NAMES)
def test_margins_respected(f):
    sigma = torch.full((B,), 0.03)
    p = f.sample_valid(B, torch.Generator().manual_seed(4), sigma, "cpu")
    assert f.margin_ok(p, sigma).all(), f.name


@pytest.mark.parametrize("f", FAMILIES, ids=NAMES)
def test_gradients_finite(f):
    p = _params(f, seed=5)
    raw = f.encode(p).detach().requires_grad_(True)
    q = f.decode(raw, _select(f, p))
    pts, w = f.dense(q, DENSE)
    tgt, wt = f.dense(_params(f, seed=6), DENSE)
    chamfer(pts, w, tgt, wt).sum().backward()
    assert torch.isfinite(raw.grad).all(), f.name


# ------------------------------------------------------------------ implicit equations of key families
def _local(f, p):
    pts, _ = f.dense(p, DENSE)
    return (pts - p["center"].unsqueeze(1)) @ p["rot"]  # back to the local frame


def test_arc_on_circle():
    f = family("arc")
    p = _params(f)
    loc = _local(f, p)
    assert torch.allclose(loc.norm(dim=-1), p["radius"].unsqueeze(-1).expand(-1, loc.shape[1]), atol=1e-5)
    assert loc[..., 2].abs().max() < 1e-5


def test_sphere_on_sphere():
    f = family("sphere")
    p = _params(f)
    loc = _local(f, p)
    assert torch.allclose(loc.norm(dim=-1), p["radius"].unsqueeze(-1).expand(-1, loc.shape[1]), atol=1e-4)


def test_helix_radius_and_rise():
    f = family("helix")
    p = _params(f)
    loc = _local(f, p)
    assert torch.allclose(loc[..., :2].norm(dim=-1), p["radius"].unsqueeze(-1).expand(-1, loc.shape[1]), atol=1e-5)
    height = loc[..., 2].amax(-1) - loc[..., 2].amin(-1)
    assert torch.allclose(height, (p["pitch"].abs() * p["turns"]) * (1 - 1 / 128), atol=1e-3)


def test_torus_implicit():
    f = family("torus")
    p = _params(f)
    loc = _local(f, p)
    rr = (loc[..., :2].norm(dim=-1) - p["R"].unsqueeze(-1)) ** 2 + loc[..., 2] ** 2
    assert torch.allclose(rr, ((p["R"] * p["ratio"]) ** 2).unsqueeze(-1).expand_as(rr), atol=1e-5)


def test_paraboloid_implicit():
    f = family("paraboloid")
    p = _params(f)
    loc = _local(f, p)
    k1, k2 = p["k1"].unsqueeze(-1), (p["k1"] * p["k_ratio"]).unsqueeze(-1)
    assert torch.allclose(loc[..., 2], 0.5 * (k1 * loc[..., 0] ** 2 + k2 * loc[..., 1] ** 2), atol=1e-4)


@pytest.mark.parametrize("f", FAMILIES, ids=NAMES)
def test_transform_under_reflection_is_same_point_set(f):
    """PCA of a 3D subspace of R^D may return a mirrored frame; transform() must still map the shape
    onto its mirror image (compared as point sets, since traversal order may flip)."""
    p = _params(f, seed=8)
    g = torch.Generator().manual_seed(9)
    q = random_rotation(B, generator=g)
    q[:, :, 0] = -q[:, :, 0]                                   # det = -1
    t, s = torch.randn(B, 3, generator=g), torch.rand(B, generator=g) + 0.5
    fine = {"curve": 1024, "surface_side": 48, "blob": 256}
    a, wa = f.dense(f.transform(p, q, t, s), fine)
    b, wb = f.dense(p, fine)
    b = s.view(-1, 1, 1) * (b @ q.transpose(-1, -2)) + t.unsqueeze(1)
    assert chamfer(a, wa, b, wb).max() < 0.05 * s.max(), f.name
