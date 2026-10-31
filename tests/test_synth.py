"""Tests for the synthetic generator: reproducibility, normalisation, and ground-truth consistency."""

import pytest
import torch

from neurogeom.config import load_config
from neurogeom.data.synth import generate_family, generate_mixed
from neurogeom.geometry.chamfer import chamfer, residual
from neurogeom.seeding import make_generator, stream_seed
from neurogeom.shapes.registry import FAMILIES, NAMES

CFG = load_config()
FINE = {"curve": 2048, "surface_side": 64, "blob": 256}
CLEAN = dict(CFG["synth"], outlier_prob=0.0, aniso_noise_prob=0.0)


def test_reproducible_and_streams_disjoint():
    a = generate_family("helix", 8, 50, CFG["synth"], CFG["dense"], make_generator(1), "cpu")
    b = generate_family("helix", 8, 50, CFG["synth"], CFG["dense"], make_generator(1), "cpu")
    assert torch.equal(a["x"], b["x"])
    seeds = {stream_seed(0, s) for s in ("train", "val", "test", "ood", "real")}
    assert len(seeds) == 5


@pytest.mark.parametrize("f", FAMILIES, ids=NAMES)
@pytest.mark.parametrize("n", [5, 300])
def test_normalised_and_finite(f, n):
    d = generate_family(f, 6, n, CFG["synth"], CFG["dense"], make_generator(2), "cpu")
    x = d["x"]
    assert torch.isfinite(x).all() and torch.isfinite(d["gt_dense"]).all()
    assert torch.allclose(x.mean(1), torch.zeros(6, 3), atol=1e-4)
    assert torch.allclose(x.pow(2).sum(-1).mean(1).sqrt(), torch.ones(6), atol=1e-4)


@pytest.mark.parametrize("f", [f for f in FAMILIES if f.posed], ids=[f.name for f in FAMILIES if f.posed])
@pytest.mark.parametrize("embed", [None, 64])
def test_noise_free_points_lie_on_ground_truth(f, embed):
    """With sigma = 0 and no outliers, observed points lie on the GT shape in the normalised frame,
    and the transformed GT parameters reproduce the transformed GT dense samples (also after R^D embedding)."""
    d = generate_family(f, 6, 200, CLEAN, FINE, make_generator(3), "cpu", sigma=torch.zeros(6), embed_dim=embed)
    if f.name != "blob":
        res = residual(d["x"], d["mask"], d["gt_dense"], d["gt_w"])
        assert res.max() < 0.03, f.name
    pts, w = f.dense(d["gt"], FINE)
    # compare as point sets: after a mirrored PCA frame the same shape is traversed in the opposite direction
    assert chamfer(pts, w, d["gt_dense"], d["gt_w"]).max() < 5e-3, f.name


def test_pointsets_cover_all_vertices():
    for name in ("clusters", "polytope"):
        d = generate_family(name, 8, 12, CLEAN, CFG["dense"], make_generator(4), "cpu", sigma=torch.zeros(8), embed_dim=None)
        f = FAMILIES[NAMES.index(name)]
        v, w = f.points(d["gt"])
        # every valid vertex has an observed point within a few spreads of it (each vertex gets >= 1 point)
        dist = torch.cdist(v, d["x"]).amin(-1)
        tol = 4.5 * d["gt"]["spread"].unsqueeze(-1).expand_as(dist) + 1e-3
        assert (dist[w > 0] < tol[w > 0]).all(), name


def test_mixed_batch_covers_all_families():
    parts = generate_mixed(40, 20, CFG["synth"], CFG["dense"], make_generator(5), "cpu")
    fams = torch.cat([p["family"] for p in parts])
    assert len(fams) == 40 and set(fams.tolist()) == set(range(len(FAMILIES)))
