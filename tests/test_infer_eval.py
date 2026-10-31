"""Tests for refinement, classical fitting, model selection and statistics."""

import numpy as np
import torch

from neurogeom.baselines.classical import fit_one_family, kmeans
from neurogeom.config import load_config
from neurogeom.data.synth import generate_family
from neurogeom.eval.stats import bootstrap_ci, ece, fit_temperature, holm, paired_bootstrap
from neurogeom.geometry.chamfer import chamfer
from neurogeom.infer.refine import refine
from neurogeom.infer.select import gaic_scores
from neurogeom.seeding import make_generator
from neurogeom.shapes.registry import INDEX, family

CFG = load_config()
CLEAN = dict(CFG["synth"], outlier_prob=0.0, aniso_noise_prob=0.0, embed_prob=0.0, density_modes=["uniform"])
DENSE = {"curve": 256, "surface_side": 24, "blob": 256}


def _data(name, b=8, n=200, sigma=0.0, seed=0):
    return generate_family(name, b, n, CLEAN, DENSE, make_generator(seed), "cpu", sigma=torch.full((b,), sigma), embed_dim=None)


def test_refine_recovers_ground_truth_from_perturbed_start():
    d = _data("helix")
    f = family("helix")
    raw_gt = f.encode(d["gt"])
    raw0 = raw_gt + 0.15 * torch.randn_like(raw_gt)
    raw, _ = refine(f, raw0, d["x"], d["mask"], None, steps=300, lr=1e-2)
    pts, w = f.dense(f.decode(raw), DENSE)
    assert chamfer(pts, w, d["gt_dense"], d["gt_w"]).median() < 0.05


def test_refine_does_not_inflate_half_circle():
    """A half circle must not be 'explained' by a full circle: the shape->points term forbids it."""
    f = family("arc")
    d = _data("arc", b=16)
    half = d["gt"]["extent_frac"] < 0.6
    raw0 = f.encode(d["gt"])
    raw0[:, -1] = 4.0                                              # start from a (nearly) full circle
    raw, _ = refine(f, raw0, d["x"], d["mask"], None, steps=400, lr=2e-2)
    frac = f.decode(raw)["extent_frac"]
    assert (frac[half] < d["gt"]["extent_frac"][half] + 0.15).all()


def test_classical_recovers_noiseless_circle_and_line():
    for name in ("arc", "line"):
        d = _data(name, b=6)
        c = fit_one_family(INDEX[name], d["x"], d["mask"], steps=300, back_w=0.25, gen=make_generator(1))
        f = family(name)
        pts, w = f.dense(f.decode(c["raw"]), DENSE)
        assert chamfer(pts, w, d["gt_dense"], d["gt_w"]).median() < 0.05, name


def test_kmeans_separates_clusters():
    x = torch.cat([torch.randn(1, 30, 3) * 0.05 + torch.tensor([3.0, 0, 0]), torch.randn(1, 30, 3) * 0.05 - torch.tensor([3.0, 0, 0])], 1)
    c = kmeans(x, torch.ones(1, 60, dtype=torch.bool), 2, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(c[0, :, 0].abs().sort().values, torch.tensor([3.0, 3.0]), atol=0.1)


def test_gaic_prefers_simpler_model_when_fit_is_equal():
    n = torch.tensor([100.0])
    cands = [{"fi": INDEX["arc"], "J": torch.tensor([1.0]), "p": torch.tensor([8.0])},
             {"fi": INDEX["helix"], "J": torch.tensor([1.0]), "p": torch.tensor([9.0])},
             {"fi": INDEX["sphere"], "J": torch.tensor([0.9]), "p": torch.tensor([7.0])}]
    aic = gaic_scores(cands, n)[0]
    assert aic[0] < aic[1]                                         # same J, fewer parameters wins
    assert aic[0] < aic[2]                                         # 2D model pays 2 N eps^2 more


def test_bootstrap_and_holm_and_ece_and_temperature():
    rng = np.random.default_rng(0)
    est, lo, hi = bootstrap_ci(rng.normal(1.0, 1.0, 400), reps=500)
    assert lo < 1.0 < hi and lo < est < hi
    d, lo, hi = paired_bootstrap(np.ones(50) * 2, np.ones(50), reps=200)
    assert d == 1.0 and lo == 1.0
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06
    probs = np.eye(3)[rng.integers(0, 3, 300)]
    assert ece(probs, probs.argmax(1)) < 1e-9
    logits = torch.randn(500, 4) * 5
    labels = torch.distributions.Categorical(logits=logits / 5).sample()
    assert abs(fit_temperature(logits, labels) - 5.0) < 1.0      # recovers the true temperature
