"""Tests for the set encoder / ShapeFitter and the training loss."""

import pytest
import torch

from neurogeom.config import load_config
from neurogeom.data.synth import generate_mixed
from neurogeom.model.net import ShapeFitter
from neurogeom.seeding import make_generator
from neurogeom.shapes.registry import FAMILIES
from neurogeom.train.losses import compute_loss

CFG = load_config()


@pytest.mark.parametrize("encoder", ["set_transformer", "deepsets"])
def test_permutation_and_padding_invariance(encoder):
    torch.manual_seed(0)
    model = ShapeFitter(dict(CFG["model"], encoder=encoder)).eval()
    x = torch.randn(3, 40, 3)
    m = torch.ones(3, 40, dtype=torch.bool)
    out = model(x, m)["logits"]
    perm = torch.randperm(40)
    assert torch.allclose(model(x[:, perm], m)["logits"], out, atol=1e-5)
    xp = torch.cat([x, 50 * torch.randn(3, 9, 3)], 1)
    mp = torch.cat([m, torch.zeros(3, 9, dtype=torch.bool)], 1)
    assert torch.allclose(model(xp, mp)["logits"], out, atol=1e-5)


def test_runs_on_tiny_and_large_clouds_and_heads_decode():
    model = ShapeFitter(CFG["model"]).eval()
    for n in (5, 2000):
        out = model(torch.randn(2, n, 3), torch.ones(2, n, dtype=torch.bool))
        assert out["logits"].shape == (2, len(FAMILIES))
        for f, raw in zip(FAMILIES, out["raw"]):
            pts, w = f.dense(f.decode(raw), CFG["dense"])
            assert torch.isfinite(pts).all() and (w.sum(-1) > 0).all(), f.name


def test_loss_near_zero_at_ground_truth():
    # no R^D embedding: there the GT params are only an approximation of the projected GT shape (documented)
    parts = generate_mixed(26, 64, dict(CFG["synth"], outlier_prob=0.0, embed_prob=0.0), CFG["dense"], make_generator(0), "cpu")
    b = sum(p["x"].shape[0] for p in parts)
    raws = [torch.zeros(b, f.n_raw) for f in FAMILIES]
    start = 0
    for p in parts:
        fi, n = int(p["family"][0]), p["x"].shape[0]
        raws[fi][start:start + n] = FAMILIES[fi].encode(p["gt"])
        start += n
    logits = torch.full((b, len(FAMILIES)), -10.0)
    logits[torch.arange(b), torch.cat([p["family"] for p in parts])] = 10.0
    _, st = compute_loss({"logits": logits, "raw": raws}, parts, CFG["loss"], CFG["dense"], aux_weight=1.0)
    cds = [v for k, v in st.items() if k.startswith("cd/")]
    assert max(cds) < 0.02, st
    assert st["acc"] == 1.0
