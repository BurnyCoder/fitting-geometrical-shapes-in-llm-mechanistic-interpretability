"""Tests for rotation6d, chamfer and PCA preprocessing (the numerical foundation of every phase)."""

import numpy as np
import pytest
import torch

from neurogeom.geometry.chamfer import chamfer, nearest_dist, residual
from neurogeom.geometry.preprocess import fit_frame, normalize_batch, to_original
from neurogeom.geometry.rotation6d import matrix_to_rot6d, nearest_rotation, random_rotation, rot6d_to_matrix


# ------------------------------------------------------------------ rotation6d
def test_rot6d_is_proper_rotation():
    r = rot6d_to_matrix(torch.randn(100, 6, dtype=torch.float64))
    eye = torch.eye(3, dtype=torch.float64).expand(100, 3, 3)
    assert torch.allclose(r.transpose(-1, -2) @ r, eye, atol=1e-10)
    assert torch.allclose(torch.linalg.det(r), torch.ones(100, dtype=torch.float64), atol=1e-10)


def test_rot6d_roundtrip():
    r = random_rotation(50, dtype=torch.float64)
    assert torch.allclose(rot6d_to_matrix(matrix_to_rot6d(r)), r, atol=1e-10)


def test_rot6d_gradcheck():
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(rot6d_to_matrix, (x,))


def test_random_rotation_det_and_nearest():
    r = random_rotation(64, dtype=torch.float64)
    assert torch.allclose(torch.linalg.det(r), torch.ones(64, dtype=torch.float64), atol=1e-10)
    noisy = r + 1e-3 * torch.randn_like(r)
    assert torch.allclose(nearest_rotation(noisy), r, atol=5e-3)


# ------------------------------------------------------------------ chamfer
def _brute_chamfer(p, wp, q, wq):
    d = np.linalg.norm(p[:, None] - q[None], axis=-1)
    wp, wq = wp / wp.sum(), wq / wq.sum()
    return (wp * d.min(1)).sum() + (wq * d.min(0)).sum()


def test_chamfer_matches_bruteforce():
    rng = np.random.default_rng(0)
    p, q = rng.normal(size=(1, 40, 3)), rng.normal(size=(1, 25, 3))
    wp, wq = rng.uniform(0.1, 1, size=(1, 40)), rng.uniform(0.1, 1, size=(1, 25))
    got = chamfer(*(torch.tensor(a) for a in (p, wp, q, wq)))
    assert got.item() == pytest.approx(_brute_chamfer(p[0], wp[0], q[0], wq[0]), abs=1e-6)


def test_chamfer_symmetric_zero_permutation_and_padding():
    p = torch.randn(2, 30, 3, dtype=torch.float64)
    w = torch.ones(2, 30, dtype=torch.float64)
    assert torch.allclose(chamfer(p, w, p, w), torch.zeros(2, dtype=torch.float64), atol=1e-3)
    q = torch.randn(2, 20, 3, dtype=torch.float64)
    wq = torch.ones(2, 20, dtype=torch.float64)
    assert torch.allclose(chamfer(p, w, q, wq), chamfer(q, wq, p, w))
    perm = torch.randperm(30)
    assert torch.allclose(chamfer(p[:, perm], w[:, perm], q, wq), chamfer(p, w, q, wq))
    # padding with zero-weight points far away must not change anything
    pad = torch.full((2, 5, 3), 100.0, dtype=torch.float64)
    p2, w2 = torch.cat([p, pad], 1), torch.cat([w, torch.zeros(2, 5, dtype=torch.float64)], 1)
    assert torch.allclose(chamfer(p2, w2, q, wq), chamfer(p, w, q, wq))


def test_residual_masks_input_padding():
    s = torch.zeros(1, 1, 3)
    x = torch.tensor([[[1.0, 0, 0], [0, 2.0, 0], [50.0, 0, 0]]])
    m = torch.tensor([[True, True, False]])
    assert residual(x, m, s, torch.ones(1, 1)).item() == pytest.approx(1.5, abs=1e-4)
    assert nearest_dist(x, s).shape == (1, 3)


# ------------------------------------------------------------------ preprocessing
def _cloud(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 4 * np.pi, n)
    return np.stack([np.cos(t), 0.6 * np.sin(t), 0.1 * t + 0.3 * t ** 2 / 10], 1)  # skewed helix-ish curve


def test_preprocess_rms_and_centre():
    y, frame = fit_frame(_cloud())
    assert np.allclose(y.mean(0), 0, atol=1e-10)
    assert np.sqrt((y ** 2).sum(1).mean()) == pytest.approx(1.0, abs=1e-10)
    assert np.linalg.det(frame.basis) > 0


def test_preprocess_similarity_invariance():
    x = _cloud()
    y0, _ = fit_frame(x)
    r = random_rotation(1, generator=torch.Generator().manual_seed(3), dtype=torch.float64)[0].numpy()
    y1, _ = fit_frame(3.7 * x @ r.T + np.array([5.0, -2.0, 1.0]))
    assert np.allclose(y0, y1, atol=1e-8)


def test_preprocess_roundtrip_highdim():
    rng = np.random.default_rng(1)
    e = np.linalg.qr(rng.normal(size=(64, 3)))[0]            # embed the 3D cloud in R^64
    x = _cloud() @ e.T + 2.0
    y, frame = fit_frame(x)
    assert np.allclose(to_original(y, frame), x, atol=1e-8)  # exact: the cloud lies in a 3D subspace


def test_normalize_batch_mask_equivalence():
    x = torch.tensor(_cloud(50))[None]
    pad = torch.cat([x, torch.full((1, 7, 3), 99.0, dtype=x.dtype)], 1)
    m = torch.cat([torch.ones(1, 50, dtype=torch.bool), torch.zeros(1, 7, dtype=torch.bool)], 1)
    y0, _ = normalize_batch(x)
    y1, _ = normalize_batch(pad, m)
    assert torch.allclose(y0, y1[:, :50], atol=1e-8)
    assert torch.all(y1[:, 50:] == 0)
