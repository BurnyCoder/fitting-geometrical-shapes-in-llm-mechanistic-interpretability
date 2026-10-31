"""Baseline B2: hand-crafted geometric/topological features + gradient-boosted trees for the family.

Global context: a learned-but-not-deep alternative. Features describe the cloud's global and local
dimensionality and topology; the classifier is trained on synthetic clouds from the same generator
(a separate sub-stream of `train`). The predicted family is then fitted with B1's classical fitter.
Feature sources:
- global PCA variance ratios;
- local covariance features (linearity, planarity, scattering) in k-NN neighbourhoods
  (Weinmann et al. 2015, "Semantic point cloud interpretation based on optimal neighborhoods,
  relevant features and efficient classifiers", ISPRS J. Photogramm. 105);
- persistent homology H0/H1 lifetimes via ripser.py (Tralie, Saul, Bar-On 2018, JOSS 3(29):925;
  https://github.com/scikit-tda/ripser.py, MIT) on a <= 150-point subsample;
- radial distance statistics and log N.
Classifier: sklearn HistGradientBoostingClassifier (Ke et al. 2017-style histogram GBDT).
"""

from __future__ import annotations

import numpy as np
import torch
from ripser import ripser
from sklearn.ensemble import HistGradientBoostingClassifier

from neurogeom.data.nuisance import log_uniform
from neurogeom.data.synth import generate_family
from neurogeom.logging_utils import get_logger
from neurogeom.seeding import make_generator, stream_seed
from neurogeom.shapes.registry import FAMILIES

log = get_logger(__name__)


def cloud_features(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Feature vector of one normalised cloud (N, 3)."""
    n = len(x)
    ev = np.sort(np.linalg.eigvalsh(np.cov(x.T) + 1e-12 * np.eye(3)))[::-1]
    glob = [ev[1] / ev[0], ev[2] / ev[0], np.log(n)]
    k = min(16, n - 1)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    nn = np.argsort(d, 1)[:, 1:k + 1]
    nb = x[nn] - x[nn].mean(1, keepdims=True)                                  # (N, k, 3) neighbourhoods
    e = np.linalg.eigvalsh(nb.transpose(0, 2, 1) @ nb / max(1, k) + 1e-12 * np.eye(3))[:, ::-1]   # (N, 3) desc
    loc = np.stack([(e[:, 0] - e[:, 1]) / e[:, 0], (e[:, 1] - e[:, 2]) / e[:, 0], e[:, 2] / e[:, 0]], 1)
    r = np.linalg.norm(x, axis=1)
    sub = x if n <= 150 else x[rng.choice(n, 150, replace=False)]
    dg = ripser(sub, maxdim=1)["dgms"]
    h0 = np.sort(dg[0][np.isfinite(dg[0][:, 1]), 1])[::-1]
    h1 = np.sort(dg[1][:, 1] - dg[1][:, 0])[::-1] if len(dg[1]) else np.zeros(0)
    top = lambda a, m: list(a[:m]) + [0.0] * (m - len(a[:m]))
    nn1 = d[np.arange(n), nn[:, 0]] if k >= 1 else np.zeros(n)
    feats = glob + list(loc.mean(0)) + list(loc.std(0)) + [r.mean(), r.std(), np.percentile(r, 10), np.percentile(r, 90)] \
        + top(h0, 3) + top(h1, 3) + [h1.sum(), float((h1 > 0.1).sum()), nn1.mean(), nn1.std()]
    return np.asarray(feats, dtype=np.float64)


def features_batch(x: torch.Tensor, mask: torch.Tensor, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs, ms = x.cpu().numpy(), mask.cpu().numpy()
    return np.stack([cloud_features(xs[i][ms[i]], rng) for i in range(len(xs))])


def train_gbm(cfg: dict, per_family: int = 1500, device="cuda") -> HistGradientBoostingClassifier:
    gen = make_generator(stream_seed(cfg["seed"], "train", salt="gbm"), device)
    feats, labels = [], []
    for fi in range(len(FAMILIES)):
        left = per_family
        while left > 0:
            c = min(50, left)
            left -= c
            n = int(round(float(log_uniform(1, cfg["synth"]["n_min"], cfg["synth"]["n_max"], gen, device))))
            d = generate_family(fi, c, n, cfg["synth"], cfg["dense"], gen, device)
            feats.append(features_batch(d["x"], d["mask"]))
            labels += [fi] * c
        log.info("GBM features: family %s done", FAMILIES[fi].name)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=cfg["seed"])
    clf.fit(np.concatenate(feats), np.array(labels))
    return clf
