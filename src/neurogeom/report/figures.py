"""Figures for README / reports / paper, generated only from results files (one command regenerates all).

Global context: the colours are the categorical slots of the validated reference palette of the
dataviz guidance (blue, orange, aqua, yellow, magenta, green, violet, red - fixed order, never cycled;
first three slots validated for all-pairs use). Systems keep the same colour in every figure.
Intervals are 95% bootstrap CIs; no dual axes; axes start at zero for accuracies.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neurogeom.config import REPO_ROOT  # noqa: E402
from neurogeom.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SYSTEM_ORDER = ["method", "method_subset", "A3_norefine", "A5_gaic", "A5_hybrid", "A0_reference", "A1_paramloss", "A2_noresidual", "A4_deepsets", "A6_noembed",
                "B1_classical", "B2_features", "B3a_null", "B3b_circle", "B3c_random"]
SYSTEM_LABEL = {"method": "Method (network + refinement)", "method_subset": "Method, baseline subset", "A3_norefine": "A3: no refinement", "A5_gaic": "A5: G-AIC selection",
                "A5_hybrid": "A5: hybrid selection", "A0_reference": "A0: method, ablation budget", "A1_paramloss": "A1: parameter-space loss", "A2_noresidual": "A2: no residual term",
                "A4_deepsets": "A4: DeepSets encoder", "A6_noembed": "A6: no R^D augmentation", "B1_classical": "B1: classical fit + G-AIC",
                "B2_features": "B2: features + boosting", "B3a_null": "B3a: always blob", "B3b_circle": "B3b: always circle",
                "B3c_random": "B3c: random shape"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _color(system: str) -> str:
    if system == "method":
        return PALETTE[0]
    if system.startswith("A"):
        return PALETTE[6]
    if system.startswith("B1"):
        return PALETTE[1]
    if system.startswith("B2"):
        return PALETTE[2]
    return INK2


def fig_dir() -> Path:
    d = REPO_ROOT / "docs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fig_systems(analysis: dict, key: str = "systems", name: str = "fig_systems.png") -> Path:
    s = analysis[key]
    order = [k for k in SYSTEM_ORDER if k in s]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(order) + 1.2), sharey=True)
    for ax, metric, title in ((axes[0], "fine_acc", "Fine family accuracy"), (axes[1], "median_chamfer", "Median Chamfer to ground truth (lower is better)")):
        for i, k in enumerate(order[::-1]):
            est, lo, hi = s[k][metric]
            ax.plot([lo, hi], [i, i], color=_color(k), linewidth=2, solid_capstyle="round")
            ax.plot(est, i, "o", color=_color(k), markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_xlim(0, 1)
    axes[1].set_xlim(left=0)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{SYSTEM_LABEL.get(k, k)} (n={s[k]['n']})" for k in order[::-1]], color=INK, fontsize=9)
    fig.text(0.01, 0.005, "Dots: estimate; bars: 95% bootstrap CI. Baselines use the baseline subset of the test set.", fontsize=8, color=INK2)
    fig.tight_layout()
    p = fig_dir() / name
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_by_bin(analysis: dict, key: str, xlabels: list[str], title: str, name: str) -> Path:
    data = analysis[key]
    # paired: every line uses the same baseline-subset clouds. A3 (no refinement) is left out: refinement never changes the
    # selected family, so its accuracy equals the method's by construction and its line would hide the method's.
    systems = [k for k in ("method_subset", "B1_classical", "B2_features") if k in data]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k in systems:
        xs = sorted(int(b) for b in data[k])
        ys = [data[k][str(b)]["fine_acc"] if str(b) in data[k] else data[k][b]["fine_acc"] for b in xs]
        ax.plot(xs, ys, "-o", color=_color("method" if k == "method_subset" else k), linewidth=2, markersize=7,
                markeredgecolor="white", label=SYSTEM_LABEL[k])
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fine family accuracy", color=INK2)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    ax.grid(axis="y", color=GRID)
    fig.tight_layout()
    p = fig_dir() / name
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_confusion(analysis: dict, names: list[str], name: str = "fig_confusion.png") -> Path:
    conf = analysis["confusion_method"]            # {pred: {true: count}}
    m = np.zeros((len(names), len(names)))
    for j, pr in enumerate(names):
        for i, tr in enumerate(names):
            m[i, j] = conf.get(pr, {}).get(tr, 0)
    m = m / m.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted family", color=INK2)
    ax.set_ylabel("True family", color=INK2)
    for i in range(len(names)):
        for j in range(len(names)):
            if m[i, j] >= 0.1:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if m[i, j] > 0.5 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised rate")
    ax.set_title("Method: confusion matrix on the synthetic test set", fontsize=10, loc="left")
    fig.tight_layout()
    p = fig_dir() / name
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_training(curve_path: Path, name: str = "fig_training.png") -> Path:
    curve = json.load(open(curve_path))
    tr = [c for c in curve if "total" in c]
    va = [c for c in curve if "val_acc" in c]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    steps = [c["step"] for c in tr]
    # running mean over 10 logged points smooths the batch-to-batch N variation
    k = max(1, min(10, len(tr)))
    sm = lambda v: np.convolve(v, np.ones(k) / k, mode="valid")
    axes[0].plot(steps, [c["total"] for c in tr], color=GRID, linewidth=1)
    axes[0].plot(steps[k - 1:], sm([c["total"] for c in tr]), color=PALETTE[0], linewidth=2, label="total loss")
    axes[0].plot(steps[k - 1:], sm([c["geo"] for c in tr]), color=PALETTE[1], linewidth=2, label="geometric terms")
    axes[0].set_xlabel("step", color=INK2)
    axes[0].set_title("Training loss (thin: raw, thick: running mean of 10 logs)", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot([c["step"] for c in va], [c["val_acc"] for c in va], "-o", color=PALETTE[0], linewidth=2, label="val family accuracy")
    axes[1].plot([c["step"] for c in va], [c["val_cd_teacher"] for c in va], "-o", color=PALETTE[1], linewidth=2,
                 label="val Chamfer (true-family head)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("step", color=INK2)
    axes[1].set_title("Validation (fixed `val` stream)", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        _style(ax)
        ax.grid(axis="y", color=GRID)
    fig.tight_layout()
    p = fig_dir() / name
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def _scatter3d(ax, pts, labels, shape, title, cyclic: bool):
    pts, shape = np.asarray(pts), np.asarray(shape)
    try:
        c = np.asarray(labels, dtype=float)
    except ValueError:
        u = list(dict.fromkeys(labels))
        c = np.array([u.index(l) for l in labels], dtype=float)
    cmap = "twilight" if cyclic else "viridis"
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, cmap=cmap, s=8, depthshade=False, alpha=0.9)
    if len(shape) <= 12:
        ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2], c=INK, s=40, marker="x")
    else:
        ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2], c=PALETTE[7], s=1.2, alpha=0.35)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])


def fig_real(results_path: Path, viz_dir: Path, ids: list[str], name: str = "fig_real.png", cols: int = 4) -> Path:
    res = {r["id"]: r for r in json.load(open(results_path))}
    ids = [i for i in ids if i in res]
    rows = int(np.ceil(len(ids) / cols))
    fig = plt.figure(figsize=(3.2 * cols, 3.1 * rows))
    for k, did in enumerate(ids):
        v = json.load(open(viz_dir / f"{did}.json"))
        r = res[did]
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        cyc = any(s in did for s in ("weekday", "months", "days", "grokking", "colors"))
        stab = r.get("stability", {}).get("fine_majority", "-")
        _scatter3d(ax, v["points"], v["labels"], v["shape"], f"{did}\nfit: {r['family']} (p={r['probs'][r['family']]:.2f}); "
                   f"subsample majority: {stab}", cyc)
    fig.text(0.01, 0.005, "Points: top-3 PCA of activations, coloured by concept value. Red: fitted shape (crosses: point-set vertices).",
             fontsize=8, color=INK2)
    fig.tight_layout()
    p = fig_dir() / name
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_reliability(probs_path: Path, cal: dict | None = None, name: str = "fig_reliability.png", bins: int = 15) -> Path:
    z = np.load(probs_path)
    probs, labels = z["probs"], z["labels"]
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            xs.append(conf[m].mean()), ys.append((pred[m] == labels[m]).mean()), ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5)
    ax.plot(xs, ys, "-o", color=PALETTE[0], linewidth=2, markersize=6, markeredgecolor="white")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (temperature-scaled)", color=INK2)
    ax.set_ylabel("accuracy", color=INK2)
    title = "Reliability diagram, method"
    if cal:
        title += f"\nECE {cal['before']['ece']:.3f} (T = 1) -> {cal['after']['ece']:.3f} (T = {cal['temperature']:.2f})"
    ax.set_title(title, fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    p = fig_dir() / name
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p
