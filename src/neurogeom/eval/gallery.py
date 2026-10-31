"""Synthetic visual examples: one test cloud per family with the method's fit and the classical fit.

Global context: README/paper/site visualisation of method vs baseline outputs on synthetic data with
known ground truth (user requirement: "visualizations output examples for new method(s), baseline(s)").
Samples are the first test samples of each family in the (N-bin 2, noise-bin 1) cell: N in 65..256,
relative noise 0.02..0.06 - a typical, not cherry-picked, setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from neurogeom.baselines.classical import fit_classical  # noqa: E402
from neurogeom.data.testsets import build_test_set  # noqa: E402
from neurogeom.infer.predict import describe_fit, fit_batch  # noqa: E402
from neurogeom.logging_utils import get_logger  # noqa: E402
from neurogeom.report.figures import INK, PALETTE, fig_dir  # noqa: E402
from neurogeom.shapes.registry import FAMILIES, NAMES  # noqa: E402

log = get_logger(__name__)
VIZ_DENSE = {"curve": 300, "surface_side": 26, "blob": 256}


def _dense_np(fi: int, raw: torch.Tensor) -> np.ndarray:
    f = FAMILIES[fi]
    pts, w = f.dense(f.decode(raw.unsqueeze(0)), VIZ_DENSE)
    return pts[0][w[0] > 0].detach().cpu().numpy()


def build_gallery(model, cfg: dict, temperature: float, device, out_json: Path) -> Path:
    cells = build_test_set(cfg, device=device)
    cell = [c for c in cells if c["n_bin"] == 2 and c["noise_bin"] == 1][0]
    idx = [int(torch.nonzero(cell["family"] == fi)[0]) for fi in range(len(FAMILIES))]
    b = {k: (v[idx].to(device) if torch.is_tensor(v) else v) for k, v in cell.items()}
    fits = fit_batch(model, b["x"], b["mask"], cfg["infer"], temperature)
    with torch.enable_grad():
        cands, sel = fit_classical(b["x"], b["mask"], cfg["eval"]["classical_steps"], cfg["infer"]["refine_back_weight"],
                                   restarts=cfg["eval"]["classical_restarts"])
    recs = []
    fig = plt.figure(figsize=(16, 9))
    for i, fi_true in enumerate(range(len(FAMILIES))):
        m = b["mask"][i].cpu().numpy()
        pts = b["x"][i].cpu().numpy()[m]
        gt = b["gt_dense"][i][b["gt_w"][i] > 0].cpu().numpy()
        fit = fits[i]
        c = cands[int(sel["cand_index"][i])]
        meth = _dense_np(fit.family, fit.raw)
        clas = _dense_np(c["fi"], c["raw"][i])
        recs.append({"id": f"synth_{NAMES[fi_true]}", "N": int(m.sum()), "D": 3, "top3": 1.0, "family": NAMES[fit.family],
                     "probs": {NAMES[j]: float(p) for j, p in enumerate(fit.probs)}, "params": describe_fit(fit),
                     "majority": NAMES[fit.family], "expected_coarse": FAMILIES[fi_true].coarse, "expected_fine": [NAMES[fi_true]],
                     "source": f"synthetic test sample (true family: {NAMES[fi_true]}); classical B1 picked {NAMES[c['fi']]}",
                     "flags": "", "cyclic": False, "points": pts.round(5).tolist(), "labels": [str(k) for k in range(int(m.sum()))],
                     "shape": meth.round(5).tolist()})
        ax = fig.add_subplot(3, 5, i + 1, projection="3d")
        ax.scatter(*pts.T, s=4, color=INK, alpha=0.5, depthshade=False)
        ax.scatter(*gt.T, s=0.6, color=PALETTE[2], alpha=0.35)
        ax.scatter(*meth.T, s=0.8, color=PALETTE[0], alpha=0.6)
        ax.scatter(*clas.T, s=0.8, color=PALETTE[1], alpha=0.4)
        ok = lambda f: "ok" if f == fi_true else "x"
        ax.set_title(f"true: {NAMES[fi_true]}\nmethod: {NAMES[fit.family]} ({ok(fit.family)}) | B1: {NAMES[c['fi']]} ({ok(c['fi'])})", fontsize=7.5)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    fig.text(0.01, 0.01, "Black: input points. Green: ground-truth shape. Blue: method fit. Orange: classical B1 fit. "
             "First test sample of each family, N 65-256, relative noise 0.02-0.06.", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir() / "fig_gallery_synth.png", dpi=150)
    plt.close(fig)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(out_json, "w"))
    log.info("gallery written: %s", out_json)
    return out_json
