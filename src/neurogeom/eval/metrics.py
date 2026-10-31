"""Per-sample evaluation metrics on synthetic data with known ground truth.

Global context: preregistered metrics (reports/preregistration.md Sec. 4):
- fine / coarse / acceptable-superset family accuracy (macro-F1 is computed from the per-sample table);
- Chamfer distance between the predicted shape and the noise-free ground-truth shape (primary
  geometric metric; symmetric, unsquared, in units of the normalised cloud's RMS radius);
- residual: mean distance from the observed points to the predicted shape;
- when the family is right: unsigned axis angle error (degrees) and |log scale ratio|.
"""

from __future__ import annotations

import math

import torch

from neurogeom.geometry.chamfer import chamfer, residual
from neurogeom.shapes.registry import COARSE, FAMILIES, NAMES, SUPERSETS

EVAL_DENSE = {"curve": 256, "surface_side": 24, "blob": 256}


@torch.no_grad()
def shape_metrics(pred_fi: list[int], raws: list[torch.Tensor], batch: dict) -> list[dict]:
    """pred_fi[i], raws[i]: predicted family and raw parameters for sample i of `batch` (on device)."""
    b = len(pred_fi)
    out = [dict() for _ in range(b)]
    fam_true = batch["family"].tolist()
    for fi in set(pred_fi):
        rows = [i for i in range(b) if pred_fi[i] == fi]
        r = torch.tensor(rows, device=batch["x"].device)
        f = FAMILIES[fi]
        p = f.decode(torch.stack([raws[i] for i in rows]).to(batch["x"].device))
        pts, w = f.dense(p, EVAL_DENSE)
        cd = chamfer(pts, w, batch["gt_dense"][r], batch["gt_w"][r])
        res = residual(batch["x"][r], batch["mask"][r], pts, w)
        axis, logscale = f.aux(p)
        for j, i in enumerate(rows):
            t = fam_true[i]
            rec = {"family_pred": NAMES[fi], "family_true": NAMES[t], "correct": fi == t,
                   "coarse_correct": COARSE[fi] == COARSE[t],
                   "superset_ok": fi == t or NAMES[fi] in SUPERSETS.get(NAMES[t], set()),
                   "chamfer": float(cd[j]), "residual": float(res[j]), "axis_err_deg": math.nan, "logscale_err": math.nan}
            if fi == t:
                if axis is not None and bool(batch["has_axis"][i]):
                    c = float((axis[j] * batch["gt_axis"][i]).sum().abs().clamp(max=1.0))
                    rec["axis_err_deg"] = math.degrees(math.acos(c))
                rec["logscale_err"] = float((logscale[j] - batch["gt_logscale"][i]).abs())
            out[i] = rec
    return out


def sample_meta(cell: dict, ii: torch.Tensor, batch: dict) -> list[dict]:
    return [{"n_bin": cell["n_bin"], "noise_bin": cell["noise_bin"], "sample_id": f"{cell['n_bin']}-{cell['noise_bin']}-{int(i)}",
             "N": int(batch["n"][j]), "sigma": float(batch["sigma"][j]), "embed_dim": int(batch["embed_dim"][j])}
            for j, i in enumerate(ii.tolist())]
