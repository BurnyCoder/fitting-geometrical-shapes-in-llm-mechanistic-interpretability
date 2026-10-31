"""Phase eval-real: fit every real activation cloud and control, with the preregistered diagnostics.

Global context (reports/preregistration.md, Sec. 3-5). Per dataset:
  - full-cloud fit: selected family, canonical parameters in top-3 PCA coordinates, class probabilities;
  - stability: 200 random 80% subsamples without replacement (PCA recomputed on each subsample) ->
    fine/coarse label frequencies and the majority label;
  - 5-fold held-out point residual of the method, of the classical baseline B1, and of the helix and
    ringing families (helix claims need p(helix)/p(ringing) >= 3 AND a lower held-out residual);
  - PCA spectrum lambda_1..lambda_6, explained-variance ratios and the lambda_3/lambda_4 gap flag;
  - C2 control: matched-covariance Gaussians with the same N and D;
  - OOD flag: residual above the synthetic 95th percentile of the method's correct fits at that N.
Outputs: results/<run>/real/real_results.json, real_summary.csv and viz/<dataset>.json for figures/site.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurogeom.baselines.classical import fit_classical
from neurogeom.eval.metrics import EVAL_DENSE
from neurogeom.geometry.chamfer import nearest_dist
from neurogeom.geometry.preprocess import fit_frame
from neurogeom.infer.predict import describe_fit, fit_batch
from neurogeom.logging_utils import get_logger
from neurogeom.realdata.common import load_cloud, real_dir
from neurogeom.realdata.controls import matched_gaussian
from neurogeom.shapes.registry import COARSE, COARSE_NAMES, FAMILIES, INDEX, NAMES

log = get_logger(__name__)


def _pad_batch(clouds: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(c) for c in clouds)
    x = torch.zeros(len(clouds), n, 3, device=device)
    m = torch.zeros(len(clouds), n, dtype=torch.bool, device=device)
    for i, c in enumerate(clouds):
        x[i, :len(c)] = torch.as_tensor(c, dtype=torch.float32, device=device)
        m[i, :len(c)] = True
    return x, m


def _heldout_dist(fi: int, raw: torch.Tensor, pts: np.ndarray, device) -> float:
    f = FAMILIES[fi]
    s, w = f.dense(f.decode(raw.unsqueeze(0).to(device)), EVAL_DENSE)
    x = torch.as_tensor(pts, dtype=torch.float32, device=device).unsqueeze(0)
    return float(nearest_dist(x, s, w > 0).mean())


def heldout_residuals(model, y: np.ndarray, cfg: dict, temperature: float, device, folds: int = 5, seed: int = 0) -> dict:
    """K-fold: fit on (K-1)/K of the normalised points, mean distance of held-out points to the fit."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    splits = [(np.concatenate([parts[j] for j in range(folds) if j != k]), parts[k]) for k in range(folds)]
    splits = [(tr, te) for tr, te in splits if len(tr) >= 4 and len(te) > 0]
    res = {"method": [], "B1_classical": [], "helix": [], "ringing": []}
    if not splits:
        return {k: float("nan") for k in res}
    # all folds are fitted in one padded batch (fitting is launch-bound, so a batch of 5 costs ~ one fit)
    x, m = _pad_batch([y[tr] for tr, _ in splits], device)
    fits = fit_batch(model, x, m, cfg["infer"], temperature)
    forced = fit_batch(model, x, m, cfg["infer"], temperature, families=[INDEX["helix"], INDEX["ringing"]])
    with torch.enable_grad():
        cands, sel = fit_classical(x, m, cfg["eval"]["classical_steps"], cfg["infer"]["refine_back_weight"],
                                   restarts=cfg["eval"]["classical_restarts"])
    for k, (_, te) in enumerate(splits):
        res["method"].append(_heldout_dist(fits[k].family, fits[k].raw, y[te], device))
        for name, (fi, _, _), raw in zip(("helix", "ringing"), forced[k].cand, forced[k].cand_raws):
            res[name].append(_heldout_dist(fi, raw, y[te], device))
        c = cands[int(sel["cand_index"][k])]
        res["B1_classical"].append(_heldout_dist(c["fi"], c["raw"][k], y[te], device))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in res.items()}


def subsample_labels(model, x_raw: np.ndarray, cfg: dict, temperature: float, device, reps: int, frac: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(x_raw)
    k = max(5, int(round(frac * n)))
    clouds = [fit_frame(x_raw[rng.choice(n, k, replace=False)])[0] for _ in range(reps)]
    fams = []
    for s in range(0, reps, 100):
        x, m = _pad_batch(clouds[s:s + 100], device)
        fams += [f.family for f in fit_batch(model, x, m, cfg["infer"], temperature)]
    fine = pd.Series([NAMES[f] for f in fams]).value_counts(normalize=True).to_dict()
    coarse = pd.Series([COARSE_NAMES[COARSE[f]] for f in fams]).value_counts(normalize=True).to_dict()
    return {"fine_freq": fine, "coarse_freq": coarse, "fine_majority": max(fine, key=fine.get),
            "coarse_majority": max(coarse, key=coarse.get), "n_sub": k, "reps": reps}


def ood_threshold(cfg: dict) -> dict:
    """Synthetic 95th percentile of the method's residual for correct fits, per N-bin."""
    p = Path(cfg["paths"]["results"]) / "synth" / "per_sample.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df = df[(df.system == "method") & df.correct]
    return {int(b): float(g.residual.quantile(0.95)) for b, g in df.groupby("n_bin")}


def n_bin_of(n: int, bins) -> int:
    for i, (lo, hi) in enumerate(bins):
        if n <= hi:
            return i
    return len(bins) - 1


def eval_dataset(model, path: Path, cfg: dict, temperature: float, device, thresholds: dict, light: bool = False) -> dict:
    x_raw, labels, meta = load_cloud(path)
    did = path.stem
    y, frame = fit_frame(x_raw)
    x, m = _pad_batch([y], device)
    fit = fit_batch(model, x, m, cfg["infer"], temperature)[0]
    desc = describe_fit(fit, frame)
    ev = frame.eigvals
    out = {"id": did, "meta": meta, "N": int(len(x_raw)), "D": int(x_raw.shape[1]), "family": NAMES[fit.family],
           "coarse": COARSE_NAMES[COARSE[fit.family]], "params": desc, "probs": {NAMES[i]: float(p) for i, p in enumerate(fit.probs)},
           "eigvals": ev.tolist(), "explained_ratio": frame.explained_ratio.tolist(),
           "top3_explained": float(frame.explained_ratio[:3].sum()),
           "gap_l3_l4": float(ev[2] / ev[3]) if len(ev) > 3 and ev[3] > 0 else float("inf")}
    f_sel = FAMILIES[fit.family]
    s_pts, s_w = f_sel.dense(f_sel.decode(fit.raw.unsqueeze(0).to(device)), EVAL_DENSE)
    out["residual_normalised"] = float(nearest_dist(x, s_pts, s_w > 0).mean())   # mean point->shape distance, RMS units
    out["gap_flag"] = out["gap_l3_l4"] < 1.5
    nb = n_bin_of(len(y), cfg["eval"]["n_bins"])
    out["ood"] = bool(thresholds and out["residual_normalised"] > thresholds.get(nb, np.inf))
    out["p_helix_over_ringing"] = float(fit.probs[INDEX["helix"]] / max(fit.probs[INDEX["ringing"]], 1e-9))
    if not light:
        ec = cfg["eval"]
        out["stability"] = subsample_labels(model, x_raw, cfg, temperature, device, ec["real_subsample_reps"], ec["real_subsample_frac"])
        out["heldout"] = heldout_residuals(model, y, cfg, temperature, device)
        rng = np.random.default_rng(0)
        gs = [fit_frame(matched_gaussian(x_raw, rng))[0] for _ in range(ec.get("matched_gaussian_reps", 20))]
        xg, mg = _pad_batch(gs, device)
        c2 = [NAMES[f.family] for f in fit_batch(model, xg, mg, cfg["infer"], temperature)]
        out["C2_matched_gaussian"] = pd.Series(c2).value_counts(normalize=True).to_dict()
        # preregistered rule: a helix claim needs p(helix)/p(ringing) >= 3 AND a lower held-out residual than ringing
        helix_pref = bool(out["p_helix_over_ringing"] >= 3 and out["heldout"]["helix"] < out["heldout"]["ringing"])
        out["helix_vs_ringing"] = {"p_ratio": out["p_helix_over_ringing"], "heldout_helix": out["heldout"]["helix"],
                                   "heldout_ringing": out["heldout"]["ringing"], "helix_preferred": helix_pref}
        out["helix_claim_supported"] = bool(out["family"] == "helix" and helix_pref) if out["family"] == "helix" else None
    # visualisation payload: normalised points (in PCA coordinates) + dense samples of the fitted shape
    f = FAMILIES[fit.family]
    pts, w = f.dense(f.decode(fit.raw.unsqueeze(0).to(device)), {"curve": 400, "surface_side": 30, "blob": 256})
    out["viz"] = {"points": (y * frame.scale).round(5).tolist(), "labels": [str(l) for l in labels],
                  "shape": (pts[0][w[0] > 0].cpu().numpy() * frame.scale).round(5).tolist()}
    log.info("%s: N=%d D=%d -> %s (p=%.2f) | top3 var %.2f | stability %s", did, out["N"], out["D"], out["family"],
             fit.probs[fit.family], out["top3_explained"], out.get("stability", {}).get("fine_majority"))
    return out


def run_real_eval(cfg: dict, light_controls: bool = True) -> list[dict]:
    from neurogeom.train.loop import load_model
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(cfg["paths"]["checkpoints"]) / "final.pt", device)
    meta_p = Path(cfg["paths"]["results"]) / "synth" / "meta.json"
    temperature = json.load(open(meta_p))["temperature"] if meta_p.exists() else 1.0
    thresholds = ood_threshold(cfg)
    out_dir = Path(cfg["paths"]["results"]) / "real"
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)
    results = []
    for p in sorted(real_dir(cfg).glob("*.npz")):
        light = light_controls and p.stem.startswith("C1_") and not p.stem.endswith("_0")
        r = eval_dataset(model, p, cfg, temperature, device, thresholds, light=light)
        json.dump(r.pop("viz"), open(out_dir / "viz" / f"{p.stem}.json", "w"))
        results.append(r)
    json.dump(results, open(out_dir / "real_results.json", "w"), indent=2, default=str)
    rows = [{"id": r["id"], "N": r["N"], "D": r["D"], "family": r["family"], "coarse": r["coarse"],
             "expected_coarse": r["meta"].get("expected_coarse"), "expected_fine": r["meta"].get("expected_fine"),
             "prereg": r["meta"].get("prereg"), "p_selected": r["probs"][r["family"]], "top3_explained": r["top3_explained"],
             "gap_flag": r["gap_flag"], "ood": r["ood"], "stability_majority": r.get("stability", {}).get("fine_majority"),
             "stability_coarse_majority": r.get("stability", {}).get("coarse_majority")} for r in results]
    pd.DataFrame(rows).to_csv(out_dir / "real_summary.csv", index=False)
    log.info("eval-real done: %d datasets", len(results))
    return results
