"""Preregistered analysis: summaries with CIs and hypothesis tests H1-H5 from the per-sample tables.

Global context: reads results/<run>/synth/per_sample.csv (one row per sample x system) and
results/<run>/real/real_results.json, and writes results/<run>/analysis.json. Every number in README,
reports and paper comes from that file via report/numbers.py (no hand-typed numbers).
Comparisons with baselines are paired on the baseline subset (identical samples); comparisons with
ablations that are separate models (A1, A2, A4, A6) are paired on the full test set (same samples,
same order, from the fixed `test` stream).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurogeom.eval.stats import (binom_ci, bootstrap_ci, bootstrap_ci_pairs, holm, macro_f1, paired_bootstrap,
                                  paired_bootstrap_f1, wilcoxon_p)
from neurogeom.logging_utils import get_logger

log = get_logger(__name__)


def system_summary(df: pd.DataFrame, reps: int) -> dict:
    out = {}
    for sysname, g in df.groupby("system"):
        acc = bootstrap_ci(g.correct.astype(float), reps=reps)
        f1 = bootstrap_ci_pairs(g.family_true, g.family_pred, macro_f1, reps=min(reps, 500))
        cd = bootstrap_ci(g.chamfer.values, stat=np.median, reps=reps)
        out[sysname] = {
            "n": int(len(g)), "fine_acc": acc, "macro_f1": f1, "coarse_acc": bootstrap_ci(g.coarse_correct.astype(float), reps=reps),
            "superset_acc": bootstrap_ci(g.superset_ok.astype(float), reps=reps), "median_chamfer": cd,
            "mean_chamfer": float(g.chamfer.mean()), "mean_residual": float(g.residual.mean()),
            "median_axis_err_deg": float(np.nanmedian(g.axis_err_deg)) if g.axis_err_deg.notna().any() else None,
            "median_logscale_err": float(np.nanmedian(g.logscale_err)) if g.logscale_err.notna().any() else None,
        }
    return out


def breakdown(df: pd.DataFrame, key: str) -> dict:
    return {s: {int(k): {"fine_acc": float(v.correct.mean()), "median_chamfer": float(v.chamfer.median()), "n": int(len(v))}
                for k, v in g.groupby(key)} for s, g in df.groupby("system")}


def per_family(df: pd.DataFrame) -> dict:
    return {s: {f: {"recall": float(v.correct.mean()), "median_chamfer": float(v.chamfer.median())} for f, v in g.groupby("family_true")}
            for s, g in df.groupby("system")}


def confusion(df: pd.DataFrame, system: str) -> dict:
    g = df[df.system == system]
    return pd.crosstab(g.family_true, g.family_pred).to_dict()


def paired(df: pd.DataFrame, a: str, b: str, reps: int) -> dict | None:
    ga, gb = df[df.system == a].set_index("sample_id"), df[df.system == b].set_index("sample_id")
    common = ga.index.intersection(gb.index)
    if len(common) == 0:
        return None
    ga, gb = ga.loc[common], gb.loc[common]
    f1 = paired_bootstrap_f1(ga.family_true.values, ga.family_pred.values, gb.family_pred.values, reps=min(reps, 1000))
    cd = paired_bootstrap(ga.chamfer.values, gb.chamfer.values, stat=np.median, reps=reps)
    return {"n": int(len(common)), "macro_f1_diff": f1, "median_chamfer_diff": cd,
            "fine_acc_diff": paired_bootstrap(ga.correct.astype(float).values, gb.correct.astype(float).values, reps=reps),
            "wilcoxon_p_chamfer": wilcoxon_p(ga.chamfer.values, gb.chamfer.values)}


def _f1_pvalue(lo: float, hi: float, est: float) -> float:
    """Two-sided bootstrap p-value approximation from the CI (normal approximation of the bootstrap SE)."""
    from scipy.stats import norm
    se = max((hi - lo) / (2 * 1.96), 1e-12)
    return float(2 * (1 - norm.cdf(abs(est) / se)))


def hypotheses_synth(df: pd.DataFrame, reps: int, ablation_frames: dict) -> dict:
    h = {}
    c1 = paired(df, "method", "B1_classical", reps)
    if c1:
        est, lo, hi = c1["macro_f1_diff"]
        h["H1"] = {"comparison": "macro-F1(method) - macro-F1(B1)", "estimate": est, "ci": [lo, hi], "supported": lo > 0,
                   "p": _f1_pvalue(lo, hi, est), "n": c1["n"]}
        est, lo, hi = c1["median_chamfer_diff"]
        h["H2"] = {"comparison": "median Chamfer(method) - median Chamfer(B1)", "estimate": est, "ci": [lo, hi],
                   "falsified": lo > 0, "supported": lo <= 0 and est <= 0, "p": c1["wilcoxon_p_chamfer"], "n": c1["n"]}
    c3 = paired(df, "method", "A3_norefine", reps)
    if c3:
        est, lo, hi = c3["median_chamfer_diff"]
        h["H5b_refinement"] = {"comparison": "median Chamfer(method) - median Chamfer(no refinement)", "estimate": est, "ci": [lo, hi],
                               "supported": hi < 0, "p": c3["wilcoxon_p_chamfer"], "n": c3["n"]}
    if "A1_paramloss" in ablation_frames:
        both = pd.concat([df[df.system == "method"], ablation_frames["A1_paramloss"]])
        c = paired(both, "method", "A1_paramloss", reps)
        est, lo, hi = c["median_chamfer_diff"]
        h["H5a_chamfer_loss"] = {"comparison": "median Chamfer(method) - median Chamfer(parameter-space loss)", "estimate": est,
                                 "ci": [lo, hi], "supported": hi < 0, "p": c["wilcoxon_p_chamfer"], "n": c["n"]}
    adj = holm({k: v["p"] for k, v in h.items() if v.get("p") is not None})
    for k, p in adj.items():
        h[k]["p_holm"] = p
    return h


def hypotheses_real(results: list[dict], seed: int = 0) -> dict:
    pre = [r for r in results if r["id"].startswith("R") and r["meta"].get("prereg") and r["meta"].get("expected_coarse")
           and not r["id"].endswith("_means")]
    lab = lambda r: r.get("stability", {}).get("coarse_majority", r["coarse"])
    agree = [lab(r) == r["meta"]["expected_coarse"] for r in pre]
    k, n = int(sum(agree)), len(agree)
    rng = np.random.default_rng(seed)
    preds = np.array([lab(r) for r in pre])
    exp = np.array([r["meta"]["expected_coarse"] for r in pre])
    null = [int((rng.permutation(preds) == exp).sum()) for _ in range(10000)]
    p_perm = float((np.array(null) >= k).mean())
    h3 = {"agree": k, "n": n, "rate": k / max(n, 1), "ci": binom_ci(k, n) if n else None, "p_permutation": p_perm,
          "supported": n > 0 and k / n >= 0.5 and p_perm < 0.05,
          "per_dataset": {r["id"]: {"predicted_coarse": lab(r), "expected_coarse": r["meta"]["expected_coarse"],
                                    "predicted_fine": r.get("stability", {}).get("fine_majority", r["family"]),
                                    "expected_fine": r["meta"].get("expected_fine")} for r in pre}}
    c2 = [r["C2_matched_gaussian"].get("blob", 0.0) for r in results if "C2_matched_gaussian" in r and r["id"].startswith("R")]
    c1 = [r for r in results if r["id"].startswith("C1_")]
    ring = float(np.mean([r["family"] == "ringing" for r in c1])) if c1 else None
    helix = float(np.mean([r["family"] == "helix" for r in c1])) if c1 else None
    blob_rate = float(np.mean(c2)) if c2 else None
    h4 = {"C2_blob_rate_mean": blob_rate, "C2_per_dataset": {r["id"]: r["C2_matched_gaussian"] for r in results if "C2_matched_gaussian" in r},
          "C1_ringing_rate": ring, "C1_helix_rate": helix, "C1_labels": {r["id"]: r["family"] for r in c1},
          "supported": blob_rate is not None and blob_rate >= 0.8 and ring is not None and ring > helix}
    c3 = {r["id"]: r["family"] for r in results if r["id"].startswith("C3_")}
    return {"H3": h3, "H4": h4, "C3_random_init": c3}


def run_analysis(cfg: dict, ablation_runs: dict | None = None) -> dict:
    res_dir = Path(cfg["paths"]["results"])
    reps = cfg["eval"]["bootstrap_reps"]
    df = pd.read_csv(res_dir / "synth" / "per_sample.csv")
    abl = {}
    for name, run in (ablation_runs or {}).items():
        p = res_dir.parent / run / "synth" / "per_sample.csv"
        if p.exists():
            a = pd.read_csv(p)
            a = a[a.system == "method"].copy()
            a["system"] = name
            abl[name] = a
    full = pd.concat([df] + list(abl.values()))
    out = {"systems": system_summary(full, reps), "by_n_bin": breakdown(full, "n_bin"), "by_noise_bin": breakdown(full, "noise_bin"),
           "per_family": per_family(full), "confusion_method": confusion(df, "method"),
           "hypotheses": hypotheses_synth(df, reps, abl)}
    if "ambiguous" in df:
        ids = set(df[(df.system == "B1_classical") & (df.ambiguous == True)].sample_id)  # noqa: E712
        sub = df[~df.sample_id.isin(ids) & df.sample_id.isin(set(df[df.system == "B1_classical"].sample_id))]
        out["systems_unambiguous_subset"] = system_summary(sub, min(reps, 500))
        out["ambiguous_fraction"] = float(len(ids) / max(1, (df.system == "B1_classical").sum()))
    rp = res_dir / "real" / "real_results.json"
    if rp.exists():
        out["real"] = hypotheses_real(json.load(open(rp)))
    json.dump(out, open(res_dir / "analysis.json", "w"), indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    log.info("analysis written: %s", res_dir / "analysis.json")
    return out
