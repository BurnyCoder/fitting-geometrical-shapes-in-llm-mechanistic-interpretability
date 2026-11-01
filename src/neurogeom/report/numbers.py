"""Phase report: regenerate every table, figure and number used in README / reports / paper.

Global context (user requirement): "every number comes from a script that reads results files; one
command regenerates all tables and figures". This module
  1. runs the preregistered analysis (eval/analysis.py) -> results/<run>/analysis.json,
  2. renders figures (report/figures.py) -> docs/figures/*.png,
  3. writes markdown tables -> reports/generated/*.md and replaces the blocks between
     <!-- BEGIN:<name> --> and <!-- END:<name> --> markers in README.md and reports/*.md,
  4. writes LaTeX macros -> paper/generated/numbers.tex and tables -> paper/generated/*.tex.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from neurogeom.config import REPO_ROOT
from neurogeom.eval.analysis import run_analysis
from neurogeom.logging_utils import get_logger
from neurogeom.realdata.common import is_prereg
from neurogeom.report import figures
from neurogeom.shapes.registry import NAMES

log = get_logger(__name__)
GEN_MD = REPO_ROOT / "reports" / "generated"
GEN_TEX = REPO_ROOT / "paper" / "generated"


def fmt_ci(t, pct: bool = False, nd: int = 3) -> str:
    est, lo, hi = t
    if pct:
        return f"{100 * est:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def systems_table(s: dict, order: list[str]) -> str:
    rows = ["| System | n | Fine accuracy [95% CI] | Macro-F1 [95% CI] | Coarse accuracy | Superset-OK accuracy | Median Chamfer to GT [95% CI] | Mean residual | Median axis error (deg) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        if k not in s:
            continue
        v = s[k]
        ax = "-" if v["median_axis_err_deg"] is None else f"{v['median_axis_err_deg']:.1f}"
        rows.append(f"| {figures.SYSTEM_LABEL.get(k, k)} | {v['n']} | {fmt_ci(v['fine_acc'], True)} | {fmt_ci(v['macro_f1'])} | "
                    f"{100 * v['coarse_acc'][0]:.1f}% | {100 * v['superset_acc'][0]:.1f}% | {fmt_ci(v['median_chamfer'])} | "
                    f"{v['mean_residual']:.3f} | {ax} |")
    return "\n".join(rows)


def hypotheses_table(h: dict, real: dict | None) -> str:
    rows = ["| Hypothesis | Comparison | Estimate [95% CI] | p (Holm) | n | Verdict |", "|---|---|---|---|---|---|"]
    for k in ("H1", "H2", "H5a_chamfer_loss", "H5b_refinement"):
        if k in h:
            v = h[k]
            verdict = "supported" if v["supported"] else ("falsified" if v.get("falsified") or (not v["supported"] and k != "H2") else "not supported")
            rows.append(f"| {k} | {v['comparison']} | {v['estimate']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] | "
                        f"{fmt_p(v.get('p_holm', v.get('p', float('nan'))))} | {v['n']} | {verdict} |")
    if real:
        h3, h4 = real["H3"], real["H4"]
        ci = h3["ci"] or [float("nan")] * 2
        rows.append(f"| H3 | Coarse-label agreement (subsample majority) with literature on preregistered real datasets | {h3['agree']}/{h3['n']} = {100 * h3['rate']:.0f}% "
                    f"[{100 * ci[0]:.0f}, {100 * ci[1]:.0f}] (exact binomial) | perm. p = {h3['p_permutation']:.3g} | {h3['n']} | "
                    f"{'supported' if h3['supported'] else 'not supported'} |")
        rows.append(f"| H4 | Controls: matched Gaussians -> blob (mean over {_n_real(h4)} real datasets); ordinal processes -> ringing (not helix) | blob rate "
                    f"{_pct(h4['C2_blob_rate_mean'])}; ringing {_pct(h4['C1_ringing_rate'])} vs helix {_pct(h4['C1_helix_rate'])} | - | - | "
                    f"{'supported' if h4['supported'] else 'not supported'} |")
    return "\n".join(rows)


RUN_DESC = {"main": ("Set Transformer", "CE + Chamfer + residual + aux", "full synthetic mix"),
            "A0": ("Set Transformer", "CE + Chamfer + residual + aux", "full synthetic mix (short budget)"),
            "A1": ("Set Transformer", "CE + parameter-space loss", "full synthetic mix (short budget)"),
            "A2": ("Set Transformer", "CE + Chamfer + aux (no residual)", "full synthetic mix (short budget)"),
            "A4": ("DeepSets", "CE + Chamfer + residual + aux", "full synthetic mix (short budget)"),
            "A6": ("Set Transformer", "CE + Chamfer + residual + aux", "no R^D embedding (short budget)")}


def models_table(results_root: Path) -> str:
    """One row per trained network found under results/*/train/summary.json (smoke runs excluded)."""
    import pandas as pd
    rows = ["| Run | Encoder | Training objective | Data / curriculum | Params | Steps | Minutes | Val. accuracy | Test fine accuracy | Test median Chamfer |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(results_root.glob("*/train/summary.json")):
        run = p.parent.parent.name
        if run.startswith("smoke"):
            continue
        s = json.load(open(p))
        enc, obj, data = RUN_DESC.get(run.split("_")[0], ("?", "?", "?"))
        ps = p.parent.parent / "synth" / "per_sample.csv"
        acc = cd = "-"
        if ps.exists():
            d = pd.read_csv(ps, usecols=["system", "correct", "chamfer"])
            d = d[d.system == "method"]
            acc, cd = f"{100 * d.correct.mean():.1f}%", f"{d.chamfer.median():.3f}"
        rows.append(f"| {run} | {enc} | {obj} | {data} | {s['n_params']:,} | {s['steps_done']} | {s['minutes']:.0f} | "
                    f"{100 * s['val_acc']:.1f}% | {acc} | {cd} |")
    return "\n".join(rows)


def fmt_p(p: float) -> str:
    """p-value for display; below 1e-16 the normal approximation (H1: from the paired-bootstrap SE) and float precision
    (H2, H5b: Wilcoxon signed-rank) carry no information, so the value is shown as a floor."""
    return "< 1e-16" if p < 1e-16 else f"{p:.2g}"


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _n_real(h4: dict) -> int:
    """Number of real (R*) datasets whose matched-Gaussian blob rates are averaged in H4 (exploratory ones included)."""
    return sum(k.startswith("R") for k in h4["C2_per_dataset"])


def _helix_rule_ids(real: dict, accepted: bool) -> list[str]:
    return [k for k, v in real.get("helix_rule", {}).items() if v == accepted]


def real_table(results: list[dict]) -> str:
    rows = ["| Dataset | Model / layer | N x D | Top-3 var. | Expected (coarse / fine) | Fit (full cloud) | p | Subsample majority (freq.) | Held-out resid. method / B1 / helix / ringing | Matched-Gaussian -> blob | Flags |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["id"].startswith("C1_") and not r["id"].endswith("_0"):
            continue
        m = r["meta"]
        st = r.get("stability", {})
        maj = st.get("fine_majority", "-")
        freq = st.get("fine_freq", {}).get(maj, float("nan"))
        ho = r.get("heldout", {})
        hos = " / ".join(f"{ho[k]:.3f}" for k in ("method", "B1_classical", "helix", "ringing")) if ho else "-"
        flags = ", ".join(f for f, on in (("gap<1.5", r["gap_flag"]), ("OOD", r["ood"])) if on) or "-"
        exp = f"{m.get('expected_coarse') or '-'} / {', '.join(m.get('expected_fine') or ['-'])}"
        c2 = r.get("C2_matched_gaussian", {}).get("blob")
        rows.append(f"| `{r['id']}`{' (prereg)' if is_prereg(r) else ''} | {m.get('model', m.get('source', ''))[:40]} / {m.get('layer', '-')} | "
                    f"{r['N']} x {r['D']} | {100 * r['top3_explained']:.0f}% | {exp} | **{r['family']}** | {r['probs'][r['family']]:.2f} | "
                    f"{maj} ({100 * freq:.0f}%) | {hos} | {_pct(c2)} | {flags} |")
    return "\n".join(rows)


_DIGITS = dict(zip("0123456789", ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")))


def _tag(dataset_id: str) -> str:
    """Letters-only LaTeX macro tag from a dataset id: R1_gpt2_wpos -> Ronegpttwowpos."""
    return re.sub(r"[^A-Za-z]", "", "".join(_DIGITS.get(c, c) for c in dataset_id))


def real_macros(results: list[dict]) -> str:
    """Per-dataset macros (full-cloud fit, its probability, subsample majority and frequency, top-3 variance) so the
    paper text never types a real-data result by hand: \\fitRonegpttwowpos, \\pRonegpttwowpos, ..."""
    lines = []
    for r in results:
        t, st = _tag(r["id"]), r.get("stability", {})
        maj = st.get("fine_majority", "-")
        vals = {"fit": r["family"], "p": f"{r['probs'][r['family']]:.2f}", "maj": maj,
                "majfreq": f"{100 * st.get('fine_freq', {}).get(maj, float('nan')):.0f}\\%", "var": f"{100 * r['top3_explained']:.0f}\\%"}
        lines += [f"\\newcommand{{\\{k}{t}}}{{{v}}}" for k, v in vals.items()]
    return "\n".join(lines) + "\n"


def headline(a: dict) -> str:
    """Markdown bullet list of the headline numbers (README, summary), all read from analysis.json."""
    s, h, out = a["systems"], a["hypotheses"], []
    acc = lambda k: f"{100 * s[k]['fine_acc'][0]:.1f}% [{100 * s[k]['fine_acc'][1]:.1f}, {100 * s[k]['fine_acc'][2]:.1f}]"
    if "method" in s:
        out.append(f"- **Synthetic test set ({s['method']['n']:,} clouds, 13 families):** the method picks the correct family in {acc('method')} "
                   f"of clouds; median Chamfer distance to the ground-truth shape {s['method']['median_chamfer'][0]:.3f} (RMS-radius units).")
    if all(k in s for k in ("method_subset", "B1_classical", "B2_features")):
        out.append(f"- **Same {s['B1_classical']['n']:,} clouds for every system:** method {acc('method_subset')}, B2 features + boosting "
                   f"{acc('B2_features')}, B1 classical fit + geometric AIC {acc('B1_classical')}, trivial baselines "
                   f"{100 * min(s[k]['fine_acc'][0] for k in ('B3a_null', 'B3b_circle', 'B3c_random') if k in s):.1f}-"
                   f"{100 * max(s[k]['fine_acc'][0] for k in ('B3a_null', 'B3b_circle', 'B3c_random') if k in s):.1f}%.")
    for k, txt in (("H1", "macro-F1 method - B1"), ("H2", "median Chamfer method - B1"), ("H5b_refinement", "median Chamfer method - no refinement")):
        if k in h:
            p = fmt_p(h[k].get("p_holm", float("nan")))
            out.append(f"- **{k}** ({txt}): {h[k]['estimate']:.4f} [{h[k]['ci'][0]:.4f}, {h[k]['ci'][1]:.4f}], "
                       f"Holm p {p if p.startswith('<') else '= ' + p}: {'supported' if h[k]['supported'] else 'not supported'}.")
    if "A5_gaic" in s:
        out.append(f"- **Selection rule:** choosing the family by geometric AIC instead of the classifier drops accuracy to {acc('A5_gaic')}.")
    c = a.get("calibration")
    if c:
        out.append(f"- **Calibration:** temperature T = {c['temperature']:.3f} (fitted on the validation stream) lowers ECE from {c['before']['ece']:.3f} "
                   f"to {c['after']['ece']:.3f} and NLL from {c['before']['nll']:.3f} to {c['after']['nll']:.3f}"
                   + (f"; the NLL-optimal T on the test set itself is {c['t_test_optimal']:.3f}." if "t_test_optimal" in c else "."))
    r = a.get("real")
    if r:
        h3, h4 = r["H3"], r["H4"]
        full = (f" Post hoc, with the full-cloud label instead of the subsample majority: {h3['agree_full_cloud_posthoc']}/{h3['n']}."
                if "agree_full_cloud_posthoc" in h3 else "")
        out.append(f"- **H3, real activations:** the coarse shape class (point set / curve / surface / blob; majority label over random 80% "
                   f"subsamples) agrees with the literature on {h3['agree']}/{h3['n']} preregistered datasets (exact 95% CI "
                   f"{100 * h3['ci'][0]:.0f}-{100 * h3['ci'][1]:.0f}%), label-permutation p = {h3['p_permutation']:.3g}: "
                   f"{'supported' if h3['supported'] else 'not supported'}.{full}")
        bp = h4.get("C1_by_process_posthoc", {})
        gp = f" Post hoc: smooth Gaussian processes alone are labelled helix in {_pct(bp['gp']['helix'])}, random walks in {_pct(bp['walk']['helix'])}." if "gp" in bp and "walk" in bp else ""
        out.append(f"- **H4, controls:** matched-covariance Gaussians are labelled blob {_pct(h4['C2_blob_rate_mean'])} of the time (mean over all "
                   f"{_n_real(h4)} real datasets, exploratory ones included); ordinal-process controls ringing {_pct(h4['C1_ringing_rate'])} vs "
                   f"helix {_pct(h4['C1_helix_rate'])}: {'supported' if h4['supported'] else 'not supported'}.{gp}")
        acc_ids, rej_ids = _helix_rule_ids(r, True), _helix_rule_ids(r, False)
        wrong = [i for i in acc_ids if i in r.get("helix_rule_non_helix_expected", [])]
        if acc_ids or rej_ids:
            out.append(f"- **Helix rule (post hoc, every full-cloud helix fit with held-out folds):** the preregistered two-part rule rejects the "
                       f"helix claim for {', '.join(f'`{i}`' for i in rej_ids) or 'none'} and accepts it for "
                       f"{', '.join(f'`{i}`' for i in acc_ids) or 'none'}"
                       + ((", all of which" if wrong == acc_ids else f"; of these, {', '.join(f'`{i}`' for i in wrong)}")
                          + " are expected to be non-helix shapes, so the rule is necessary, not sufficient." if wrong else "."))
    return "\n".join(out)


def _tex_escape(t: str) -> str:
    t = t.replace("\\", "/").replace("**", "").replace("`", "")
    for a, b in (("{", r"\{"), ("}", r"\}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("^", r"\textasciicircum{}"),
                 ("~", r"\textasciitilde{}"), ("->", r"$\to$"), ("<", r"$<$"), (">", r"$>$")):
        t = t.replace(a, b)
    return t


def md_to_tex(md: str, size: str = r"\scriptsize") -> str:
    """Convert a generated markdown pipe table into a LaTeX booktabs tabular."""
    lines = [l for l in md.strip().splitlines() if l.startswith("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines if not set(l) <= set("|-: ")]
    ncol = len(rows[0])
    out = [size, r"\begin{tabular}{" + "l" * ncol + "}", r"\toprule",
           " & ".join(r"\textbf{" + _tex_escape(c) + "}" for c in rows[0]) + r" \\", r"\midrule"]
    out += [" & ".join(_tex_escape(c) for c in r) + r" \\" for r in rows[1:]]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def select_cols(md: str, cols: list[str]) -> str:
    """Keep only the named columns of a generated markdown pipe table (compact tables for the paper)."""
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in md.strip().splitlines() if l.startswith("|")]
    idx = [rows[0].index(c) for c in cols]
    return "\n".join("| " + " | ".join(r[i] for i in idx) + " |" for r in rows)


PAPER_COMPACT = {   # paper tables: the same generated tables, fewer columns so they stay legible
    "results_table": ["System", "n", "Fine accuracy [95% CI]", "Macro-F1 [95% CI]", "Coarse accuracy", "Median Chamfer to GT [95% CI]"],
    "real_table": ["Dataset", "N x D", "Top-3 var.", "Expected (coarse / fine)", "Fit (full cloud)", "p", "Subsample majority (freq.)", "Flags"],
}


def replace_block(path: Path, name: str, content: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
    if pat.search(text):
        text = pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", text)
        path.write_text(text, encoding="utf-8")


def tex_macros(analysis: dict) -> str:
    """\\newcommand macros for every headline number (paper/generated/numbers.tex)."""
    lines = []

    def mac(name, val):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    s = analysis["systems"]
    for key, tag in (("method", "Method"), ("method_subset", "Methodsub"), ("B1_classical", "Bone"), ("B2_features", "Btwo"), ("A3_norefine", "Athree"),
                     ("A5_gaic", "Afivegaic"), ("B3a_null", "Bthreea"), ("B3b_circle", "Bthreeb"), ("B3c_random", "Bthreec"),
                     ("A1_paramloss", "Aone"), ("A2_noresidual", "Atwo"), ("A4_deepsets", "Afour"), ("A6_noembed", "Asix")):
        if key in s:
            v = s[key]
            mac(f"acc{tag}", f"{100 * v['fine_acc'][0]:.1f}\\%")
            mac(f"accCI{tag}", f"[{100 * v['fine_acc'][1]:.1f}, {100 * v['fine_acc'][2]:.1f}]")
            mac(f"fone{tag}", f"{v['macro_f1'][0]:.3f}")
            mac(f"coarse{tag}", f"{100 * v['coarse_acc'][0]:.1f}\\%")
            mac(f"cd{tag}", f"{v['median_chamfer'][0]:.3f}")
            mac(f"cdCI{tag}", f"[{v['median_chamfer'][1]:.3f}, {v['median_chamfer'][2]:.3f}]")
            mac(f"n{tag}", str(v["n"]))
    for k, v in analysis["hypotheses"].items():
        tag = re.sub(r"[^A-Za-z]", "", k.replace("1", "one").replace("2", "two").replace("5", "five"))
        mac(f"est{tag}", f"{v['estimate']:.3f}")
        mac(f"ci{tag}", f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
        mac(f"p{tag}", fmt_p(v.get('p_holm', v.get('p', float('nan')))).replace("< ", "$<$ "))
    if "real" in analysis:
        h3, h4 = analysis["real"]["H3"], analysis["real"]["H4"]
        mac("hthreeAgree", str(h3["agree"]))
        mac("hthreeN", str(h3["n"]))
        mac("hthreeP", f"{h3['p_permutation']:.3g}")
        mac("hfourBlob", _pct(h4["C2_blob_rate_mean"]).replace("%", "\\%"))
        mac("hfourRing", _pct(h4["C1_ringing_rate"]).replace("%", "\\%"))
        mac("hfourHelix", _pct(h4["C1_helix_rate"]).replace("%", "\\%"))
        mac("hthreeAgreeFull", str(h3.get("agree_full_cloud_posthoc", "n/a")))
        mac("hfourNreal", str(_n_real(h4)))
        tt = lambda ids: ", ".join(r"\texttt{" + i.replace("_", r"\_") + "}" for i in ids) or "none"
        mac("helixRuleAccepted", tt(_helix_rule_ids(analysis["real"], True)))
        mac("helixRuleRejected", tt(_helix_rule_ids(analysis["real"], False)))
        bp = h4.get("C1_by_process_posthoc", {})   # post-hoc split of C1 into Gaussian processes and random walks
        for k, tag in (("gp", "Gp"), ("walk", "Walk")):
            if k in bp:
                mac(f"hfour{tag}Helix", _pct(bp[k]["helix"]).replace("%", "\\%"))
                mac(f"hfour{tag}Ring", _pct(bp[k]["ringing"]).replace("%", "\\%"))
    cal = analysis.get("calibration")
    if cal:
        mac("temperature", f"{cal['temperature']:.3f}")
        if "t_test_optimal" in cal:
            mac("temperatureTest", f"{cal['t_test_optimal']:.3f}")
        mac("eceBefore", f"{cal['before']['ece']:.3f}")
        mac("eceAfter", f"{cal['after']['ece']:.3f}")
        mac("nllBefore", f"{cal['before']['nll']:.3f}")
        mac("nllAfter", f"{cal['after']['nll']:.3f}")
    if "ambiguous_fraction" in analysis:
        mac("ambiguousFrac", f"{100 * analysis['ambiguous_fraction']:.1f}\\%")
    return "\n".join(lines) + "\n"


def build_all(cfg: dict, ablation_runs: dict | None = None) -> dict:
    res = Path(cfg["paths"]["results"])
    ablation_runs = ablation_runs if ablation_runs is not None else cfg.get("report", {}).get("ablation_runs", {})
    cached = res / "analysis.json"
    # report.reuse_analysis: rebuild tables, figures and macros from the stored analysis without re-running the bootstraps
    if cfg.get("report", {}).get("reuse_analysis") and cached.exists():
        analysis = json.load(open(cached))
    else:
        analysis = run_analysis(cfg, ablation_runs)
    GEN_MD.mkdir(parents=True, exist_ok=True)
    GEN_TEX.mkdir(parents=True, exist_ok=True)
    order = [k for k in figures.SYSTEM_ORDER if k in analysis["systems"]]
    tables = {"results_table": systems_table(analysis["systems"], order),
              "hypotheses_table": hypotheses_table(analysis["hypotheses"], analysis.get("real")), "headline": headline(analysis)}
    rr = res / "real" / "real_results.json"
    if rr.exists():
        results = json.load(open(rr))
        tables["real_table"] = real_table(results)
    tables["models_table"] = models_table(res.parent)
    if "systems_unambiguous_subset" in analysis:
        tables["unambiguous_table"] = systems_table(analysis["systems_unambiguous_subset"], order)
    for name, t in tables.items():
        (GEN_MD / f"{name}.md").write_text(t + "\n", encoding="utf-8")
        if t.lstrip().startswith("|"):   # only pipe tables become LaTeX tables (the headline is a bullet list)
            (GEN_TEX / f"{name}.tex").write_text(md_to_tex(t) + "\n", encoding="utf-8")
        if name in PAPER_COMPACT:
            (GEN_TEX / f"{name}_compact.tex").write_text(md_to_tex(select_cols(t, PAPER_COMPACT[name])) + "\n", encoding="utf-8")
        for doc in [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "reports").glob("*.md"))]:
            replace_block(doc, name, t)
    (GEN_TEX / "numbers.tex").write_text(tex_macros(analysis) + (real_macros(results) if rr.exists() else ""), encoding="utf-8")
    # figures
    figures.fig_systems(analysis)
    nb = [f"{lo}-{hi}" for lo, hi in cfg["eval"]["n_bins"]]
    figures.fig_by_bin(analysis, "by_n_bin", nb, "Accuracy by number of points N", "fig_by_n.png")
    sb = [f"{lo}-{hi}" for lo, hi in cfg["eval"]["noise_bins"]]
    figures.fig_by_bin(analysis, "by_noise_bin", sb, "Accuracy by relative noise sigma", "fig_by_noise.png")
    figures.fig_confusion(analysis, NAMES)
    if (res / "train" / "curve.json").exists():
        figures.fig_training(res / "train" / "curve.json")
    if (res / "synth" / "probs.npz").exists():
        figures.fig_reliability(res / "synth" / "probs.npz", analysis.get("calibration"))
    if rr.exists():
        ids = [r["id"] for r in results if r["id"].startswith("R") and not r["id"].endswith("_means")]
        figures.fig_real(rr, res / "real" / "viz", ids, "fig_real.png")
        ctrl = [r["id"] for r in results if r["id"].startswith(("C1_gp_0", "C1_walk_0", "C3_"))]
        figures.fig_real(rr, res / "real" / "viz", ctrl, "fig_controls.png")
    log.info("report built: %s", list(tables))
    return analysis
