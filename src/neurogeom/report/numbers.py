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
                        f"{v.get('p_holm', v.get('p', float('nan'))):.2g} | {v['n']} | {verdict} |")
    if real:
        h3, h4 = real["H3"], real["H4"]
        ci = h3["ci"] or [float("nan")] * 2
        rows.append(f"| H3 | Coarse-label agreement with literature on real datasets | {h3['agree']}/{h3['n']} = {100 * h3['rate']:.0f}% "
                    f"[{100 * ci[0]:.0f}, {100 * ci[1]:.0f}] (exact binomial) | perm. p = {h3['p_permutation']:.3g} | {h3['n']} | "
                    f"{'supported' if h3['supported'] else 'not supported'} |")
        rows.append(f"| H4 | Controls: matched Gaussians -> blob; ordinal processes -> ringing (not helix) | blob rate "
                    f"{_pct(h4['C2_blob_rate_mean'])}; ringing {_pct(h4['C1_ringing_rate'])} vs helix {_pct(h4['C1_helix_rate'])} | - | - | "
                    f"{'supported' if h4['supported'] else 'not supported'} |")
    return "\n".join(rows)


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


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
        rows.append(f"| `{r['id']}`{' (prereg)' if m.get('prereg') else ''} | {m.get('model', m.get('source', ''))[:40]} / {m.get('layer', '-')} | "
                    f"{r['N']} x {r['D']} | {100 * r['top3_explained']:.0f}% | {exp} | **{r['family']}** | {r['probs'][r['family']]:.2f} | "
                    f"{maj} ({100 * freq:.0f}%) | {hos} | {_pct(c2)} | {flags} |")
    return "\n".join(rows)


def _tex_escape(t: str) -> str:
    t = t.replace("\\", "/").replace("**", "").replace("`", "")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("->", r"$\to$"), ("<", r"$<$"), (">", r"$>$")):
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
    for key, tag in (("method", "Method"), ("B1_classical", "Bone"), ("B2_features", "Btwo"), ("A3_norefine", "Athree"),
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
        mac(f"p{tag}", f"{v.get('p_holm', v.get('p', float('nan'))):.2g}")
    if "real" in analysis:
        h3, h4 = analysis["real"]["H3"], analysis["real"]["H4"]
        mac("hthreeAgree", str(h3["agree"]))
        mac("hthreeN", str(h3["n"]))
        mac("hthreeP", f"{h3['p_permutation']:.3g}")
        mac("hfourBlob", _pct(h4["C2_blob_rate_mean"]).replace("%", "\\%"))
        mac("hfourRing", _pct(h4["C1_ringing_rate"]).replace("%", "\\%"))
        mac("hfourHelix", _pct(h4["C1_helix_rate"]).replace("%", "\\%"))
    if "ambiguous_fraction" in analysis:
        mac("ambiguousFrac", f"{100 * analysis['ambiguous_fraction']:.1f}\\%")
    return "\n".join(lines) + "\n"


def build_all(cfg: dict, ablation_runs: dict | None = None) -> dict:
    res = Path(cfg["paths"]["results"])
    ablation_runs = ablation_runs if ablation_runs is not None else cfg.get("report", {}).get("ablation_runs", {})
    analysis = run_analysis(cfg, ablation_runs)
    GEN_MD.mkdir(parents=True, exist_ok=True)
    GEN_TEX.mkdir(parents=True, exist_ok=True)
    order = [k for k in figures.SYSTEM_ORDER if k in analysis["systems"]]
    tables = {"results_table": systems_table(analysis["systems"], order),
              "hypotheses_table": hypotheses_table(analysis["hypotheses"], analysis.get("real"))}
    rr = res / "real" / "real_results.json"
    if rr.exists():
        results = json.load(open(rr))
        tables["real_table"] = real_table(results)
    if "systems_unambiguous_subset" in analysis:
        tables["unambiguous_table"] = systems_table(analysis["systems_unambiguous_subset"], order)
    for name, t in tables.items():
        (GEN_MD / f"{name}.md").write_text(t + "\n", encoding="utf-8")
        (GEN_TEX / f"{name}.tex").write_text(md_to_tex(t) + "\n", encoding="utf-8")
        for doc in [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "reports").glob("*.md"))]:
            replace_block(doc, name, t)
    (GEN_TEX / "numbers.tex").write_text(tex_macros(analysis), encoding="utf-8")
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
        figures.fig_reliability(res / "synth" / "probs.npz")
    if rr.exists():
        ids = [r["id"] for r in results if r["id"].startswith("R") and not r["id"].endswith("_means")]
        figures.fig_real(rr, res / "real" / "viz", ids, "fig_real.png")
        ctrl = [r["id"] for r in results if r["id"].startswith(("C1_gp_0", "C1_walk_0", "C3_"))]
        figures.fig_real(rr, res / "real" / "viz", ctrl, "fig_controls.png")
    log.info("report built: %s", list(tables))
    return analysis
