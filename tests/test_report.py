"""Report helpers: compact paper tables, p-value display and LaTeX macro names (every paper number is a macro)."""

import re

import numpy as np

from neurogeom.eval.analysis import hypotheses_real
from neurogeom.realdata.common import is_prereg
from neurogeom.report.numbers import _tag, fmt_p, md_to_tex, select_cols

TABLE = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| x | y | z |"


def test_select_cols_keeps_named_columns_in_order():
    out = select_cols(TABLE, ["C", "A"])
    assert out.splitlines()[0] == "| C | A |"
    assert out.splitlines()[-1] == "| z | x |"
    assert "\\begin{tabular}{ll}" in md_to_tex(out)   # separator row is dropped by md_to_tex


def test_fmt_p_floors_underflowed_values():
    assert fmt_p(0.0) == "< 1e-16"
    assert fmt_p(5.6e-124) == "< 1e-16"
    assert fmt_p(0.125) == "0.12"


def test_macro_tags_are_letters_only_and_distinct():
    ids = ["R1_gpt2_wpos", "R3_tms_n6", "R3_tms_n8", "R3_tms_n20", "R5a_llama_weekdays", "R5a_llama_weekdays_means", "C1_gp_0", "C1_gp_1"]
    tags = [_tag(i) for i in ids]
    assert all(re.fullmatch(r"[A-Za-z]+", t) for t in tags)
    assert len(set(tags)) == len(tags)


def _res(did, fit, coarse, majority, expected, prereg=True, helix_ok=None):
    """Minimal real_results.json entry for the H3/H4 bookkeeping."""
    return {"id": did, "family": fit, "coarse": coarse, "meta": {"prereg": prereg, "expected_coarse": expected, "expected_fine": ["x"]},
            "stability": {"coarse_majority": majority, "fine_majority": fit}, "helix_claim_supported": helix_ok}


def test_class_mean_views_are_not_preregistered_even_if_flagged():
    # deviations.md lists *_means as exploratory although their meta inherited prereg=True
    assert is_prereg(_res("R5a_llama_weekdays", "helix", "1d", "1d", "1d"))
    assert not is_prereg(_res("R5a_llama_weekdays_means", "helix", "1d", "1d", "1d"))
    assert not is_prereg(_res("R3_tms_n20", "sphere", "2d", "2d", "0d", prereg=False))


def test_h3_uses_subsample_majority_and_reports_full_cloud_agreement_post_hoc():
    res = [_res("R1_a", "helix", "1d", "1d", "1d"), _res("R3_b", "polytope", "0d", "2d", "0d"),
           _res("R5a_c_means", "arc", "1d", "1d", "1d"), _res("R9_d", "ringing", "1d", "1d", "0d", helix_ok=None),
           _res("R2_e", "helix", "1d", "1d", "1d", helix_ok=True), _res("C1_gp_0", "helix", "1d", "1d", "1d", helix_ok=False)]
    h3 = hypotheses_real(res)["H3"]
    assert (h3["agree"], h3["n"]) == (2, 4)                               # R1, R2 agree by subsample majority; _means excluded
    assert h3["agree_full_cloud_posthoc"] == 3                            # R3_b agrees only by its full-cloud label
    assert hypotheses_real(res)["helix_rule"] == {"R2_e": True, "C1_gp_0": False}


def test_calibration_reports_test_optimal_temperature(tmp_path):
    import json
    from neurogeom.eval.analysis import calibration
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4000, 5)) * 3
    y = np.array([rng.choice(5, p=np.exp(l) / np.exp(l).sum()) for l in logits])   # labels drawn from softmax(logits): T* = 1
    t = 0.5
    q = np.exp(logits / t)
    (tmp_path / "synth").mkdir()
    np.savez(tmp_path / "synth" / "probs.npz", probs=q / q.sum(1, keepdims=True), labels=y)
    json.dump({"temperature": t}, open(tmp_path / "synth" / "meta.json", "w"))
    assert abs(calibration(tmp_path)["t_test_optimal"] - 1.0) < 0.1
