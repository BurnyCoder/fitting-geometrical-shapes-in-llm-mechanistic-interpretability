"""Report helpers: compact paper tables, p-value display and LaTeX macro names (every paper number is a macro)."""

import re

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
