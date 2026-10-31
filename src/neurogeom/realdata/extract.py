"""Phase extract-real: build every real point-cloud dataset and control (saved to data_cache/real/).

Global context: runs the dataset builders in order of GPU memory need. Each builder is independent and
idempotent (skipped when its output exists unless cfg["real"]["overwrite"]); a failing builder is
logged and recorded as failed (preregistration: failed extractions are reported, never dropped).
"""

from __future__ import annotations

import json
import traceback

from neurogeom.logging_utils import get_logger
from neurogeom.realdata import controls, llm_sets, weights_sets
from neurogeom.realdata.common import real_dir

log = get_logger(__name__)

BUILDERS = [
    ("controls_ordinal", ["C1_gp_0"], controls.ordinal_controls),
    ("gpt2_wpos", ["R1_gpt2_wpos"], weights_sets.gpt2_wpos),
    ("grokking", ["R2_grokking_WE"], weights_sets.grokking),
    ("tms", ["R3_tms_n6"], weights_sets.tms_sets),
    ("llama_l5", ["R4_llama_counting", "R5a_llama_weekdays"], llm_sets.llama_layer5),
    ("llama_l5_random", ["C3_llama_counting_init"], lambda cfg: llm_sets.llama_layer5(cfg, random_init=True)),
    ("llama_l19", ["R6_llama_years"], llm_sets.llama_layer19),
    ("llama_l12_truth", ["R9_llama_truth_cities"], llm_sets.llama_layer12_truth),
    ("pythia_counting", ["R4b_pythia_counting"], llm_sets.pythia_counting),
]


def extract_all(cfg: dict, only: list[str] | None = None) -> dict:
    status = {}
    d = real_dir(cfg)
    overwrite = cfg.get("real", {}).get("overwrite", False)
    for name, outputs, fn in BUILDERS:
        if only and name not in only:
            continue
        if not overwrite and all((d / f"{o}.npz").exists() for o in outputs):
            log.info("skip %s (exists)", name)
            status[name] = "exists"
            continue
        log.info(">>> building %s", name)
        try:
            fn(cfg)
            status[name] = "ok"
        except Exception as e:  # noqa: BLE001 - record and continue; reported in results
            log.error("builder %s failed: %s\n%s", name, e, traceback.format_exc())
            status[name] = f"failed: {e}"
    json.dump(status, open(d / "extract_status.json", "w"), indent=2)
    log.info("extraction status: %s", status)
    return status
