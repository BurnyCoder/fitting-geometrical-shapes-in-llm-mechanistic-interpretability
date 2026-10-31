"""Single entry point: `uv run python run.py <phase> [<phase> ...] --config configs/<name>.yaml`.

Global context: each phase is one clearly named step of the research pipeline, implemented in a
module of src/neurogeom; this file only wires them together, sets up timestamped logging per phase,
and enforces the preregistration guard (no training before reports/preregistration.md is committed).
`all` runs the phases in order: check-env -> catalog -> train -> eval-synth -> extract-real ->
eval-real -> report. See README.md ("What happens when you run it").
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from neurogeom.config import REPO_ROOT, load_config  # noqa: E402
from neurogeom.logging_utils import get_logger, setup_logging  # noqa: E402

log = get_logger("run")


def prereg_committed() -> bool:
    """True if reports/preregistration.md is tracked by git with at least one commit."""
    try:
        out = subprocess.run(["git", "log", "--oneline", "--", "reports/preregistration.md"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return False


def phase_check_env(cfg):
    import torch
    log.info("python %s | torch %s | cuda %s | available %s", sys.version.split()[0], torch.__version__, torch.version.cuda,
             torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("gpu %s | arch list %s", torch.cuda.get_device_name(0), torch.cuda.get_arch_list())


def phase_catalog(cfg):
    from neurogeom.report.render import render_catalog
    render_catalog()


def phase_train(cfg):
    if not prereg_committed():
        raise SystemExit("refusing to train: commit reports/preregistration.md first")
    from neurogeom.train.loop import train
    train(cfg)


def phase_eval_synth(cfg):
    from neurogeom.eval.synth_eval import run_synth_eval
    run_synth_eval(cfg)


def phase_extract_real(cfg):
    from neurogeom.realdata.extract import extract_all
    extract_all(cfg)


def phase_eval_real(cfg):
    from neurogeom.eval.real_eval import run_real_eval
    run_real_eval(cfg)


def phase_report(cfg):
    from neurogeom.report.numbers import build_all
    build_all(cfg)


PHASES = {
    "check-env": phase_check_env,
    "catalog": phase_catalog,
    "train": phase_train,
    "eval-synth": phase_eval_synth,
    "extract-real": phase_extract_real,
    "eval-real": phase_eval_real,
    "report": phase_report,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases", nargs="+", choices=list(PHASES) + ["all"])
    ap.add_argument("--config", default=None, help="YAML file in configs/ (merged over configs/base.yaml)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.run_name:
        overrides["run_name"] = args.run_name
    cfg = load_config(args.config, overrides)
    phases = list(PHASES) if "all" in args.phases else args.phases
    for ph in phases:
        log_path = setup_logging(cfg["paths"]["logs"], f"{cfg['run_name']}_{ph}")
        log.info("=== phase %s | config %s | run %s | log %s", ph, cfg["config_name"], cfg["run_name"], log_path)
        PHASES[ph](cfg)
        log.info("=== phase %s done", ph)


if __name__ == "__main__":
    main()
