#!/usr/bin/env bash
# Seed-sensitivity and ablation sweep (run after the first model's results are published).
# Each run: train -> method-only evaluation on the fixed per-family test subset (same clouds for every run,
# because the evaluation data seed is fixed), so all runs are paired with each other and with main_s0.
# Usage: bash scripts/sweep.sh            (from the repo root; logs go to logs/)
set -euo pipefail
cd "$(dirname "$0")/.."
EVAL_ARGS=(--set eval.baselines=false --set eval.variants=false --set eval.method_subset_per_cell=100)

for s in 1 2; do
  uv run python run.py train eval-synth --config main.yaml --seed "$s" --run-name "main_s$s" "${EVAL_ARGS[@]}"
done
for a in A0_reference A1_paramloss A2_noresidual A4_deepsets A6_noembed; do
  uv run python run.py train eval-synth --config "ablation_${a}.yaml" "${EVAL_ARGS[@]}"
done
