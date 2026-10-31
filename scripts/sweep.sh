#!/usr/bin/env bash
# Staged follow-up experiments after main_s0 (and any trained control or baseline models) is published. Each stage
# is run, then published (paper, results, reports, README, website, models on Hugging Face) before the next starts:
#   bash scripts/sweep.sh ablations     # A0 reference + A1/A2/A4/A6, shared short budget (12k steps / 60 min)
#   bash scripts/sweep.sh sensitivity   # seeds 1-2 of the main config; learning rate x3 and /3 at the ablation
#                                       # budget; inference hyperparameters re-evaluated on main_s0's weights
# Every run evaluates the method on the same fixed per-family test subset (the evaluation data seed is fixed),
# so all runs are paired with each other and with main_s0.
set -euo pipefail
cd "$(dirname "$0")/.."
EVAL_ARGS=(--set eval.baselines=false --set eval.variants=false --set eval.method_subset_per_cell=100)

case "${1:-}" in
  ablations)
    for a in A0_reference A1_paramloss A2_noresidual A4_deepsets A6_noembed; do
      uv run python run.py train eval-synth --config "ablation_${a}.yaml" "${EVAL_ARGS[@]}"
    done ;;
  sensitivity)
    for s in 1 2; do
      uv run python run.py train eval-synth --config main.yaml --seed "$s" --run-name "main_s$s" "${EVAL_ARGS[@]}"
    done
    # training hyperparameter: learning rate (default 3e-4), compared with A0_reference_s0 at the same budget
    for lr in 1e-4 1e-3; do
      uv run python run.py train eval-synth --config ablation_A0_reference.yaml --run-name "H_lr${lr}_s0" --set train.lr="$lr" "${EVAL_ARGS[@]}"
    done
    # inference hyperparameters: same main_s0 weights, one setting changed (defaults: 300 steps, back weight 0.25, top-3)
    for hp in infer.refine_steps=100 infer.refine_steps=600 infer.refine_back_weight=0.1 infer.refine_back_weight=0.5 infer.top_k=1 infer.top_k=5; do
      name="H_${hp#infer.}"
      uv run python run.py eval-synth --config main.yaml --run-name "${name//=/}_main_s0" \
        --set eval.checkpoint=checkpoints/main_s0/final.pt --set "$hp" "${EVAL_ARGS[@]}"
    done ;;
  *) echo "usage: bash scripts/sweep.sh ablations|sensitivity"; exit 1 ;;
esac
