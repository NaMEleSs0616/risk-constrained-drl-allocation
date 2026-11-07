#!/usr/bin/env bash
# Robustness training runs (sequential; each step is resumable with --resume).
#  1. Independent synthetic markets under next-open execution: three standard data seeds
#     and one "bear" scenario (negative calm-regime drift, twice as frequent stress).
#  2. Warm-start vs from-scratch control on a subset of ETF folds.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for SEED in 101 202 303; do
  $PY scripts/run_real_experiments.py --config config/synthetic_nextopen.yaml --resume \
      --output-dir outputs_syn_nextopen/seed_${SEED} --set seed=${SEED} \
      > outputs_syn_nextopen/seed_${SEED}.log 2>&1
done
$PY scripts/run_real_experiments.py --config config/synthetic_nextopen.yaml --resume \
    --output-dir outputs_syn_nextopen/bear_404 --set seed=404 \
    "data.synthetic.annual_drift=[-0.08, 0.0]" data.synthetic.stress_entry_prob=0.016 \
    > outputs_syn_nextopen/bear_404.log 2>&1
$PY scripts/warmstart_control.py --config config/real_etf_nextopen.yaml \
    --variants S_smoothed --seeds 0 1 2 --folds 4 8 13 \
    > outputs_real_nextopen/warmstart_control.log 2>&1
echo ROBUSTNESS_DONE
