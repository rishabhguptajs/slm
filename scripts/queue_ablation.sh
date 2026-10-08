#!/bin/zsh
# Ablation queue (2026-10-06):
#   D_gdn sweep+full -> DW_gdn_swa sweep+full -> eval suites on both.
# Progress is logged to results/ablation_queue.log.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
Q=results/ablation_queue.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $Q; }

for c in D_gdn DW_gdn_swa; do
  say "sweep+full $c start"
  python scripts/sweep.py --config configs/$c.json > results/${c}_sweep.log 2>&1 && say "sweep+full $c done" || say "sweep+full $c FAILED (see results/${c}_sweep.log)"
done
for r in D_gdn_full DW_gdn_swa_full; do
  [[ -f results/$r/model.safetensors ]] || { say "skip eval $r (no model)"; continue; }
  say "eval suite $r start"
  python scripts/eval_model.py --run $r > results/eval_$r.log 2>&1 && say "eval suite $r done" || say "eval suite $r FAILED (see results/eval_$r.log)"
done
say "QUEUE DONE"
