#!/bin/zsh
# Milestone 1 queue (2026-10-06; cancelled before any step ran):
#   wait for DP_gdn_pma_full -> eval suites A + DP -> D sweep+full -> DW sweep+full -> eval suites D + DW
# Each step is logged to results/m1_queue.log; a failed eval does not block training steps.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
Q=results/m1_queue.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $Q; }

WAIT_PID=${1:-}
if [[ -n $WAIT_PID ]]; then
  say "waiting for pid $WAIT_PID (DP_gdn_pma_full)"
  while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
fi
[[ -f results/DP_gdn_pma_full/metrics.json ]] || { say "DP_gdn_pma_full has no metrics.json; stopping queue"; exit 1; }

for r in A_transformer_full DP_gdn_pma_full; do
  say "eval suite $r start"
  python scripts/eval_model.py --run $r > results/eval_$r.log 2>&1 && say "eval suite $r done" || say "eval suite $r FAILED (see results/eval_$r.log)"
done

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
