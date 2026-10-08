#!/bin/zsh
# Sequential GPU queue for the Phase 2 bake-off (one job at a time => clean throughput numbers).
cd /Users/rishabh/startups/slm
source .venv/bin/activate
export PYTHONPATH=src
# wait for the baseline sweep/full run to finish
while pgrep -f "sweep.py --config configs/A_transformer.json" > /dev/null; do sleep 60; done
for c in DP_gdn_pma D_gdn DW_gdn_swa G_gla; do
  python scripts/sweep.py --config configs/$c.json --lrs 1e-3,2e-3,4e-3 --sweep_steps 1500 --full_steps 6000 > results/${c}_sweep.log 2>&1
done
for r in A_transformer_full DP_gdn_pma_full D_gdn_full DW_gdn_swa_full G_gla_full; do
  python scripts/eval_model.py --run $r > results/${r}_evalsuite.log 2>&1
done
echo "BAKEOFF QUEUE DONE $(date)" >> results/queue.log
