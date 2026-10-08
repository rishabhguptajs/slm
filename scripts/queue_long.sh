#!/bin/zsh
# Long-context stage (2026-10-08):
# continue-train each full model at T=2048 (B=4, same 8,192 tokens/step) on the usual mixture,
# with 25% of rows replaced by synthetic long-range recall rows (src/slm/data/recall.py).
# 1,500 steps = 12.3M tokens per model; peak LR = 0.5 x that model's best LR; warmup 100; cosine to 10%.
# Then needle tests (fillers up to 8,192) on the new models AND the original full models, plus tool calls.
# Progress: results/long_queue.log. A failure in one step does not stop the others.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
Q=results/long_queue.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $Q; }

train_long() {  # $1 config name, $2 source run, $3 peak lr
  say "train $1_long2k start (from $2, lr $3)"
  python - > results/$1_long2k.log 2>&1 <<EOF
from slm.train import load_config, run
mc, tc = load_config('configs/$1.json')
tc.update(batch_size=4, seq_len=2048, steps=1500, warmup=100, lr=$3, eval_every=10**9,
          init_from='results/$2/model.safetensors', recall_frac=0.25)
run(mc, tc, '$1_long2k')
EOF
  if [[ -f results/$1_long2k/metrics.json ]]; then say "train $1_long2k done: $(grep FINAL results/$1_long2k.log)"; else say "train $1_long2k FAILED (see results/$1_long2k.log)"; fi
}

train_long DP_gdn_pma DP_gdn_pma_full 2e-3
train_long D_gdn D_gdn_full 1e-3
train_long A_transformer A_transformer_full 1e-3

NL=0,512,1024,1800,4096,8192
for r in DP_gdn_pma_long2k D_gdn_long2k A_transformer_long2k; do
  [[ -f results/$r/model.safetensors ]] || { say "skip eval $r (no model)"; continue; }
  say "eval $r start"
  python scripts/eval_model.py --run $r --skip eff --needle_lengths $NL --tag long > results/eval_$r.log 2>&1 \
    && say "eval $r done" || say "eval $r FAILED (see results/eval_$r.log)"
done
for r in DP_gdn_pma_full D_gdn_full A_transformer_full; do
  say "needle-only eval $r start"
  python scripts/eval_model.py --run $r --skip tool,eff --needle_lengths $NL --tag long > results/eval_${r}_long.log 2>&1 \
    && say "needle-only eval $r done" || say "needle-only eval $r FAILED (see results/eval_${r}_long.log)"
done
say "QUEUE DONE"
