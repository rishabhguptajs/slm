"""LR sweep on a short budget, then (optionally) a full run at the best LR.
python scripts/sweep.py --config configs/A_transformer.json --lrs 1e-3,2e-3,4e-3 --sweep_steps 1500 --full_steps 6000
Sweep runs:  results/<name>_sweep_lr<lr>/     Full run: results/<name>_full/
"""
import argparse, json, os, sys
sys.path.insert(0, "src")
from slm.train import load_config, run

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--lrs", default="1e-3,2e-3,4e-3")
ap.add_argument("--sweep_steps", type=int, default=1500)
ap.add_argument("--full_steps", type=int, default=6000)
ap.add_argument("--name")
ap.add_argument("--no_full", action="store_true")
a = ap.parse_args()
name = a.name or os.path.splitext(os.path.basename(a.config))[0]
res = {}


def sweep_point(lr):
    mc, tc = load_config(a.config)
    tc.update(lr=lr, steps=a.sweep_steps, warmup=max(50, a.sweep_steps // 10), eval_every=10**9,
              final_eval_tokens=65536, save=False)
    m = run(mc, tc, f"{name}_sweep_lr{lr:g}")
    res[lr] = m["eval"]["all_text"]["bpb"] if not m["diverged"] else float("inf")


for lr in [float(x) for x in a.lrs.split(",")]:
    sweep_point(lr)
# if the optimum sits on the edge of the grid, extend one step (x2 or /2) outward, up to twice
for _ in range(2):
    best = min(res, key=res.get)
    if best == max(res):
        sweep_point(best * 2)
    elif best == min(res):
        sweep_point(best / 2)
    else:
        break
best = min(res, key=res.get)
summary = dict(config=a.config, sweep_steps=a.sweep_steps, bpb_by_lr={str(k): v for k, v in res.items()}, best_lr=best)
os.makedirs(f"results/{name}_sweep_summary", exist_ok=True)
json.dump(summary, open(f"results/{name}_sweep_summary/metrics.json", "w"), indent=2)
print("SWEEP", json.dumps(summary), flush=True)
if not a.no_full:
    mc, tc = load_config(a.config)
    tc.update(lr=best, steps=a.full_steps, warmup=max(100, a.full_steps // 20), eval_every=a.full_steps // 3)
    run(mc, tc, f"{name}_full")
