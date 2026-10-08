"""MQAR grid: every architecture (4-layer, d=128 versions with the same layer ratios) x
(seq len, #pairs), best of 2 LRs. results/mqar/<pattern>_T<T>_D<D>_lr<lr>/metrics.json"""
import json, os, sys
sys.path.insert(0, "src")
from slm.eval.mqar import train_mqar

PATTERNS = {"A": "AAAA", "G": "GGGG", "D": "DDDD", "DW": "DDDW", "DP": "DDDP"}
SETTINGS = [(256, 16), (512, 32), (1024, 64)]
LRS = [1e-3, 3e-3]
summary = {}
for T, D in SETTINGS:
    for name, pat in PATTERNS.items():
        best = None
        for lr in LRS:
            out = f"results/mqar/{name}_T{T}_D{D}_lr{lr:g}"
            if os.path.exists(f"{out}/metrics.json"):
                r = json.load(open(f"{out}/metrics.json"))
            else:
                r = train_mqar(pat, T, D, lr, out=out)
            print(f"{name:3s} T={T:5d} D={D:3d} lr={lr:g}: acc {r['acc']:.3f} ({r['seconds']:.0f}s)", flush=True)
            if best is None or r["acc"] > best["acc"]:
                best = r
        summary[f"{name}_T{T}_D{D}"] = best
        json.dump(summary, open("results/mqar/summary.json", "w"), indent=2)
