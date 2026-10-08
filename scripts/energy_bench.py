"""Energy per token via macOS powermetrics (needs sudo; run through scripts/energy.sh).

Workload: for each run, an idle phase (baseline power), then a sustained decode phase
(batch 1, streaming state) and a prefill phase, each >= `seconds` long. Phase boundaries
are written to results/energy/phases.json; parse_energy() integrates (power - idle) over
each phase using the powermetrics sample log -> joules per token.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")


def workload(runs, seconds=30, out="results/energy/phases.json"):
    import mlx.core as mx
    import numpy as np
    from eval_model import load_run

    phases = []
    time.sleep(2)
    t0 = time.time(); time.sleep(seconds)
    phases.append(dict(run="idle", phase="idle", start=t0, end=time.time(), tokens=0))
    for run in runs:
        model, _ = load_run(run)
        rng = np.random.default_rng(0)
        V = model.cfg.vocab_size
        # decode phase: stream after a 512-token prefix
        st = model.init_state(1)
        lg, st = model(mx.array(rng.integers(0, V, (1, 512)).astype(np.int32)), st)
        mx.eval(lg, st)
        tok = mx.array([[1]], dtype=mx.int32)
        n, t0 = 0, time.time()
        while time.time() - t0 < seconds:
            lg, st = model(tok, st)
            mx.eval(lg, st)
            n += 1
        phases.append(dict(run=run, phase="decode", start=t0, end=time.time(), tokens=n))
        # prefill phase: 512-token chunks on a fresh stream each 8k tokens
        n, t0 = 0, time.time()
        st = model.init_state(1)
        while time.time() - t0 < seconds:
            if n % 8192 == 0:
                st = model.init_state(1)
            lg, st = model(mx.array(rng.integers(0, V, (1, 512)).astype(np.int32)), st)
            mx.eval(lg, st)
            n += 512
        phases.append(dict(run=run, phase="prefill", start=t0, end=time.time(), tokens=n))
        time.sleep(5)  # let power settle between models
        print(run, "done", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(phases, open(out, "w"), indent=2)


def parse_energy(pm_log="results/energy/powermetrics.txt", phases_path="results/energy/phases.json"):
    """powermetrics prints a header per sample with the wall time, then power lines."""
    hdr = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \((\d+(?:\.\d+)?)ms elapsed\) \*\*\*")
    samples, cur = [], None
    for line in open(pm_log, errors="ignore"):
        m = hdr.search(line)
        if m:
            ts = datetime.strptime(m.group(1).rsplit(" ", 1)[0], "%a %b %d %H:%M:%S %Y").timestamp()
            cur = dict(t=ts, dt=float(m.group(2)) / 1000)
            samples.append(cur)
        elif cur is not None and "Combined Power" in line:
            cur["mw"] = float(re.findall(r"(\d+(?:\.\d+)?) mW", line)[0])
    samples = [s for s in samples if "mw" in s]
    phases = json.load(open(phases_path))
    idle = [s["mw"] for s in samples if phases[0]["start"] + 1 <= s["t"] <= phases[0]["end"] - 1]
    p_idle = sum(idle) / max(1, len(idle))
    out = dict(idle_mw=p_idle, phases=[])
    for ph in phases[1:]:
        ss = [s for s in samples if ph["start"] + 1 <= s["t"] <= ph["end"] - 1]
        if not ss:
            continue
        p = sum(s["mw"] for s in ss) / len(ss)
        dur = ph["end"] - ph["start"]
        tps = ph["tokens"] / dur
        out["phases"].append(dict(run=ph["run"], phase=ph["phase"], avg_mw=p, tok_s=tps,
                                  mj_per_token_total=p / tps, mj_per_token_above_idle=(p - p_idle) / tps))
    json.dump(out, open("results/energy/energy.json", "w"), indent=2)
    for r in out["phases"]:
        print(f"{r['run']:22s} {r['phase']:8s} {r['avg_mw']:7.0f} mW  {r['tok_s']:9.1f} tok/s  "
              f"{r['mj_per_token_above_idle']:8.3f} mJ/token (above idle)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="A_transformer_full,D_gdn_full,DP_gdn_pma_full,DW_gdn_swa_full,G_gla_full")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--parse", action="store_true")
    a = ap.parse_args()
    if a.parse:
        parse_energy()
    else:
        workload(a.runs.split(","), a.seconds)
