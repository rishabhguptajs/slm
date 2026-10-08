"""Run the full eval suite on a trained run: needle-in-a-stream, tool calls, efficiency.
(bpb is already in results/<run>/metrics.json from training.)
Writes results/<run>/eval_suite.json.

python scripts/eval_model.py --run A_transformer_full [--quick]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "src")
import mlx.core as mx  # noqa: E402

from slm.data.tokenizer import load_tokenizer  # noqa: E402
from slm.eval.efficiency import measure  # noqa: E402
from slm.eval.needle import run_needle  # noqa: E402
from slm.eval.toolcall import run_toolcall  # noqa: E402
from slm.models.model import ModelConfig, TokenLM  # noqa: E402


def load_run(run, root="results"):
    cfg = json.load(open(os.path.join(root, run, "config.json")))
    mc = ModelConfig(**cfg["model"])
    model = TokenLM(mc)
    model.load_weights(os.path.join(root, run, "model.safetensors"))
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", default="")
    ap.add_argument("--needle_lengths", default="", help="comma-separated filler lengths (default: built-in)")
    ap.add_argument("--tag", default="", help="write eval_suite_<tag>.json instead of eval_suite.json")
    a = ap.parse_args()
    model, cfg = load_run(a.run)
    tok = load_tokenizer(cfg["train"]["tokenizer"])
    out_path = os.path.join("results", a.run, f"eval_suite_{a.tag}.json" if a.tag else "eval_suite.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    skip = set(a.skip.split(","))
    print(f"== eval {a.run}", flush=True)
    if "needle" not in skip:
        lengths = (0, 128, 512) if a.quick else (0, 128, 512, 2048, 8192)
        if a.needle_lengths:
            lengths = tuple(int(x) for x in a.needle_lengths.split(","))
        res["needle"] = run_needle(model, tok, lengths=lengths, trials=4 if a.quick else 16)
    if "tool" not in skip:
        res["toolcall"] = run_toolcall(model, tok, n=20 if a.quick else 300)
    if "eff" not in skip:
        lengths = (512, 2048) if a.quick else (512, 2048, 8192, 32768)
        res["efficiency"] = measure(model, lengths=lengths)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
