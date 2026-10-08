"""Tool-call accuracy on held-out function-calling samples.

For each val sample, take the first tool call: the prompt is everything up to and including
'<|assistant|><|tool_call|>', and the target is the canonical JSON call. Greedy decode until
'<|/tool_call|>'. Only prompts <= max_prompt tokens are used (inside every model's training
context, so the transformer is not penalised for length extrapolation).

Metrics: json_valid, name_acc, args_exact (parsed dict equality), exact (string equality), and
call_bpb (teacher-forced bits per byte of the call text, a continuous signal for tiny models).
"""
import json
import math

import numpy as np

from slm.data.tooldata import read_tool_jsonl
from slm.eval.generate import continuation_logprob, greedy

MARK = "<|assistant|><|tool_call|>"


def build_examples(tok, n=300, max_prompt=448, seed=0, path="data/raw/tool/val.jsonl.zst"):
    rows = read_tool_jsonl(path)
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    ex = []
    for r in rows:
        if not r["calls"]:
            continue
        i = r["text"].find(MARK)
        if i < 0:
            continue
        prompt = r["text"][: i + len(MARK)]
        call = r["calls"][0]
        p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt:
            continue
        ex.append(dict(prompt_ids=p_ids, call=call, target_ids=tok.encode(call) + [tok.special["<|/tool_call|>"]]))
        if len(ex) >= n:
            break
    return ex


def run_toolcall(model, tok, n=300, max_new=96):
    ex = build_examples(tok, n)
    stop = {tok.special["<|/tool_call|>"], tok.special["<|end|>"], tok.eot}
    m = dict(json_valid=0, name_acc=0, args_exact=0, exact=0)
    bits = nbytes = 0.0
    for e in ex:
        out = tok.decode(greedy(model, e["prompt_ids"], max_new=max_new, stop_ids=stop))
        gold = json.loads(e["call"])
        try:
            pred = json.loads(out)
            m["json_valid"] += 1
            m["name_acc"] += int(pred.get("name") == gold["name"])
            m["args_exact"] += int(pred.get("name") == gold["name"] and pred.get("arguments") == gold["arguments"])
        except (json.JSONDecodeError, AttributeError):
            pass
        m["exact"] += int(out.strip() == e["call"])
        lp = continuation_logprob(model, e["prompt_ids"], e["target_ids"])
        bits += -lp / math.log(2)
        nbytes += len(e["call"].encode("utf-8")) + 1
    res = {k: v / len(ex) for k, v in m.items()}
    res["call_bpb"] = bits / nbytes
    res["n"] = len(ex)
    print(f"  toolcall n={len(ex)}: " + " ".join(f"{k} {v:.3f}" for k, v in res.items() if k != "n"), flush=True)
    return res
