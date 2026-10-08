"""Convert open function-calling datasets into one canonical format.

Canonical sample (a single string, special tokens from tokenizer.SPECIALS):

  <|functions|>[{schema}, ...]<|end|>
  <|user|>...<|end|>
  <|assistant|><|tool_call|>{"name": "f", "arguments": {...}}<|/tool_call|><|end|>
  <|tool_result|>{...}<|end|>
  <|assistant|>...<|end|>

Tool calls are serialized with json.dumps(sort_keys=False, ensure_ascii=False) so
that the eval can compare exact strings *and* parsed JSON.
Output: data/raw/tool/{train,val}.jsonl.zst with {"text", "src", "calls"} fields.
"""
import ast
import json
import os
import random
import re

import zstandard as zstd

from slm.data.download import ZstdJsonlWriter

SEP = "<|end|>"


def canon_call(name, args):
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def parse_glaive(ex):
    system, chat = ex["system"], ex["chat"]
    # functions: one or more JSON objects after the preamble
    m = system.find("{")
    funcs = []
    if m >= 0:
        dec = json.JSONDecoder()
        s = system[m:]
        i = 0
        while i < len(s):
            j = s.find("{", i)
            if j < 0:
                break
            try:
                obj, end = dec.raw_decode(s, j)
                funcs.append(obj)
                i = end
            except json.JSONDecodeError:
                return None
    parts = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat)
    out = [f"<|functions|>{json.dumps(funcs, ensure_ascii=False)}{SEP}"] if funcs else []
    calls = []
    role = None
    for p in parts:
        p_s = p.strip()
        if p_s in ("USER:", "ASSISTANT:", "FUNCTION RESPONSE:"):
            role = p_s
            continue
        if role is None or not p_s:
            continue
        text = p_s.replace("<|endoftext|>", "").strip()
        if role == "USER:":
            out.append(f"<|user|>{text}{SEP}")
        elif role == "FUNCTION RESPONSE:":
            out.append(f"<|tool_result|>{text}{SEP}")
        else:
            if text.startswith("<functioncall>"):
                body = text[len("<functioncall>"):].strip()
                mm = re.match(r'\{"name":\s*"([^"]+)",\s*"arguments":\s*\'(.*)\'\}\s*$', body, re.S)
                if not mm:
                    return None
                try:
                    args = json.loads(mm.group(2))
                except json.JSONDecodeError:
                    return None
                c = canon_call(mm.group(1), args)
                calls.append(c)
                out.append(f"<|assistant|><|tool_call|>{c}<|/tool_call|>{SEP}")
            else:
                out.append(f"<|assistant|>{text}{SEP}")
    if len(out) < 2:
        return None
    return "".join(out), calls


def parse_hermes(ex):
    conv = ex["conversations"]
    if isinstance(conv, str):
        conv = ast.literal_eval(conv)
    tools = ex.get("tools")
    try:
        tools = json.loads(tools) if isinstance(tools, str) else tools
        funcs = [t.get("function", t) for t in tools]
    except Exception:
        return None
    out = [f"<|functions|>{json.dumps(funcs, ensure_ascii=False)}{SEP}"]
    calls = []
    for turn in conv:
        role, val = turn["from"], turn["value"].strip()
        if role == "system":
            continue
        if role == "human":
            out.append(f"<|user|>{val}{SEP}")
        elif role == "tool":
            val = re.sub(r"</?tool_response>", "", val).strip()
            out.append(f"<|tool_result|>{val}{SEP}")
        elif role == "gpt":
            tcs = re.findall(r"<tool_call>(.*?)</tool_call>", val, re.S)
            if tcs:
                seg = []
                for tc in tcs:
                    try:
                        obj = json.loads(tc.strip())
                    except json.JSONDecodeError:
                        try:
                            obj = ast.literal_eval(tc.strip())
                        except Exception:
                            return None
                    c = canon_call(obj["name"], obj.get("arguments", {}))
                    calls.append(c)
                    seg.append(f"<|tool_call|>{c}<|/tool_call|>")
                out.append("<|assistant|>" + "".join(seg) + SEP)
            else:
                out.append(f"<|assistant|>{val}{SEP}")
    return "".join(out), calls


def main(out_dir="data/raw/tool", val_frac=0.03, seed=0):
    from datasets import load_dataset

    rng = random.Random(seed)
    tr = ZstdJsonlWriter(os.path.join(out_dir, "train.jsonl.zst"))
    va = ZstdJsonlWriter(os.path.join(out_dir, "val.jsonl.zst"))
    stats = {}
    seen = set()  # global dedup so no example can land in both train and val
    for src, name, cfg, fn in [
        ("glaive", "glaiveai/glaive-function-calling-v2", None, parse_glaive),
        ("hermes", "NousResearch/hermes-function-calling-v1", "func_calling_singleturn", parse_hermes),
    ]:
        ds = load_dataset(name, cfg, split="train")
        ok = bad = 0
        for ex in ds:
            try:
                r = fn(ex)
            except Exception:
                r = None
            if r is None:
                bad += 1
                continue
            text, calls = r
            h = hash(text)
            if h in seen:
                bad += 1
                continue
            seen.add(h)
            w = va if rng.random() < val_frac else tr
            line = json.dumps({"text": text, "src": src, "calls": calls}, ensure_ascii=False) + "\n"
            w.w.write(line.encode("utf-8"))
            w.bytes += len(text.encode("utf-8"))
            w.docs += 1
            ok += 1
        stats[src] = dict(ok=ok, bad=bad)
        print(src, stats[src], flush=True)
    tr.close()
    va.close()
    meta = dict(stats=stats, train_bytes=tr.bytes, train_docs=tr.docs, val_bytes=va.bytes, val_docs=va.docs)
    with open(os.path.join(out_dir, "DONE.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(meta)


def read_tool_jsonl(path):
    with open(path, "rb") as f:
        data = zstd.ZstdDecompressor().stream_reader(f).read()
    return [json.loads(l) for l in data.decode("utf-8").split("\n") if l]


if __name__ == "__main__":
    main()
