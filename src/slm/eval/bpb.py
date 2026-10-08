"""Bits-per-byte on held-out val streams, per source. Tokenizer-independent:
bits = sum of token NLLs / ln 2; bytes = sum of UTF-8 lengths of predicted tokens."""
import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from slm.data.shards import load_val


def _nll_windows(model, arr, T, B):
    """Non-overlapping windows of T+1 tokens; returns per-target NLL (nats) and targets."""
    n_win = (len(arr) - 1) // T
    nll, tgt = [], []
    for s in range(0, n_win, B):
        idx = range(s, min(s + B, n_win))
        batch = np.stack([arr[i * T:i * T + T + 1] for i in idx])
        x, y = mx.array(batch[:, :-1]), mx.array(batch[:, 1:])
        logits, _ = model(x)
        l = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
        mx.eval(l)
        nll.append(np.array(l).reshape(-1))
        tgt.append(batch[:, 1:].reshape(-1))
    return np.concatenate(nll), np.concatenate(tgt)


def evaluate_bpb(model, tok_name, srcs, T=512, max_tokens=65536, B=16, root="data/tok"):
    tb = np.load(os.path.join(root, tok_name, "token_bytes.npy"))
    out = {}
    tot_bits = tot_bytes = 0.0
    for src in srcs:
        arr = load_val(tok_name, src, max_tokens + 1, root)
        nll, tgt = _nll_windows(model, arr, T, B)
        bits = nll.sum() / math.log(2)
        nbytes = tb[tgt].sum()
        out[src] = dict(bpb=float(bits / nbytes), loss=float(nll.mean()), tokens=int(len(tgt)),
                        bytes=int(nbytes))
        if src != "tool":
            tot_bits += bits
            tot_bytes += nbytes
    out["all_text"] = dict(bpb=float(tot_bits / tot_bytes))
    # macro average over natural languages (each language weighted equally)
    langs = [s for s in srcs if s != "tool"]
    out["macro_lang"] = dict(bpb=float(np.mean([out[s]["bpb"] for s in langs])))
    return out
