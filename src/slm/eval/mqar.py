"""Multi-query associative recall (MQAR, Zoology-style) trained from scratch.

Sequence of length T: D key-value pairs (k1 v1 ... kD vD), then the remaining positions are
pad (0) except at query positions where a previously seen key appears; the target after it is
its value. Keys are drawn from [1, V/2), values from [V/2, V). Loss and accuracy are only on
query targets. Queries are placed uniformly in the remainder, so gaps between a pair and its
query range up to ~T tokens (this probes memory beyond a sliding window).
"""
import json
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from slm.models.model import ModelConfig, TokenLM, count_params


def make_batch(rng, B, T, D, V=8192):
    x = np.zeros((B, T), dtype=np.int32)
    y = np.full((B, T), -100, dtype=np.int32)
    n_free = T - 2 * D
    for b in range(B):
        keys = rng.choice(np.arange(1, V // 2), D, replace=False)
        vals = rng.integers(V // 2, V, D)
        x[b, 0:2 * D:2] = keys
        x[b, 1:2 * D:2] = vals
        # query slots: pairs of positions (key at p, answer target at p) in the free region
        n_q = min(D, n_free // 2)
        slots = np.sort(rng.choice(n_free // 2, n_q, replace=False)) * 2 + 2 * D
        order = rng.permutation(D)[:n_q]
        x[b, slots] = keys[order]
        y[b, slots] = vals[order]  # predict value from the query-key position
    return x, y


def train_mqar(pattern, T, D, lr, steps=2000, B=32, d=128, L=4, seed=0, out=None, n_slots=64, window=128):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    cfg = ModelConfig(vocab_size=8192, d_model=d, n_layers=L, pattern=pattern, n_heads=2,
                      mlp_hidden=2 * d, window=window, n_slots=n_slots, chunk=64)
    model = TokenLM(cfg)
    sched = optim.join_schedules([optim.linear_schedule(lr * 0.01, lr, steps // 10),
                                  optim.cosine_decay(lr, steps - steps // 10, lr * 0.01)], [steps // 10])
    opt = optim.AdamW(sched, weight_decay=0.1)

    def loss_fn(m, x, y):
        logits, _ = m(x)
        mask = y >= 0
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), mx.maximum(y, 0), reduction="none")
        return (ce * mask).sum() / mask.sum()

    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, g = lg(model, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return loss

    t0 = time.time()
    for it in range(steps):
        x, y = make_batch(rng, B, T, D)
        loss = step(mx.array(x), mx.array(y))
        mx.eval(state, loss)
    # eval on fresh sequences
    erng = np.random.default_rng(10_000 + seed)
    correct = total = 0
    for _ in range(8):
        x, y = make_batch(erng, 64, T, D)
        logits, _ = model(mx.array(x))
        pred = np.array(mx.argmax(logits, axis=-1))
        m = y >= 0
        correct += int((pred[m] == y[m]).sum())
        total += int(m.sum())
    res = dict(pattern=pattern, T=T, D=D, lr=lr, steps=steps, acc=correct / total,
               final_loss=float(loss.item()), params=count_params(model), seconds=time.time() - t0)
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(res, f, indent=2)
    return res
