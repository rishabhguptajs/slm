"""Training loop (MLX). One run = results/<run>/{config.json, log.txt, curve.jsonl,
metrics.json, model.safetensors}.

Usage: python -m slm.train --config configs/xxx.json [--run name] [--lr 3e-3] [--steps N]
"""
import argparse
import copy
import json
import math
import os
import platform
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from slm.data.shards import MixtureLoader
from slm.eval.bpb import evaluate_bpb
from slm.models.model import ModelConfig, TokenLM, count_params

DEFAULT_TRAIN = dict(
    tokenizer="mbpe16k",
    weights={"en": 0.35, "de": 0.055, "fr": 0.055, "es": 0.055, "ru": 0.055, "zh": 0.055,
             "ja": 0.055, "ar": 0.055, "hi": 0.055, "sw": 0.055, "tr": 0.055, "tool": 0.10},
    batch_size=16, seq_len=512, steps=10000, lr=3e-3, warmup=500, min_lr_frac=0.1,
    weight_decay=0.1, beta2=0.95, grad_clip=1.0, seed=0, eval_every=2000,
    eval_tokens=32768, final_eval_tokens=131072, log_every=50, dtype="float32",
    save=True,
)


class AdamW2D(optim.Adam):
    """AdamW with decoupled weight decay applied only to matrices (ndim >= 2)."""

    def __init__(self, learning_rate, betas, eps=1e-8, weight_decay=0.1):
        super().__init__(learning_rate=learning_rate, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def apply_single(self, gradient, parameter, state):
        if parameter.ndim >= 2:
            lr = self.learning_rate.astype(gradient.dtype)
            parameter = parameter * (1 - lr * self.weight_decay)
        return super().apply_single(gradient, parameter, state)


def make_schedule(tc):
    warm = optim.linear_schedule(tc["lr"] * 0.01, tc["lr"], tc["warmup"])
    cos = optim.cosine_decay(tc["lr"], tc["steps"] - tc["warmup"], tc["lr"] * tc["min_lr_frac"])
    return optim.join_schedules([warm, cos], [tc["warmup"]])


def run(model_cfg: ModelConfig, tc: dict, run_name: str, results_root="results"):
    out = os.path.join(results_root, run_name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "metrics.json")):
        print(f"{run_name}: already finished, skipping")
        return json.load(open(os.path.join(out, "metrics.json")))
    log_f = open(os.path.join(out, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    mx.random.seed(tc["seed"])
    np.random.seed(tc["seed"])
    model = TokenLM(model_cfg)
    if tc.get("init_from"):  # continued training (e.g. long-context stage) from a saved run
        model.load_weights(tc["init_from"])
    if tc["dtype"] != "float32":
        model.set_dtype(getattr(mx, tc["dtype"]))
    n_params = count_params(model)
    n_nonemb = count_params(model, exclude_embedding=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(model=model_cfg.to_dict(), train=tc, params=n_params,
                       params_nonembedding=n_nonemb, host=platform.platform(),
                       mlx=mx.__version__), f, indent=2)
    log(f"== {run_name} layers={model_cfg.layer_types()} params={n_params/1e6:.2f}M "
        f"(non-emb {n_nonemb/1e6:.2f}M) lr={tc['lr']} steps={tc['steps']} "
        f"tokens={tc['steps']*tc['batch_size']*tc['seq_len']/1e6:.0f}M")

    opt = AdamW2D(make_schedule(tc), betas=[0.9, tc["beta2"]], weight_decay=tc["weight_decay"])
    loader = MixtureLoader(tc["tokenizer"], tc["weights"], tc["batch_size"], tc["seq_len"], seed=tc["seed"])
    if tc.get("recall_frac", 0) > 0:  # mix in synthetic long-range recall rows
        from slm.data.recall import RecallMixLoader
        loader = RecallMixLoader(loader, tc["tokenizer"], frac=tc["recall_frac"], seed=tc["seed"] + 1)

    def loss_fn(m, x, y):
        logits, _ = m(x)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, gnorm = optim.clip_grad_norm(grads, tc["grad_clip"])
        opt.update(model, grads)
        return loss, gnorm

    curve = open(os.path.join(out, "curve.jsonl"), "a")
    srcs = list(tc["weights"])
    t_start = time.time()
    t_last, tok_last = t_start, 0
    tokens = 0
    loss_acc, n_acc = 0.0, 0
    diverged = False
    for it in range(1, tc["steps"] + 1):
        batch = loader.next()
        x, y = mx.array(batch[:, :-1]), mx.array(batch[:, 1:])
        loss, gnorm = step(x, y)
        mx.eval(state, loss, gnorm)
        tokens += x.size
        lv = loss.item()
        if not math.isfinite(lv):
            log(f"step {it}: non-finite loss, stopping")
            diverged = True
            break
        loss_acc += lv
        n_acc += 1
        if it % tc["log_every"] == 0:
            now = time.time()
            tps = (tokens - tok_last) / (now - t_last)
            rec = dict(step=it, tokens=tokens, loss=loss_acc / n_acc, gnorm=gnorm.item(),
                       lr=float(opt.learning_rate), tok_s=tps, elapsed=now - t_start)
            curve.write(json.dumps(rec) + "\n")
            curve.flush()
            log(f"step {it:6d} loss {rec['loss']:.4f} gnorm {rec['gnorm']:.2f} "
                f"lr {rec['lr']:.2e} {tps:,.0f} tok/s {rec['elapsed']/60:.1f}min")
            t_last, tok_last = now, tokens
            loss_acc, n_acc = 0.0, 0
        if it % tc["eval_every"] == 0 and it < tc["steps"]:
            ev = evaluate_bpb(model, tc["tokenizer"], srcs, tc["seq_len"], tc["eval_tokens"])
            curve.write(json.dumps(dict(step=it, tokens=tokens, eval=ev)) + "\n")
            log(f"  eval@{it}: all_text bpb {ev['all_text']['bpb']:.4f} "
                f"macro {ev['macro_lang']['bpb']:.4f} en {ev['en']['bpb']:.4f} tool {ev['tool']['bpb']:.4f}")
    train_time = time.time() - t_start
    ev = evaluate_bpb(model, tc["tokenizer"], srcs, tc["seq_len"], tc["final_eval_tokens"])
    peak = mx.get_peak_memory() / 1e9
    metrics = dict(run=run_name, params=n_params, params_nonembedding=n_nonemb, tokens=tokens,
                   train_seconds=train_time, train_tok_s=tokens / train_time, diverged=diverged,
                   peak_train_mem_gb=peak, final_train_loss=lv, eval=ev)
    log(f"FINAL {run_name}: all_text bpb {ev['all_text']['bpb']:.4f} macro {ev['macro_lang']['bpb']:.4f} "
        f"tool {ev['tool']['bpb']:.4f} | {train_time/60:.1f} min, {tokens/train_time:,.0f} tok/s, peak {peak:.2f} GB")
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if tc["save"]:
        model.save_weights(os.path.join(out, "model.safetensors"))
    return metrics


def load_config(path, overrides=None):
    cfg = json.load(open(path))
    mc = ModelConfig(**cfg["model"])
    tc = copy.deepcopy(DEFAULT_TRAIN)
    tc.update(cfg.get("train", {}))
    for k, v in (overrides or {}).items():
        if v is not None:
            tc[k] = v
    return mc, tc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    mc, tc = load_config(a.config, dict(lr=a.lr, steps=a.steps, seed=a.seed))
    name = a.run or os.path.splitext(os.path.basename(a.config))[0]
    run(mc, tc, name)
