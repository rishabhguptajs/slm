"""Efficiency measurements (batch 1, the on-device streaming setting).

Measured on the M4 (wall clock, MLX):
  - decode tok/s and memory after streaming L tokens, for growing L
  - prefill tok/s (chunked, 512-token chunks)
  - exact inference-state bytes (KV cache / recurrent state / slots) vs L
Analytic PROXIES (clearly labelled; not energy measurements):
  - MACs per decoded token at context L
  - bytes moved per decoded token = weight bytes + state bytes read/written
"""
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from slm.models.model import count_params


def state_bytes(state):
    n = 0
    for st in state:
        for v in (st or {}).values():
            if isinstance(v, mx.array):
                n += v.nbytes
    return n


def analytic_per_token(cfg, L, weight_bytes_per_param=4, act_bytes=4):
    """MACs and bytes moved to decode one token at context length L (batch 1)."""
    d, H, V = cfg.d_model, cfg.n_heads, cfg.vocab_size
    dh = d // H
    macs = d * V  # tied output head
    state_rw = 0  # bytes of state read + written for this token
    for t in cfg.layer_types():
        macs += 3 * d * cfg.mlp_hidden
        if t == "A":
            macs += 4 * d * d + 2 * L * d
            state_rw += (2 * L * d + 2 * d) * act_bytes  # read all K,V; append one
        elif t == "W":
            w = min(L, cfg.window)
            macs += 4 * d * d + 2 * w * d
            state_rw += (2 * w * d + 2 * d) * act_bytes
        elif t in "GD":
            macs += 5 * d * d + 2 * d * H + 3 * d * cfg.conv_k
            macs += (3 if t == "D" else 2) * H * dh * dh
            state_rw += 2 * H * dh * dh * 4 + 2 * 3 * d * (cfg.conv_k - 1) * act_bytes
        elif t == "P":
            m = min(L, cfg.n_slots)
            macs += 4 * d * d + d * H + 2 * m * d
            state_rw += (2 * m * d + 2 * H * m) * act_bytes + 2 * d * act_bytes
        elif t == "C":
            macs += 4 * d * d + d * 3
            state_rw += 2 * d * 2 * act_bytes
    return dict(macs=macs, flops=2 * macs, state_bytes_rw=state_rw)


def measure(model, lengths=(512, 2048, 8192, 32768), decode_steps=32, chunk=512, seed=0):
    cfg = model.cfg
    n_params = count_params(model)
    wbytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    rng = np.random.default_rng(seed)
    out = dict(params=n_params, weight_bytes=wbytes, by_length={})
    # prefill throughput from an empty state
    ids = mx.array(rng.integers(0, cfg.vocab_size, (1, 4096)).astype(np.int32))
    st = model.init_state(1)
    lg, st = model(ids[:, :chunk], st)  # warm-up
    mx.eval(lg, st)
    st = model.init_state(1)
    t0 = time.time()
    for s in range(0, 4096, chunk):
        lg, st = model(ids[:, s:s + chunk], st)
        mx.eval(lg, st)
    out["prefill_tok_s"] = 4096 / (time.time() - t0)

    # stream up to each length, then time decode
    st = model.init_state(1)
    done = 0
    for L in sorted(lengths):
        mx.clear_cache()
        mx.reset_peak_memory()
        while done < L:
            n = min(chunk, L - done)
            x = mx.array(rng.integers(0, cfg.vocab_size, (1, n)).astype(np.int32))
            lg, st = model(x, st)
            mx.eval(lg, st)
            done += n
        tok = mx.array([[1]], dtype=mx.int32)
        for _ in range(4):  # warm-up at this length (does not advance 'done' accounting much)
            lg, st2 = model(tok, st)
            mx.eval(lg, st2)
        t0 = time.time()
        st2 = st
        for _ in range(decode_steps):
            lg, st2 = model(tok, st2)
            mx.eval(lg, st2)
        dt = (time.time() - t0) / decode_steps
        an = analytic_per_token(cfg, L)
        out["by_length"][L] = dict(
            decode_tok_s=1.0 / dt, decode_ms=dt * 1000,
            state_bytes=state_bytes(st), peak_mem_bytes=mx.get_peak_memory(),
            active_mem_bytes=mx.get_active_memory(),
            proxy_macs_per_token=an["macs"], proxy_flops_per_token=an["flops"],
            proxy_bytes_moved_per_token=wbytes + an["state_bytes_rw"],
        )
        r = out["by_length"][L]
        print(f"  L={L:6d}: decode {r['decode_tok_s']:7.1f} tok/s  state {r['state_bytes']/1e6:8.2f} MB  "
              f"peak {r['peak_mem_bytes']/1e6:8.1f} MB  proxy {r['proxy_flops_per_token']/1e6:.1f} MFLOP/tok", flush=True)
    return out
