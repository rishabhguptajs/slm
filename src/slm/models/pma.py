"""Priority Memory Attention (PMA): attention over a fixed bank of M exact slots.

Each token gets a learned salience s_i (per head). Its priority is
    p_i = s_i + lam_h * i
with lam_h a fixed per-head recency rate (multi-scale across heads). At time t the
memory holds exactly the top-M tokens by priority among all tokens seen so far.

Because p_i never changes after it is written (the recency term is the same
offset for every token when compared at a common time), the streaming policy
"insert new token, evict the lowest priority if full" yields, at every t, the
top-M of {p_i : i <= t}. That set can be computed *in parallel* for training:
    rank_t(i) = #{j <= t : p_j beats p_i},  keep(t, i) = i <= t and rank_t(i) < M
so train-time and inference-time behaviour are identical. Memory is O(M) forever.

lam large  -> sliding window of size M.   lam -> 0 -> keep the M most salient tokens.
Salience also enters the read as a bias log(sigmoid(s_i)), which is what gives s
its gradient (tokens that are useful to read get higher salience, hence live longer).
No positional encoding (NoPE): distances in a stream are unbounded.
"""
import math

import mlx.core as mx
import mlx.nn as nn

from slm.models.layers import RMSNorm, merge_heads, split_heads


class PriorityMemoryAttention(nn.Module):
    def __init__(self, d, n_heads, n_slots=64, lams=None, head_dim=None):
        super().__init__()
        self.h = n_heads
        self.dh = head_dim or d // n_heads
        self.M = n_slots
        # fixed multi-scale recency rates, e.g. 4 heads -> 1, 0.1, 0.01, 0.001
        self.lams = lams or [10.0 ** (-3.0 * i / max(1, n_heads - 1)) for i in range(n_heads)]
        inner = self.h * self.dh
        self.wq = nn.Linear(d, inner, bias=False)
        self.wkv = nn.Linear(d, 2 * inner, bias=False)
        self.ws = nn.Linear(d, self.h, bias=True)
        self.qn = RMSNorm(self.dh)
        self.kn = RMSNorm(self.dh)
        self.wo = nn.Linear(inner, d, bias=False)

    def __call__(self, x, state=None):
        B, T, _ = x.shape
        q = self.qn(split_heads(self.wq(x), self.h))
        k, v = mx.split(self.wkv(x), 2, axis=-1)
        k, v = self.kn(split_heads(k, self.h)), split_heads(v, self.h)
        s = self.ws(x).astype(mx.float32).transpose(0, 2, 1)  # [B,H,T]
        lam = mx.array(self.lams, dtype=mx.float32)[None, :, None]
        p = mx.stop_gradient(s) + lam * mx.arange(T, dtype=mx.float32)[None, None, :]

        if state is not None and state["k"] is not None:
            Ms = state["k"].shape[2]
            keys = mx.concatenate([state["k"], k], axis=2)
            vals = mx.concatenate([state["v"], v], axis=2)
            pri = mx.concatenate([state["p"], p], axis=2)
            sal = mx.concatenate([state["s"], s], axis=2)
        else:
            Ms = 0
            keys, vals, pri, sal = k, v, p, s
        N = Ms + T

        # beats[j, c] : candidate j outranks candidate c (ties -> newer wins)
        idx = mx.arange(N)
        pj, pc = pri[..., :, None], pri[..., None, :]
        beats = (pj > pc) | ((pj == pc) & (idx[:, None] > idx[None, :]))
        beats = beats.astype(mx.float32)  # [B,H,N,N]; counts < 2^24 so float is exact
        # rank of candidate c as seen by query t (new token index t in 0..T-1).
        # Prefix count via a lower-triangular matmul, NOT mx.cumsum(axis=-2): in MLX 0.32.2
        # an int32 cumsum along a non-last axis of a sliced view intermittently wrote out
        # of bounds and corrupted model weights (see docs/notes_mlx_bugs.md).
        slot_part = beats[..., :Ms, :].sum(axis=-2, keepdims=True) if Ms else 0
        tri = mx.tril(mx.ones((T, T), dtype=mx.float32))
        new_part = tri @ beats[..., Ms:, :]  # [B,H,T,N]
        rank = new_part + slot_part
        visible = idx[None, :] <= (Ms + mx.arange(T))[:, None]  # [T,N]
        keep = visible & (rank < self.M) & (pri[..., None, :] > -1e30)
        bias = nn.log_sigmoid(sal)[..., None, :]
        # large finite negative, not -inf: fused SDPA kernels can produce NaN when a
        # whole tile of a row is -inf (exp(-inf - -inf)); found as an intermittent NaN.
        neg = -1e9 if q.dtype == mx.float32 else -3e4
        mask = mx.where(keep, bias, neg).astype(q.dtype)
        o = mx.fast.scaled_dot_product_attention(q, keys, vals, scale=1.0 / math.sqrt(self.dh), mask=mask)
        y = self.wo(merge_heads(o))

        new_state = None
        if state is not None:
            M = min(self.M, N)
            top = mx.argsort(-pri, axis=-1)[..., :M]  # [B,H,M]
            g = lambda a: mx.take_along_axis(a, top[..., None], axis=2)
            new_state = {
                "k": g(keys), "v": g(vals),
                # re-express priorities relative to the next chunk's start
                "p": mx.take_along_axis(pri, top, axis=-1) - lam * T,
                "s": mx.take_along_axis(sal, top, axis=-1),
            }
        return y, new_state

    def init_state(self, B):
        return {"k": None, "v": None, "p": None, "s": None}
