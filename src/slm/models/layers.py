"""Shared building blocks. Plain ops only (matmul, elementwise, cumsum, softmax)
so everything ports 1:1 to PyTorch / C / integer kernels.

Every sequence mixer implements:
    __call__(x, state=None) -> (y, new_state)
x: [B, T, D]. state=None means "fresh stream". The same code path serves
parallel training (state=None, T large) and streaming inference (state carried,
T = chunk or 1).
"""
import math

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w_in = nn.Linear(d, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d, bias=False)

    def __call__(self, x):
        a, b = mx.split(self.w_in(x), 2, axis=-1)
        return self.w_out(nn.silu(a) * b)


class ShortConv(nn.Module):
    """Causal depthwise conv over time (kernel k). Streaming state = last k-1 inputs."""

    def __init__(self, channels, k=4):
        super().__init__()
        self.k = k
        self.weight = mx.random.normal((channels, k)) * (1.0 / math.sqrt(k))
        self.bias = mx.zeros((channels,))

    def __call__(self, x, state=None):
        B, T, C = x.shape
        if state is None:
            state = mx.zeros((B, self.k - 1, C), dtype=x.dtype)
        xp = mx.concatenate([state, x], axis=1)  # [B, T+k-1, C]
        # depthwise conv == sum_i xp[:, i:i+T] * weight[:, i]  (native kernel is faster)
        y = mx.conv1d(xp, self.weight[:, :, None].astype(x.dtype), groups=C) + self.bias
        new_state = xp[:, -(self.k - 1):, :]
        return y, new_state


def l2norm(x, eps=1e-6):
    return x * mx.rsqrt((x * x).sum(-1, keepdims=True) + eps)


def split_heads(x, h):
    B, T, D = x.shape
    return x.reshape(B, T, h, D // h).transpose(0, 2, 1, 3)  # [B,H,T,dh]


def merge_heads(x):
    B, H, T, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)


# ---------------------------------------------------------------- attention
class Attention(nn.Module):
    """Causal softmax attention with RoPE. window=None -> full (growing KV cache);
    window=W -> sliding window (KV cache capped at W)."""

    def __init__(self, d, n_heads, n_kv_heads=None, window=None, rope_base=10000.0):
        super().__init__()
        self.h = n_heads
        self.hkv = n_kv_heads or n_heads
        self.dh = d // n_heads
        self.window = window
        self.wq = nn.Linear(d, self.h * self.dh, bias=False)
        self.wkv = nn.Linear(d, 2 * self.hkv * self.dh, bias=False)
        self.wo = nn.Linear(self.h * self.dh, d, bias=False)
        self.rope = nn.RoPE(self.dh, traditional=False, base=rope_base)

    def __call__(self, x, state=None):
        B, T, _ = x.shape
        q = split_heads(self.wq(x), self.h)
        k, v = mx.split(self.wkv(x), 2, axis=-1)
        k, v = split_heads(k, self.hkv), split_heads(v, self.hkv)
        offset = 0 if state is None else state["offset"]
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if state is not None and state["k"] is not None:
            k = mx.concatenate([state["k"], k], axis=2)
            v = mx.concatenate([state["v"], v], axis=2)
        S = k.shape[2]  # total keys visible
        scale = 1.0 / math.sqrt(self.dh)
        if state is None and self.window is None:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        else:
            qi = mx.arange(S - T, S)[:, None]
            ki = mx.arange(S)[None, :]
            m = ki <= qi
            if self.window is not None:
                m = m & (ki > qi - self.window)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=m)
        y = self.wo(merge_heads(o))
        new_state = None
        if state is not None:
            if self.window is not None:
                k, v = k[:, :, -(self.window - 1):], v[:, :, -(self.window - 1):]
            new_state = {"k": k, "v": v, "offset": offset + T}
        return y, new_state

    def init_state(self, B):
        return {"k": None, "v": None, "offset": 0}
