"""Fixed-state recurrent mixers: gated linear attention (scalar decay, Mamba-2/SSD
style) and Gated DeltaNet (delta-rule). State per head is a dk x dv matrix,
independent of stream length.

Training uses an exact chunkwise-parallel algorithm; streaming uses the same
function with a carried state (a chunk of length 1 is the plain recurrence).

Notation (per head, state S in R^{dk x dv}, o_t = S_t^T q_t):
  GLA: S_t = a_t S_{t-1} + k_t v_t^T
  GDN: S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T
"""
import math

import mlx.core as mx
import mlx.nn as nn

from slm.models.layers import RMSNorm, ShortConv, l2norm, merge_heads, split_heads


def _chunk(x, C):
    """[B,H,T,...] -> [B,H,n,C,...] with zero padding."""
    B, H, T = x.shape[:3]
    n = -(-T // C)
    pad = n * C - T
    if pad:
        widths = [(0, 0), (0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 3)
        x = mx.pad(x, widths)
    return x.reshape(B, H, n, C, *x.shape[3:])


def _decay_matrix(g):
    """g: [..., C] cumulative log-decay. Returns Gamma[..., i, j] = exp(g_i - g_j) for
    i >= j, else 0."""
    C = g.shape[-1]
    diff = g[..., :, None] - g[..., None, :]
    tri = mx.tril(mx.ones((C, C), dtype=mx.bool_))
    return mx.exp(mx.where(tri, diff, -mx.inf))


def unit_lower_inverse(X, base=16):
    """Inverse of unit lower-triangular X [..., n, n] by batched recursive block
    forward substitution (numerically stable, O(log n) levels of matmuls)."""
    n = X.shape[-1]
    if n <= base:
        eye = mx.eye(n, dtype=X.dtype)
        rows = [mx.broadcast_to(eye[0:1], X.shape[:-2] + (1, n))]
        for i in range(1, n):
            prev = mx.concatenate(rows, axis=-2)  # [..., i, n]
            r = eye[i:i + 1] - X[..., i:i + 1, :i] @ prev
            rows.append(r)
        return mx.concatenate(rows, axis=-2)
    h = n // 2
    if 2 * h == n:
        both = mx.stack([X[..., :h, :h], X[..., h:, h:]], axis=0)
        inv = unit_lower_inverse(both, base)
        I1, I2 = inv[0], inv[1]
    else:  # odd n (short final chunk at inference): diagonal blocks differ in size
        I1 = unit_lower_inverse(X[..., :h, :h], base)
        I2 = unit_lower_inverse(X[..., h:, h:], base)
    Bm = -(I2 @ X[..., h:, :h] @ I1)
    top = mx.concatenate([I1, mx.zeros(I1.shape[:-1] + (n - h,), dtype=I1.dtype)], axis=-1)
    bot = mx.concatenate([Bm, I2], axis=-1)
    return mx.concatenate([top, bot], axis=-2)


def gla_chunked(q, k, v, log_a, S0=None, C=64):
    """q,k: [B,H,T,dk], v: [B,H,T,dv], log_a: [B,H,T] (<=0). Returns o [B,H,T,dv], S.
    Fully parallel: chunk-start states come from one decay-weighted matmul over chunks."""
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    C = min(C, T)
    q, k, v, la = _chunk(q, C), _chunk(k, C), _chunk(v, C), _chunk(log_a, C)
    n = q.shape[2]
    g = mx.cumsum(la, axis=-1)  # [B,H,n,C]
    Gam = _decay_matrix(g)
    O_intra = ((q @ k.swapaxes(-1, -2)) * Gam) @ v
    q_dec = q * mx.exp(g)[..., None]
    k_end = k * mx.exp(g[..., -1:] - g)[..., None]
    kv = (k_end.swapaxes(-1, -2) @ v).reshape(B, H, n, dk * dv)  # per-chunk contribution
    L = mx.cumsum(g[..., -1], axis=-1)       # [B,H,n] log decay through end of chunk c
    Lprev = L - g[..., -1]                   # through end of chunk c-1
    strict = mx.tril(mx.ones((n, n), dtype=mx.bool_), k=-1)
    D = mx.exp(mx.where(strict, Lprev[..., :, None] - L[..., None, :], -mx.inf))  # [B,H,n,n]
    S_start = (D @ kv).reshape(B, H, n, dk, dv)
    if S0 is not None:
        S_start = S_start + mx.exp(Lprev)[..., None, None] * S0[:, :, None]
    o = O_intra + q_dec @ S_start
    o = o.reshape(B, H, n * C, dv)[:, :, :T]
    S = S_start[:, :, -1] * mx.exp(g[..., -1, -1])[..., None, None] + kv[:, :, -1].reshape(B, H, dk, dv)
    return o, S


def gdn_chunked(q, k, v, log_a, beta, S0=None, C=64):
    """Gated delta rule, exact chunkwise form (WY/UT transform).
    q,k: [B,H,T,dk] (k L2-normalised), v: [B,H,T,dv], log_a, beta: [B,H,T]."""
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    C = min(C, T)
    q, k, v = _chunk(q, C), _chunk(k, C), _chunk(v, C)
    la, bt = _chunk(log_a, C), _chunk(beta, C)
    n = q.shape[2]
    g = mx.cumsum(la, axis=-1)
    Gam = _decay_matrix(g)  # incl. diagonal (=1)
    strict = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    A = mx.where(strict, Gam * (k @ k.swapaxes(-1, -2)), 0.0) * bt[..., None, :]
    Minv = unit_lower_inverse(mx.eye(C, dtype=A.dtype) + A)
    eg = mx.exp(g)[..., None]
    U_t = Minv @ v                       # [B,H,n,C,dv]
    W = Minv @ (k * eg)                  # [B,H,n,C,dk]
    P = (Gam * (q @ k.swapaxes(-1, -2))) * bt[..., None, :]
    q_dec = q * eg
    k_end = k * (bt * mx.exp(g[..., -1:] - g))[..., None]
    cdec = mx.exp(g[..., -1])[..., None, None]
    S = mx.zeros((B, H, dk, dv), dtype=q.dtype) if S0 is None else S0
    # split (not index) along chunks: the VJP of split is one concatenate, whereas
    # indexing creates a full-size zero cotangent per chunk.
    sp = lambda a: [x.squeeze(2) for x in mx.split(a, n, axis=2)] if n > 1 else [a.squeeze(2)]
    U_t, W, P, q_dec, k_end_T, cdec = map(sp, (U_t, W, P, q_dec, k_end.swapaxes(-1, -2), cdec))
    outs = []
    for c in range(n):
        U = U_t[c] - W[c] @ S
        outs.append(q_dec[c] @ S + P[c] @ U)
        S = cdec[c] * S + k_end_T[c] @ U
    o = mx.stack(outs, axis=2).reshape(B, H, n * C, dv)[:, :, :T]
    return o, S


def gdn_reference(q, k, v, log_a, beta, S0=None):
    """Token-by-token recurrence, for testing only."""
    B, H, T, dk = q.shape
    S = mx.zeros((B, H, dk, v.shape[-1])) if S0 is None else S0
    outs = []
    for t in range(T):
        a = mx.exp(log_a[:, :, t])[..., None, None]
        b = beta[:, :, t][..., None, None]
        kt, vt, qt = k[:, :, t][..., :, None], v[:, :, t][..., None, :], q[:, :, t][..., :, None]
        S = a * (S - b * kt @ (kt.swapaxes(-1, -2) @ S)) + b * kt @ vt
        outs.append((S.swapaxes(-1, -2) @ qt)[..., 0])
    return mx.stack(outs, axis=2), S


class RecurrentMixer(nn.Module):
    """Shared wrapper for GLA ('gla') and Gated DeltaNet ('gdn').

    x -> [q,k,v] -> short causal conv + SiLU -> q,k L2-norm -> recurrence ->
    per-head RMSNorm -> * SiLU(gate) -> out proj.
    Decay a_t = exp(-exp(A_log) * softplus(w_a x + dt_bias)) (Mamba-2 style).
    """

    def __init__(self, d, n_heads, kind="gdn", head_dim=None, conv_k=4, chunk=64):
        super().__init__()
        self.kind = kind
        self.h = n_heads
        self.dh = head_dim or d // n_heads
        self.chunk = chunk
        inner = self.h * self.dh
        self.w_qkv = nn.Linear(d, 3 * inner, bias=False)
        self.w_gate = nn.Linear(d, inner, bias=False)
        n_scalar = 2 * self.h if kind == "gdn" else self.h
        self.w_ab = nn.Linear(d, n_scalar, bias=False)
        self.conv = ShortConv(3 * inner, conv_k)
        # decay init: per-head timescales spread log-uniformly (as in Mamba-2)
        self.A_log = mx.log(mx.linspace(1.0, 16.0, self.h))
        dt = mx.exp(mx.linspace(math.log(1e-3), math.log(1e-1), self.h))
        self.dt_bias = dt + mx.log(-mx.expm1(-dt))  # inverse softplus
        self.norm = RMSNorm(self.dh)
        self.wo = nn.Linear(inner, d, bias=False)

    def __call__(self, x, state=None):
        B, T, _ = x.shape
        conv_state = None if state is None else state["conv"]
        qkv, conv_state = self.conv(self.w_qkv(x), conv_state)
        qkv = nn.silu(qkv)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q, k, v = split_heads(q, self.h), split_heads(k, self.h), split_heads(v, self.h)
        q = l2norm(q.astype(mx.float32)) * (self.dh ** -0.5)
        k = l2norm(k.astype(mx.float32))
        v = v.astype(mx.float32)
        ab = self.w_ab(x).astype(mx.float32).transpose(0, 2, 1)  # [B, n_scalar, T]
        dt = nn.softplus(ab[:, :self.h] + self.dt_bias[None, :, None])
        log_a = -mx.exp(self.A_log)[None, :, None] * dt
        S0 = None if state is None else state["S"]
        if self.kind == "gdn":
            beta = mx.sigmoid(ab[:, self.h:])
            o, S = gdn_chunked(q, k, v, log_a, beta, S0, self.chunk)
        else:
            o, S = gla_chunked(q, k, v, log_a, S0, self.chunk)
        o = self.norm(o.astype(x.dtype))
        y = merge_heads(o) * nn.silu(self.w_gate(x))
        y = self.wo(y)
        new_state = None if state is None else {"S": S, "conv": conv_state}
        return y, new_state

    def init_state(self, B):
        inner = self.h * self.dh
        return {"S": mx.zeros((B, self.h, self.dh, self.dh)),
                "conv": mx.zeros((B, self.conv.k - 1, 3 * inner))}
