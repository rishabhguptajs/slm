"""Chunked-parallel forms must match token-by-token recurrences, and streaming
(chunk-by-chunk with carried state) must match one-shot processing."""
import mlx.core as mx

from slm.models.recurrent import gdn_chunked, gdn_reference, gla_chunked, unit_lower_inverse, RecurrentMixer
from slm.models.layers import l2norm


def rand_inputs(B=2, H=3, T=100, d=16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, T, d)) * d ** -0.5
    k = l2norm(mx.random.normal((B, H, T, d)))
    v = mx.random.normal((B, H, T, d))
    log_a = -mx.random.uniform(0, 0.3, (B, H, T))
    beta = mx.random.uniform(0, 1, (B, H, T))
    return q, k, v, log_a, beta


def test_inverse():
    mx.random.seed(1)
    A = mx.tril(mx.random.normal((4, 64, 64)), k=-1) * 0.5
    X = mx.eye(64) + A
    inv = unit_lower_inverse(X)
    err = mx.abs(inv @ X - mx.eye(64)).max().item()
    assert err < 1e-3, err


def test_inverse_odd_sizes():
    # short final chunks at inference give n that is not a power of two (crashed before 2026-10-06)
    mx.random.seed(2)
    for n in (17, 37, 63):
        X = mx.eye(n) + mx.tril(mx.random.normal((3, n, n)), k=-1) * 0.5
        err = mx.abs(unit_lower_inverse(X) @ X - mx.eye(n)).max().item()
        assert err < 1e-3, (n, err)


def test_gdn_short_sequences_match_reference():
    for T in (17, 37, 63):
        q, k, v, la, b = rand_inputs(T=T)
        o_ref, S_ref = gdn_reference(q, k, v, la, b)
        o, S = gdn_chunked(q, k, v, la, b, C=64)
        assert mx.abs(o - o_ref).max().item() < 1e-3, T
        assert mx.abs(S - S_ref).max().item() < 1e-3, T


def test_gdn_matches_reference():
    q, k, v, la, b = rand_inputs()
    o_ref, S_ref = gdn_reference(q, k, v, la, b)
    for C in (16, 32, 64):
        o, S = gdn_chunked(q, k, v, la, b, C=C)
        assert mx.abs(o - o_ref).max().item() < 1e-3, C
        assert mx.abs(S - S_ref).max().item() < 1e-3, C


def test_gdn_worst_case_aligned_keys():
    # identical keys, beta=1, no decay: the regime where Neumann-series inversion blows up
    B, H, T, d = 1, 1, 64, 8
    k = l2norm(mx.ones((B, H, T, d)))
    q = mx.random.normal((B, H, T, d))
    v = mx.random.normal((B, H, T, d))
    o_ref, _ = gdn_reference(q, k, v, mx.zeros((B, H, T)), mx.ones((B, H, T)))
    o, _ = gdn_chunked(q, k, v, mx.zeros((B, H, T)), mx.ones((B, H, T)), C=64)
    assert mx.abs(o - o_ref).max().item() < 1e-3


def test_gla_matches_reference():
    q, k, v, la, _ = rand_inputs()
    S = mx.zeros((2, 3, 16, 16))
    outs = []
    for t in range(q.shape[2]):
        S = mx.exp(la[:, :, t])[..., None, None] * S + k[:, :, t][..., :, None] @ v[:, :, t][..., None, :]
        outs.append((S.swapaxes(-1, -2) @ q[:, :, t][..., None])[..., 0])
    o_ref = mx.stack(outs, axis=2)
    o, _ = gla_chunked(q, k, v, la, C=32)
    assert mx.abs(o - o_ref).max().item() < 1e-3


def test_mixer_streaming_equivalence():
    for kind in ("gdn", "gla"):
        mx.random.seed(0)
        m = RecurrentMixer(64, 4, kind=kind, chunk=16)
        x = mx.random.normal((2, 50, 64))
        y_full, _ = m(x)
        st = m.init_state(2)
        ys = []
        for s, e in [(0, 7), (7, 8), (8, 40), (40, 50)]:
            y, st = m(x[:, s:e], st)
            ys.append(y)
        y_stream = mx.concatenate(ys, axis=1)
        assert mx.abs(y_full - y_stream).max().item() < 1e-3, kind
