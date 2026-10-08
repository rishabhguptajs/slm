"""Every model type: streaming chunk-by-chunk == one-shot forward (for windows /
slot counts larger than or smaller than the sequence)."""
import mlx.core as mx
import pytest

from slm.models.model import ModelConfig, TokenLM, count_params
from slm.models.pma import PriorityMemoryAttention


@pytest.mark.parametrize("pattern", ["A", "W", "G", "D", "P", "C", "DDPD", "CDWD"])
def test_stream_equals_full(pattern):
    mx.random.seed(0)
    cfg = ModelConfig(vocab_size=97, d_model=64, n_layers=4, pattern=pattern, n_heads=4,
                      mlp_hidden=128, window=16, n_slots=8, chunk=16)
    m = TokenLM(cfg)
    toks = mx.random.randint(0, 97, (2, 60))
    full, _ = m(toks)
    st = m.init_state(2)
    outs = []
    for s, e in [(0, 5), (5, 6), (6, 30), (30, 31), (31, 60)]:
        o, st = m(toks[:, s:e], st)
        outs.append(o)
    stream = mx.concatenate(outs, axis=1)
    err = mx.abs(full - stream).max().item()
    assert err < 2e-3, (pattern, err)


def test_pma_is_topM_by_priority():
    """Direct check: which keys each query can see equals a brute-force heap simulation."""
    import numpy as np
    mx.random.seed(3)
    layer = PriorityMemoryAttention(32, 2, n_slots=5, lams=[0.5, 0.0])
    x = mx.random.normal((1, 40, 32))
    s = np.array(layer.ws(x).transpose(0, 2, 1))[0]  # [H,T]
    for h, lam in enumerate(layer.lams):
        heap = []
        for t in range(40):
            heap.append((s[h, t] + lam * t, t))
            heap = sorted(heap, reverse=True)[:5]
        # the final memory must be the top-5 overall
        top = sorted([(s[h, i] + lam * i, i) for i in range(40)], reverse=True)[:5]
        assert {i for _, i in heap} == {i for _, i in top}


def test_param_count_runs():
    cfg = ModelConfig(vocab_size=1000, d_model=64, n_layers=2, pattern="D")
    assert count_params(TokenLM(cfg)) > 0


@pytest.mark.parametrize("pattern", ["A", "W", "G", "D", "P", "C"])
def test_forward_never_modifies_weights(pattern):
    """Regression for an MLX out-of-bounds write (int32 cumsum on a non-last axis of a
    sliced view) that silently corrupted weights. Run full + streaming passes many
    times and require the weights to be bit-identical afterwards."""
    import numpy as np
    from mlx.utils import tree_flatten
    mx.random.seed(0)
    cfg = ModelConfig(vocab_size=97, d_model=64, n_layers=4, pattern=pattern, n_heads=4,
                      mlp_hidden=128, window=16, n_slots=8, chunk=16)
    m = TokenLM(cfg)
    toks = mx.random.randint(0, 97, (2, 60))
    mx.eval(m.parameters())
    ref = {k: np.array(v) for k, v in tree_flatten(m.parameters())}
    for _ in range(15):
        full, _ = m(toks)
        st = m.init_state(2)
        for s, e in [(0, 5), (5, 6), (6, 30), (30, 31), (31, 60)]:
            o, st = m(toks[:, s:e], st)
            mx.eval(o, st, full)
    for k, v in tree_flatten(m.parameters()):
        assert np.array_equal(np.array(v), ref[k]), (pattern, k)
