"""Streaming generation for any model with init_state(): chunked prefill, then
token-by-token decode with the carried state (constant memory for fixed-state models)."""
import mlx.core as mx
import numpy as np


def prefill(model, ids, state=None, chunk=512):
    """Feed ids [B, T] through the model chunk by chunk. Returns (last_logits, state)."""
    if state is None:
        state = model.init_state(ids.shape[0])
    logits = None
    for s in range(0, ids.shape[1], chunk):
        logits, state = model(ids[:, s:s + chunk], state)
        mx.eval(logits, state)
    return logits[:, -1], state


def greedy(model, prompt_ids, max_new=64, stop_ids=(), chunk=512):
    ids = mx.array(np.asarray(prompt_ids, dtype=np.int32)[None])
    logits, state = prefill(model, ids, chunk=chunk)
    out = []
    for _ in range(max_new):
        nxt = int(mx.argmax(logits, axis=-1).item())
        if nxt in stop_ids:
            break
        out.append(nxt)
        lg, state = model(mx.array([[nxt]], dtype=mx.int32), state)
        logits = lg[:, -1]
        mx.eval(logits, state)
    return out


def continuation_logprob(model, prompt_ids, target_ids, chunk=512):
    """Sum of log p(target | prompt) in nats, teacher forced."""
    ids = np.asarray(list(prompt_ids) + list(target_ids), dtype=np.int32)[None]
    x = mx.array(ids)
    state = model.init_state(1)
    logps = []
    n_p = len(prompt_ids)
    for s in range(0, x.shape[1] - 1, chunk):
        e = min(s + chunk, x.shape[1] - 1)
        logits, state = model(x[:, s:e], state)
        lp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
        tgt = x[:, s + 1:e + 1]
        tok_lp = mx.take_along_axis(lp, tgt[..., None], axis=-1)[0, :, 0]
        mx.eval(tok_lp, state)
        logps.append(np.array(tok_lp))
    lp = np.concatenate(logps)
    return float(lp[n_p - 1:].sum())
