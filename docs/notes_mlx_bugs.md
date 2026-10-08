# MLX issues found (and workarounds)

## 1. Out-of-bounds write from `mx.cumsum` on a non-last axis of a sliced view (MLX 0.32.2, M4)

**Symptom.** A PMA-only model gave intermittent NaNs, with a fixed seed. Model *weights* that had already been materialized changed after a forward pass.

**Bisection** (the regression test is in the repo):
- Weights changed only after the chunk that concatenated 6 memory slots with 24 new tokens.
- Checkpointing every op in that forward showed that weights first change right after
  `new_part = mx.cumsum(beats[..., Ms:, :], axis=-2)`, where `beats` is an int32 `[2,4,30,30]` array.
- These did *not* cause corruption: replacing fused SDPA with manual softmax, NumPy argsort, or explicit index broadcasting in `take_along_axis`.

**Workaround.** The prefix count is now a lower-triangular matmul, `tril(ones(T,T)) @ beats` in float32 (exact for counts < 2^24). This is also a plainer op for porting.
**Guard.** `tests/test_models.py::test_forward_never_modifies_weights` runs every mixer type for 15 full + streaming passes and requires the weights to be bit-identical afterwards.

Note: `mx.cumsum` along the *last* axis of contiguous arrays (used in the recurrences) passes the same guard test.

## 2. `-inf` additive masks
I replaced `-inf` with a large finite negative in PMA masks as a precaution. This turned out not to be the NaN cause; it's kept because it's harmless and safer for low precision.
