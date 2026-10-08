# Plan: an efficient streaming language model

## Goal (restated)

Build a language model architecture that is not a transformer, or uses only a little attention. Ranked by priority, it must beat a well-tuned small transformer on:

1. **Energy per token** (inference first, then training).
2. **Memory**: inference state is *constant* no matter how long the input stream is. No growing KV cache.
3. **Speed** (prefill and decode tokens/sec, latency).
4. **Training compute.**
5. **Accuracy** at equal parameters and equal training tokens, measured in **bits-per-byte** so tokenizer choice can't flatter a model.

Required capabilities: multilingual, tool calls (exact copying of names and arguments), unbounded streaming with fixed memory, and a modality-agnostic input interface. Deployment targets (MCUs, Raspberry Pi, cheap servers, licensing) mean we prefer simple ops that are easy to quantize and port, and low memory bandwidth.

## Ground rules
- Nothing gets deleted. Superseded files and interrupted runs go to `archive/`.
- Every reported number comes from a logged run in `results/<run>/`. Negative results get reported too.
- Disk stays modest: streamed, capped downloads (under 10 GB of data in total).
- Hardware: M4 MacBook Air, 16 GB, passively cooled. Sustained throughput throttles, so speed is measured after warm-up.

## Initial technical choices

**Stack.** Python 3.12 venv, MLX 0.32. Model code is plain modules (`src/slm/models/`) with no MLX-only tricks in the maths, so a PyTorch port is mechanical. Recurrences get two code paths: a *chunked parallel* form for training and a *recurrent step* form for streaming inference. A test checks that the two agree numerically.

**Data** (streamed and capped, stored as uint16/uint32 token shards):
- English: FineWeb-Edu (sample-10BT).
- Multilingual: FineWeb-2 subsets covering diverse scripts and resource levels: de, fr, es, ru, zh (cmn_Hani), ja, ar, hi, sw, plus possibly tr/vi.
- Tool use: an open function-calling dataset (glaive-function-calling-v2 or similar), converted to one fixed call format. A held-out split is used for the tool-call eval.
- Held-out validation shards per language, never trained on.

**Tokenizer.** I compare byte-level (256), byte-level BPE at 8k/16k/32k trained on a language-balanced sample, and hashed n-gram embeddings. I report fertility (tokens/word, or tokens/char for zh/ja) and bytes/token per language. My prior is a ~16k BPE: small models can't afford a large embedding table, and bytes make sequences 3–4× longer. All models are compared in bpb.

**Training recipe (shared by all models).** AdamW with β=(0.9, 0.95), wd 0.1, warmup plus cosine (or WSD) schedule, grad clip 1.0, fixed seeds, identical data order. A quick LR sweep (3–4 values) per architecture on a short budget, then a full run at the best LR.

**Baseline (Phase 0).** Pre-norm decoder with RoPE, SwiGLU, RMSNorm, and tied embeddings, deep and thin, ~10–20M params.

**Candidate families (Phase 1–2)** (details and hypotheses go in `docs/survey.md`):
- **A. Transformer baseline** (full causal attention, growing KV cache).
- **B. Gated linear attention / Mamba-2-style SSD**: scalar-decay matrix state, plus short conv and SwiGLU. Fixed state.
- **C. Gated DeltaNet**: delta-rule (error-correcting) state update with gating, plus short conv. Fixed state; the strongest known recall among fixed-state models.
- **D. Hybrid with sliding-window attention**: mostly C layers plus a few local-window attention layers. Fixed memory (window W).
- **E. Our proposal: Priority Memory hybrid.** C layers plus a few *Priority Memory Attention* (PMA) layers. Each PMA head keeps a fixed bank of **M exact key/value slots**. Each token gets a learned salience score s_i. The slot bank at time t is the top-M tokens by priority p_i = s_i + λ·i, where λ is learned per head. The priority is time-invariant, so the exact streaming eviction policy (a size-M heap) can be reproduced *in parallel at training time* with a rank mask. Training therefore matches inference exactly, and memory is O(M) forever. With λ large the layer becomes sliding-window attention; with λ small it behaves as "keep the most salient tokens". The model learns where on that line to sit, per head. The goal is to restore exact copying (tool args, names, needles) that pure fixed-state models lose, without a growing cache.
- Later (Phase 3): adaptive depth (looped shared blocks with halting), ternary (1.58-bit) weights, learned write/erase gates, and ablations of every component.

**Evals (one harness for every model).**
- Validation bpb, overall and per language.
- Synthetic multi-query associative recall (MQAR) trained from scratch at increasing sequence lengths and numbers of pairs. This is the standard architecture probe.
- Needle-in-a-stream: plant a key→value fact, then stream N filler tokens (N up to 64k+), then query. Report exact-match vs N.
- Tool-call accuracy on held-out examples: exact function name and exact argument JSON (greedy decode).
- Peak memory vs stream length (1k → 64k+ tokens); fixed-state models must be flat.
- Prefill and decode tokens/sec on the M4 (batch 1, and batched decode).
- Energy: `powermetrics` needs sudo, so it runs through a separate script (`sudo scripts/energy.sh`). Until that has been run, efficiency numbers are clearly labelled **proxies**: analytic FLOPs/token, bytes moved/token (weights + state/cache reads), and wall time.

**Compute budget reality.** A 10–20M model on an M4 Air probably trains at roughly tens of thousands of tokens/s. I'll measure this. The bake-off budget per model will be set from that measurement (roughly 100–300M tokens per full run), and LR sweeps will use ~10–20% of the budget. Milestone 2 scale (~100M+ params) will be limited by time. If we can't match public-model token counts, I'll say so plainly.

## Phases and deliverables
| Phase | Deliverable |
|---|---|
| 0 | data pipeline, tokenizer study (`reports/tokenizer.md`), training loop, eval harness, transformer baseline run |
| 1 | `docs/survey.md` (families, weaknesses esp. recall/copy, 2026 work), candidate specs + hypotheses |
| 2 | fair bake-off; single comparison table in `EXPERIMENTS.md` |
| 3 | ablations + new mechanisms (recall fixes, adaptive compute, ternary), one paragraph per experiment |
| 4 | Milestone 1 report, then Milestone 2 (scale + public-model comparison) report, in `reports/` |
| 5 | instruction/tool fine-tune + fixed-memory streaming demo |
| side | `sidetrack/plasticity/`: online fast-weight / TTT adaptation experiments, reported separately |

## Repo layout
```
src/slm/data      download, tokenizer training, sharding, batching
src/slm/models    layers (norms, conv, attention, recurrences, PMA), model assembly, registry
src/slm/eval      bpb, recall, needle, tool-call, efficiency/memory/energy
scripts/          entry points (prepare data, train, eval, bench)
configs/          model + run configs (JSON)
results/<run>/    config.json, log.txt, metrics.json per run
reports/          milestone reports + plots
docs/             survey, design notes
sidetrack/plasticity/
archive/          superseded files (never deleted)
```
