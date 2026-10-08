# EXPERIMENTS

Every entry has a hypothesis, change, result, and verdict. Numbers come from `results/<run>/metrics.json`.
Shared setup unless stated otherwise:
- Tokenizer `mbpe16k`.
- Mixture: en 35%, 10 other languages at 5.5% each, tool calls 10%.
- B=16, T=512 (8,192 tokens/step).
- AdamW (β2 0.95, wd 0.1 on matrices), warmup then cosine to 10%, grad clip 1.0, seed 0.
- fp32 on an M4 MacBook Air (16 GB).
- Validation: 131k tokens per source, bpb = bits / UTF-8 bytes. `all_text` is byte-weighted over the 11 languages; `macro` is the unweighted mean over languages.

---

## E0. Infrastructure and correctness (2026-09-29)
**Hypothesis:** the chunked-parallel training forms and step-by-step streaming forms of every mixer compute the same function.
**Result:** 21 tests pass.
- Chunked GDN and GLA match token-by-token recurrences (max error < 1e-3), including the worst case of identical keys with β=1.
- Chunk-by-chunk streaming equals one-shot forward for all six mixer types and for hybrids.
- PMA memory contents equal a brute-force top-M heap.
**Found along the way:** an MLX 0.32.2 out-of-bounds write (`cumsum` on a non-last axis of a sliced view) that silently corrupted weights. It's worked around, with a regression test (`docs/notes_mlx_bugs.md`).
**Verdict:** the train-time math matches inference.

## E1. Tokenizer study (2026-09-29)
See `reports/tokenizer.md`. The GPT-2 regex splits Hindi words at every vowel sign; the multilingual regex cuts Hindi tokens by 37% with no loss elsewhere. **Chosen: `mbpe16k`.**

## E2. Training-throughput microbenchmarks (2026-09-29, before optimizations)
Random tokens, d=256, 12 layers, B16×T512, fp32, compiled step. Numbers are indicative only (the machine also ran data jobs).

| pattern | params | tok/s | peak GB |
|---|---|---|---|
| A (transformer) | 13.83M | 10,506 | 5.8 |
| A bf16 | 13.83M | 13,267 | 4.5 |
| D (Gated DeltaNet) | 14.69M | 5,453 | 8.4 |
| G (GLA) | 14.68M | 6,786 | 7.2 |
| DDDP | 14.48M | 5,844 | 7.8 |
| DDDW | 14.48M | 5,717 | 7.7 |
| C (short conv) | 13.85M | 10,903 | 5.3 |

Profiling one mixer (fwd+bwd, B16×T512): attention 23 ms, GDN 54 ms, GLA 39 ms. The recurrence core itself is only 8.6 ms (GLA) and 24 ms (GDN). The rest is projections (GDN has 5d² vs 4d²), the short conv, and non-fused elementwise ops. Changes made: GLA is now fully parallel across chunks, GDN splits by chunk instead of indexing, and the conv uses native depthwise `conv1d`.
**Verdict:** recurrent layers currently train at ~0.5–0.65× transformer speed in wall-clock on MLX. This is an implementation gap (no fused kernels), not a FLOP gap. The bake-off will report both FLOPs and wall-clock.

---

## E3. Transformer baseline (A), 13.83M params (2026-10-05)
**Hypothesis:** a tuned deep-and-thin transformer (12 layers, d=256, RoPE, SwiGLU, RMSNorm, tied embeddings) is the bar the candidates must beat.
**LR sweep** (1,500 steps = 12M tokens each, bpb on all text, lower is better): 1e-3 → 2.164, **2e-3 → 2.115**, 4e-3 → 2.151. Best LR 2e-3 sits inside the grid, so no extension was needed. (The 4e-3 run's wall-clock and tok/s are invalid because the Mac slept during it; its bpb is unaffected.)
**Full run** (`results/A_transformer_full`): 6,000 steps, 49.2M tokens, 85.5 min, 9,578 tok/s, peak 5.89 GB.

| metric | bpb |
|---|---|
| all text (byte-weighted) | **1.642** |
| macro mean over languages | 1.728 |
| tool-call text | 0.840 |
| en / de / fr / es | 1.572 / 2.003 / 1.876 / 1.858 |
| ru / zh / ja / ar | 1.296 / 2.484 / 1.659 / 1.445 |
| hi / sw / tr | 0.881 / 1.923 / 2.007 |

Per-language bpb is not comparable across languages (different text, different information density per byte); it is only for comparing models on the same language.
**Verdict:** baseline established. Not yet run: needle-in-a-stream, tool-call accuracy, and efficiency suites (queued for Stage 6). The eval suite for this run has not been executed.

## E4. Priority Memory hybrid (DP = 3×[D,D,D,P]) LR sweep, 13.82M params (2026-10-06)
**Hypothesis:** at equal params and tokens, GDN + Priority Memory Attention (64 slots/head, NoPE, multi-scale recency rates 1/0.1/0.01/0.001) matches or beats the transformer on bpb while keeping fixed inference memory.
**LR sweep** (1,500 steps = 12M tokens, same protocol as E3):

| LR | DP all-text bpb | A (transformer) bpb | DP tool bpb | A tool bpb |
|---|---|---|---|---|
| 1e-3 | 2.064 | 2.164 | 1.310 | 1.460 |
| 2e-3 | 2.040 | 2.115 | 1.282 | 1.453 |
| 4e-3 | **2.012** | 2.151 | 1.286 | 1.527 |
| 8e-3 | 2.024 | – | 1.309 | – |

Best vs best at 12M tokens: DP 2.012 vs A 2.115 (−0.103 bpb, −4.9%). Throughput 6.8–7.5k tok/s (vs A ~9.2–9.9k), peak 7.62 GB.
The 8e-3 check (rerun on 2026-10-06 after a first attempt was stopped at step 150 and archived) scored 2.024, worse than 4e-3, so 4e-3 is an interior optimum. Its training loss trailed 4e-3 at every checkpoint (step 500: 5.705 vs 5.618; step 1,250: 4.860 vs 4.800). Single seed; short runs.
**Verdict:** a promising early signal; the full run decides. The full run uses LR 4e-3.

## E5. Priority Memory hybrid (DP) full run vs transformer, 13.82M params (2026-10-06)
**Hypothesis:** at equal params (13.82M vs 13.83M) and tokens (49.2M), DP beats the tuned transformer on bpb.
**Change:** `results/DP_gdn_pma_full`, LR 4e-3 (best from E4), 6,000 steps, warmup 300, same data order, seed and eval protocol as E3.
**Result** (bpb, lower is better):

| metric | DP | A (E3) | Δ |
|---|---|---|---|
| all text | **1.6192** | 1.6422 | −0.023 (−1.4%) |
| macro over languages | **1.7033** | 1.7277 | −1.4% |
| tool-call text | **0.8059** | 0.8399 | −4.0% |
| en / de / fr / es | 1.541 / 1.967 / 1.844 / 1.830 | 1.572 / 2.003 / 1.876 / 1.858 | −2.0 / −1.8 / −1.7 / −1.5% |
| ru / zh / ja / ar | 1.264 / 2.459 / 1.641 / 1.437 | 1.296 / 2.484 / 1.659 / 1.445 | −2.5 / −1.0 / −1.1 / −0.6% |
| hi / sw / tr | 0.874 / 1.892 / 1.990 | 0.881 / 1.923 / 2.007 | −0.8 / −1.6 / −0.9% |

- **Intermediate evals (all-text bpb):** step 2,000: 1.943 vs 1.994; step 4,000: 1.715 vs 1.726.
- **Training-loss gap (A − DP):** peaked at ~0.25 (steps 450–800), shrank to ~0.05 by step 3,300, then held at ~0.045 through step 4,400.
- **Cost:** 114.4 min, 7,162 tok/s, peak 7.62 GB (vs A: 85.5 min, 9,578 tok/s, 5.89 GB). The run was mostly with the Mac kept awake; speed varied between 5.5k and 8.7k tok/s (thermal / other use).

**Verdict:** DP beats the transformer on all 11 languages and on tool text, but the margin is small (1.4%). Most of the large early lead came from faster early learning and faded as training went on. Single seed, so a ~1% gap is a positive result but not yet a robust one. Still to check: whether the PMA layer itself is responsible (D and DW ablations), and the inference claims (memory, speed by length, recall, energy).

## E6. Ability tests: needle recall, tool calls, inference efficiency (A vs DP full runs, 2026-10-06)
**Hypothesis:** DP keeps a constant inference state and constant per-token decode cost on long streams, while A's KV cache and per-token cost grow with length. DP is expected to be weaker on exact recall.
**Change:** `scripts/eval_model.py` (full settings) on `A_transformer_full` and `DP_gdn_pma_full`. Batch 1, fp32, M4.
**Bug found and fixed first:** `unit_lower_inverse` crashed on odd chunk sizes (short final chunks of 17–63 tokens at inference). The fix only touches odd sizes, so training (all 64-token chunks) is unaffected. Regression tests were added (23 pass). The crashed log is archived in `archive/failed_evals/`.

**Results.**

Needle recall (16 trials per length; gain = log2 p(true code) − log2 p(decoy code)):

| filler tokens | A exact | A gain | DP exact | DP gain |
|---|---|---|---|---|
| 0 | 3/16 | 10.59 | 0/16 | 6.29 |
| 128 | 2/16 | 8.50 | 0/16 | 4.14 |
| 512 | 0/16 | 1.90 | 0/16 | 2.69 |
| 2,048 | 0/16 | 0.03 | 0/16 | 0.13 |
| 8,192 | 0/16 | −0.01 | 0/16 | 0.00 |

Tool calls (300 held-out prompts, greedy decoding):

| model | valid JSON | right name | exact args | fully exact |
|---|---|---|---|---|
| A | 84.3% | 79.3% | 38.0% | 37.7% |
| DP | **89.0%** | 79.3% | **44.0%** | **44.0%** |

Decode after streaming L tokens:

| L | A tok/s | A state | A peak mem | DP tok/s | DP state | DP peak mem |
|---|---|---|---|---|---|---|
| 512 | **585** | 12.6 MB | 220 MB | 325 | 1.07 MB | 166 MB |
| 2,048 | **359** | 50.3 MB | 331 MB | 326 | 1.07 MB | 176 MB |
| 8,192 | 124 | 201 MB | 709 MB | **326** | 1.07 MB | 176 MB |
| 32,768 | 30 | 805 MB | 2,475 MB | **327** | 1.07 MB | 176 MB |

- **Proxy FLOPs per token:** A grows from 33.9 to 430 MFLOP; DP stays at 28.7 MFLOP.
- **Prefill (4,096 tokens, 512-token chunks):** A 18,755 tok/s, DP 25,443 tok/s.

**Verdict:** the flat-memory claim is confirmed. DP's state is 1.07 MB at every length, against A's KV cache growing 64× to 805 MB (peak 2.5 GB at 32k). DP decode speed is flat at ~326 tok/s; A is faster below ~2k tokens and 2.8× / 11× slower at 8k / 32k. DP is better on tool calls (+6.3 points fully exact). DP is worse at short-range exact recall (gain 6.3 vs 10.6 bits at distance 0). Neither model recalls beyond its 512-token training length; both are 14M-param models, so recall is weak overall. Single seed, single machine.

## E7. Energy per token, measured with powermetrics (A vs DP, 2026-10-06)
**Hypothesis:** DP uses less energy per token than A, more so as context grows.
**Change:** `sudo scripts/energy.sh --runs A_transformer_full,DP_gdn_pma_full`, run manually (it needs sudo).
- **Sampling:** powermetrics CPU+GPU "Combined Power", every 250 ms.
- **Idle baseline:** 30 s.
- **Decode phase:** 30 s per model, batch 1, streaming one token at a time after a 512-token prefix, so context grows during the phase (A reached ~5.9k tokens, DP ~9.2k).
- **Prefill phase:** 30 s, 512-token chunks, fresh stream every 8k tokens.
- **Energy figure:** above-idle (average power − idle power) / tok/s.

**Result** (idle 1,037 mW):

| model | phase | avg power | tok/s | mJ/token above idle |
|---|---|---|---|---|
| A | decode | 5,268 mW | 179.5 | 23.57 |
| DP | decode | 6,223 mW | 289.3 | **17.92** (−24%) |
| A | prefill | 15,462 mW | 17,422 | 0.828 |
| DP | prefill | 9,157 mW | 25,049 | **0.324** (−61%) |

**Verdict:** DP uses 24% less energy per generated token and 61% less per token read, in this setting. DP draws slightly more power while decoding but produces 1.6× more tokens per second. A's decode cost here averages over contexts of 512 to ~5.9k tokens, where it slows as its cache grows. At very short contexts (<~2k tokens, where A decodes faster per E6), A probably uses less energy per token; that regime was not measured separately. Single 30 s window per phase, on one machine.

## E8 (in progress). Ablation: all-DeltaNet (D = 12×GDN, no PMA), 13.81M params (2026-10-06)
**Hypothesis:** if Priority Memory Attention contributes, DP beats D at equal params and tokens.
**LR sweep so far** (1,500 steps, same protocol as E3/E4):

| LR | D all-text | DP all-text | D tool | DP tool |
|---|---|---|---|---|
| 1e-3 | 2.0643 | 2.0644 | 1.2995 | 1.3103 |
| 2e-3 | **2.0007** | 2.0397 | 1.2518 | 1.2823 |
| 4e-3 | (stopped at step 700; archived) | 2.0115 | – | 1.2856 |

D's best so far (2.001) already beats DP's best (2.012). Per language at their best LRs, D is better on en (1.849 vs 1.897) and tool; DP is better on zh (3.049 vs 3.076) and hi (1.098 vs 1.116). The lr 1e-3 run's wall-clock and tok/s are invalid: the Mac slept from 21:07 to 22:44 when the lid closed.
**Interim verdict:** on 512-token text, PMA adds nothing to bpb; the gains over the transformer come from Gated DeltaNet. Still to do: D lr 4e-3, the D full run, and long-context recall tests (where PMA is meant to matter).
**Resume:** `PYTHONPATH=src python scripts/sweep.py --config configs/D_gdn.json` (it skips finished LRs).

**Update 2026-10-08: sweep finished and full run done.**
- **lr 4e-3 short run:** 2.0076, so the best LR stays 2e-3.
- **Full run** (`results/D_gdn_full`, lr 2e-3, 6,000 steps): 122.5 min, 6,688 tok/s, peak 8.03 GB.

| metric | A | DP | D | DP vs D |
|---|---|---|---|---|
| all text | 1.6422 | **1.6192** | 1.6356 | −1.0% |
| macro | 1.7277 | **1.7033** | 1.7204 | −1.0% |
| tool | 0.8399 | **0.8059** | 0.8331 | −3.3% |
| en / de / fr / es | 1.572 / 2.003 / 1.876 / 1.858 | **1.541 / 1.967 / 1.844 / 1.830** | 1.555 / 1.975 / 1.846 / 1.835 | −0.9 / −0.4 / −0.1 / −0.3% |
| ru / zh / ja / ar | 1.296 / 2.484 / 1.659 / 1.445 | **1.264 / 2.459 / 1.641 / 1.437** | 1.280 / 2.511 / 1.671 / 1.452 | −1.3 / −2.0 / −1.8 / −1.1% |
| hi / sw / tr | 0.881 / 1.923 / 2.007 | **0.874 / 1.892 / 1.990** | 0.881 / 1.911 / 2.008 | −0.8 / −1.0 / −0.9% |

- **Checkpoint evals** (all-text): step 2,000: D 1.916, DP 1.943; step 4,000: D 1.717, DP 1.715.
- **Training loss:** D led until ~step 3,400; DP led after that (step 4,450: 3.907 vs 3.922).

**Verdict (revises the interim verdict above):**
- **The short runs were misleading.** D won the 12M-token sweep, but over 49M tokens DP beats D on every language and on tool text. PMA contributes −1.0% bpb (−3.3% tool).
- **Without PMA, plain GDN barely beats the transformer** (−0.4%), and it loses on zh, ja and ar, and ties on tr and hi.
- **Caveats:** single seed. D used its short-run best LR (2e-3) while DP used 4e-3; D at 4e-3 was close in the short run (2.008 vs 2.001) and was not tried as a full run.

## E9. Ability tests on all-DeltaNet (D_gdn_full) vs DP and A (2026-10-08)
**Hypothesis:** if PMA's slots do their job, DP recalls planted facts better than D. D has the same fixed-state efficiency.
**Change:** `scripts/eval_model.py --run D_gdn_full` (same settings as E6).
**Result:**

| test | A | DP | D |
|---|---|---|---|
| needle gain, 0 filler (bits) | **10.59** | 6.29 | 1.60 |
| needle gain, 128 filler | **8.50** | 4.14 | 1.55 |
| needle gain, 512 filler | 1.90 | **2.69** | 0.62 |
| needle gain, 2,048 / 8,192 | ~0 | ~0 | ~0 |
| needle exact (any length) | 3/16 at most | 0/16 | 0/16 |
| tool: valid JSON | 84.3% | **89.0%** | 88.3% |
| tool: right name | **79.3%** | **79.3%** | 73.7% |
| tool: fully exact | 37.7% | **44.0%** | 37.0% |
| state at any length | grows to 805 MB at 32k | 1.07 MB | **0.90 MB** |
| decode tok/s, 512 → 32k | 585 → 30 | ~326 flat | ~309–317 flat |
| peak memory at 32k | 2,475 MB | 176 MB | **153 MB** |

**Verdict:**
- **PMA is what gives the hybrid its recall.** Without it, the model barely tells the true code from the decoy (1.6 bits vs 6.3 with PMA at distance 0), so PMA is about 4× stronger here. The transformer is still best at short range; DP is best at 512.
- **PMA also drives the tool-call gain** (44% vs 37% fully exact; right name 79% vs 74%).
- **Cost:** +0.17 MB of state, and no measurable decode slowdown (DP was slightly faster in these runs; single short timing windows, within thermal noise).
- **Overall:** with E8, PMA is a net positive on bpb, recall and tool use at a negligible memory cost. Recall beyond the 512-token training length is still zero for all three models.

## E10. Long-context continued training + recall practice (A, DP, D; 2026-10-08)
**Hypothesis:** given long text (T=2048) and recall practice, PMA's slots learn to keep planted facts, so DP's recall advantage over D grows. A, as full attention, also improves within 2,048 tokens but not beyond.
**Change:** `scripts/queue_long.sh`.
- Each full model is continue-trained for 1,500 steps (12.3M tokens) at B=4×T=2048.
- The usual mixture, with 25% of rows replaced by synthetic recall rows (`src/slm/data/recall.py`). Each row has 2–5 facts planted in the first half, with queries later. Templates and names are disjoint from the needle eval.
- Peak LR is 0.5× each model's best (DP 2e-3, D 1e-3, A 1e-3), warmup 100, cosine to 10%. Same data order for all.
- Then needle tests (16 trials, ±SE) on the new and original models, with fillers 0–8,192.

**Result: bpb** (eval at T=2048, so not directly comparable with the T=512 numbers in E5/E8):

| model | all text | macro | tool | train time | tok/s | peak |
|---|---|---|---|---|---|---|
| DP long | **1.6167** | **1.7002** | **0.7923** | 50.1 min | 4,086 | 8.68 GB |
| A long | 1.6188 | 1.7033 | 0.8078 | 35.2 min | 5,825 | 8.59 GB |
| D long | 1.6288 | 1.7130 | 0.8127 | 39.5 min | 5,179 | 8.09 GB |

**Result: needle gain in bits** (true vs decoy code; mean ± SE):

| filler | DP orig | DP long | D orig | D long | A orig | A long |
|---|---|---|---|---|---|---|
| 0 | 6.29±0.67 | 3.62±0.40 | 1.60±0.29 | 1.03±0.16 | 10.59±0.84 | **15.88±0.78** (7/16 exact) |
| 512 | 1.51±0.45 | 1.16±0.19 | 0.53±0.13 | 0.33±0.08 | 1.49±0.24 | **9.86±1.02** (3/16 exact) |
| 1,024 | 0.91±0.16 | 0.49±0.09 | 0.27±0.07 | 0.17±0.04 | 0.16±0.12 | **4.61±0.55** |
| 1,800 | 0.17±0.09 | 0.07±0.04 | 0.04±0.02 | 0.03±0.02 | 0.03±0.05 | **1.68±0.32** |
| 4,096 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.09±0.11 |
| 8,192 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.12±0.06 |

**Tool calls, long models** (300 prompts):

| model | valid JSON | right name | fully exact |
|---|---|---|---|
| DP long | 83.3% | 75.7% | **42.3%** |
| D long | 90.0% | 81.0% | 37.7% |
| A long | 83.3% | 80.7% | 38.3% |

These are down from 89/79/44 for DP and 84/79/38 for A in E6.

The DP-orig needle numbers at 512 differ from E6 (1.51 vs 2.69) because the trial draws change with the length list. E6's ±SE was 0.37, so the earlier 512-token edge of DP over A (2.69 vs 1.90) is not robust.

**Verdict: a clear negative result for PMA's recall in its current form.**
- Long-context training made the transformer's recall far better: +5.3 bits at distance 0, and 9.9 bits at 512 (from 1.5). It now recalls at 1,024–1,800.
- It made both recurrent models' recall worse at every distance. DP still beats D at every distance (≈3–3.5× at short range), but the gap to A grew.
- Beyond the 2,048 training length, all models are ≈0.
- On bpb, DP still edges A (−0.1%) and beats D (−0.7%).

**Likely causes (hypotheses, untested):**
1. **PMA's keep/evict decision has no direct gradient.** Salience is only trained through the read bias, so the model can't learn "keep this fact because it's asked about later".
2. **Fixed recency rates.** λ = 1, 0.1, 0.01 make 3 of the 4 heads effectively recency windows over 1,000+ tokens; only the λ = 0.001 head can hold a fact that long.
3. **Too little recall practice:** ≈1 recall row per step, for 1,500 steps.
4. **Restarting at 5× the final LR** disrupted the already-annealed recall circuits (all models had this, and only A recovered).

**Next:** fix the mechanism (learned retention that is trained by whether a slot gets read later; learnable per-head rates), or design anew (docs/new_architecture_prompt_v3.md).
