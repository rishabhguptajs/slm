# Survey: efficient sequence-model families (as of Sept 2026)

The goal is to choose ingredients for a model that has constant inference memory, low energy per token, and strong exact recall (tool arguments, names, IDs). Each family below gets its mechanism, its strengths, and its **known weaknesses**, with special attention to exact recall and copying.

## 1. State space models: Mamba-2, Mamba-3
- **Mechanism.** Matrix state per head, updated as `S_t = a_t S_{t-1} + k_t v_t^T` with a scalar data-dependent decay (Mamba-2 / SSD). Training uses chunkwise-parallel matmuls; inference is O(1) per token.
- **Mamba-3** (Lahoti, …, Dao, Gu; ICLR 2026, [arXiv 2603.15569](https://arxiv.org/abs/2603.15569)) adds three things: trapezoidal discretization; a MIMO formulation, which raises arithmetic intensity so decode is less memory-bound; and complex-valued states, shown equivalent to a *data-dependent RoPE*, which restores state-tracking (parity, modular arithmetic) that real-valued diagonal SSMs can't do.
- **Strengths.** Simplest recurrence. Decode touches only weights plus a small state. Hardware-friendly.
- **Weaknesses.**
  - *Dense writes*: every token perturbs the whole state, so specific past items interfere and exact recall degrades with distance and with the number of stored items. This is the MQAR / "Zoology" gap versus attention.
  - Real-valued diagonal decay can't state-track (Mamba-3 fixes this with complex states).

## 2. Gated linear recurrences with delta rule: DeltaNet, Gated DeltaNet, RWKV-7, GDN-2
- **Mechanism.** `S_t = a_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`. The delta rule *replaces* the value stored at key k instead of adding to it, so the state behaves like an error-correcting associative memory. Gating (a_t) gives fast forgetting.
  - RWKV-7 "Goose" ([2503.14456](https://arxiv.org/abs/2503.14456)) uses a generalized delta rule with vector-valued gates and in-context learning rates.
  - Gated DeltaNet-2 ([2605.22791](https://arxiv.org/abs/2605.22791)) and Erase-then-Delta Attention ([2606.26560](https://arxiv.org/abs/2606.26560)) decouple the *erase* address from the *write* address. They report the best recall among recurrent models at 1.3B, with small overhead.
  - FG²-GDN ([2604.19021](https://arxiv.org/abs/2604.19021)) adds finer-grained gating for long context.
- **Strengths.** The best exact-recall family among pure fixed-state models. Chunkwise-parallel training via the WY/UT transform. The delta rule can do some state tracking once eigenvalues are allowed to go negative.
- **Weaknesses.** Capacity is still bounded by dk×dv per head: once more distinct key–value pairs arrive than the state can hold, recall collapses. It's more expensive than SSD (a triangular solve per chunk, ~25% more projection parameters). Pure-recurrent models still lose to attention on long-context retrieval (RULER-style).

## 3. xLSTM (mLSTM)
- **Mechanism.** Matrix-memory LSTM with exponential input gates and a normalizer state. It's essentially gated linear attention with normalization.
- **Weaknesses.** The same dense-write recall limits as family 1.

## 4. Short-convolution hybrids: Liquid LFM2
- **Mechanism.** Mostly *gated short convolutions* (`y = C ⊙ conv_k(B ⊙ x)`, k≈3) plus a minority of grouped-query attention layers (roughly 10 conv + 6 GQA in LFM2).
- **Strengths.** Short convs are extremely cheap and fast on CPUs and phones: tiny state, pure local ops. This is the most device-oriented design in production.
- **Weaknesses.** The conv layers provide no long-range memory; *all* recall comes from the attention layers, whose KV cache grows. That breaks our constant-memory requirement unless those layers are windowed or bounded.

## 5. Hybrids and what attention is for
- Hybrid analysis ([2507.06457](https://arxiv.org/abs/2507.06457)): recall rises steadily with the fraction of full-attention layers. DeltaNet/GDN hybrids peak around 3:1 linear:attention.
- "What Attention Recalls and Recurrence Controls" ([2609.04434](https://arxiv.org/abs/2609.04434)) studies Qwen3.5 (3:1 GDN:attention) and Falcon-H1. The split is functional:
  - Attention handles **addressable exact retrieval**. Exact KV retrieval drops to ~0 when attention is ablated.
  - Recurrent state handles **mode, persona, language and style**. Recurrent-only models produce "false familiars": feature-level rather than item-level memory.
  - *This is the strongest argument that tool use (copying exact names and arguments) needs some token-level exact memory.*
- "Short window attention enables long-term memorization" ([2509.24552](https://arxiv.org/abs/2509.24552)): SWA + xLSTM hybrids. Short windows force the recurrent layers to learn long-term memory.

## 6. Fixed-slot and routed memories (closest to our proposal)
- **Raven / Routing Slot Memories** (Afzal, Bick, Xing, Cevher, Gu; [2607.25357](https://arxiv.org/abs/2607.25357)):
  - It frames SSMs as dense routing (write every slot) and SWA as FIFO routing (write one slot in round-robin order). Raven uses a learned *sparse top-k router* that writes and decays only selected slots.
  - It reports perfect NIAH where SWA fails and extrapolates to 16× training length.
  - Slots are linear-recurrence rows (read `o = S q`), not stored tokens.
- **Learned KV eviction.**
  - KVP ([2602.10238](https://arxiv.org/abs/2602.10238)) trains per-head RL scoring agents *post hoc* on a frozen LLM.
  - LKV ([2605.06676](https://arxiv.org/abs/2605.06676)) learns head-wise budgets and token selection.
  - IndexMem ([2605.25475](https://arxiv.org/abs/2605.25475)) explicitly lists "native eviction learned jointly in pretraining" as open future work.
  - Heuristic predecessors: H2O, TOVA, StreamingLLM (attention sinks).
- **Weakness of the eviction line.** It's retrofitted onto transformers, trained with a proxy objective, and the train-time model never experiences eviction, so there is a train/inference mismatch.

## 7. Looped / recursive models with adaptive computation
- Universal Transformers with ACT halting; TRM (tiny recursive model, 2025); Ouro/LoopLM (2025).
- 2026 work:
  - Diagnosing learned halting gates ([2607.20519](https://arxiv.org/abs/2607.20519)).
  - Continuous depth batching for serving looped models ([2608.09444](https://arxiv.org/abs/2608.09444)).
  - "Looped LMs improve compositional tool calling" ([2608.18171](https://arxiv.org/abs/2608.18171)).
- **Strengths.** Weight sharing gives more effective depth per parameter, which matters for tiny devices where *weights* dominate memory. Adaptive halting spends compute only on hard tokens.
- **Weaknesses.**
  - Compute per token rises with loop count, which is the opposite of our energy goal unless halting is aggressive.
  - Halting gates are hard to train (the 2607.20519 diagnosis shows they often collapse to fixed depth).
  - Batching variable depth is awkward on servers.

## 8. Ternary / 1.58-bit weights: BitNet b1.58 and successors
- Weights in {−1, 0, +1} with per-tensor or per-layer scales, trained from scratch with straight-through estimators.
  - BitNet b1.58 2B4T matches FP16 at 2B.
  - TernaryLM ([2602.07374](https://arxiv.org/abs/2602.07374)) at 132M.
  - "Breaking the 1.58-bit barrier" ([2609.16338](https://arxiv.org/abs/2609.16338)).
  - Ternary ROM accelerators for edge ([2602.20662](https://arxiv.org/abs/2602.20662)).
- **Strengths.** About 10× smaller weights than fp16 (plus packing). Matmuls become add/subtract, a large energy win on MCUs and ASICs. Decode is memory-bound, so fewer weight bytes means faster and cheaper decode.
- **Weaknesses.** Quality loss grows at *small* scale (the published parity claims are at ≥1B; tiny models lose more). Training needs latent fp weights, so training memory isn't reduced. Activations still need 8-bit.

## 9. Test-time training and fast-weight memory: TTT, Titans, LaCT
- **Mechanism.** The memory is a small network (or matrix) whose weights take gradient steps on a self-supervised loss during inference.
  - Titans (NeurIPS) uses surprise-weighted updates with momentum and forgetting.
  - LaCT uses large-chunk updates plus window attention.
  - "TTT with KV binding is secretly linear attention" ([2602.21204](https://arxiv.org/abs/2602.21204)): one-step TTT with a linear memory reduces to (delta-rule-like) linear attention.
- **Strengths.** Richer (nonlinear) memory with the same constant-memory property. It also enables *persistent* adaptation to a user, which is the side track.
- **Weaknesses.** Expensive (inner-loop gradients), awkward to parallelize, and its gains over GDN at small scale are unclear. Kept to `sidetrack/plasticity/`.

---

## Synthesis: what the literature says about our goals
1. Constant memory plus good *language modelling* is solved well enough: GDN/Mamba-class models match transformers in perplexity at equal size.
2. Constant memory plus *exact recall* is the open problem. It's exactly what tool calling needs (copy a function name or an argument verbatim from 2k tokens back).
3. Current fixes:
   - (a) Keep a few full-attention layers. The memory grows, which violates our requirement.
   - (b) Sliding windows. Exact, but forgets anything outside the window.
   - (c) Routed linear slots (Raven). Not stored tokens.
   - (d) Post-hoc eviction on transformers. Train/inference mismatch.
4. For tiny devices, weight bytes and memory bandwidth matter more than FLOPs. That favors ternary weights, weight sharing, and small fixed state, and it disfavors big embedding tables and KV caches.

## Candidate architectures for the bake-off (all ≈14M params, `mbpe16k`, same data and tokens)

| id | pattern | inference memory | hypothesis |
|---|---|---|---|
| **A** | 12×Attention (RoPE, SwiGLU) | grows O(T) | The bar. Best exact recall within the training context; KV cache and quadratic prefill are the costs. |
| **G** | 12×GLA/SSD (Mamba-2-style scalar decay) | constant | Matches A on bpb at a fraction of decode memory; weak on recall. |
| **D** | 12×Gated DeltaNet | constant | Better recall than G at similar bpb (the delta rule reduces interference). Still fails beyond state capacity. |
| **DW** | 3×(D,D,D,SWA-128) | constant (window 128) | Local exact attention improves bpb and short-range copying; can't recall beyond 128 tokens. |
| **DP (ours)** | 3×(D,D,D,PMA-64) | constant (64 slots/head) | *Priority Memory Attention.* Learned salience plus multi-scale recency decides which exact tokens survive in a fixed bank. Hypothesis: equal or better bpb than DW (a PMA head with large λ *is* SWA) and much better long-range exact recall than D/DW/G. Memory stays flat, and train-time behaviour equals streaming behaviour exactly. |
| **CP (ours, device variant)** | LFM2-style gated short conv + GDN + PMA | constant | Cheapest per token for CPU/MCU. Tested after the main bake-off if DP wins. |

**What is new in PMA.**
- Existing learned-eviction methods are post-hoc and trained through proxies, and they never train the model *under* eviction.
- Raven routes writes into linear slots.
- PMA makes the eviction policy part of the architecture from step 0. The priority `p_i = s_i + λ_h·i` is time-invariant, so "keep the top-M by priority" is decomposable. Its exact streaming behaviour (a size-M heap) can be computed in parallel at training time with a rank mask: `rank_t(i) = #{j≤t : p_j > p_i} < M`.
- The model is therefore trained under exactly the memory constraint it will run with.
- λ per head spans sliding-window-like heads (λ=1) to salience-only heads (λ=10⁻³), so the model can learn which horizon each head serves.

Planned Phase 3 ingredients:
- Erase/write decoupling (GDN-2 style).
- A learned λ via a differentiable surrogate.
- A GQA-shared slot bank (less memory).
- Ternary weights.
- Adaptive looping of the middle blocks.
