# Side track: online plasticity (models that adapt to their user at inference)

Isolated from the main architecture. Ideas merge back only if they clearly help the main goals.

## Question
Can a small streaming model become measurably better at *this user's* text or tool patterns over time, with **no backprop through the whole model**, fixed memory, and no catastrophic drift?

## Distinction that matters
- **Within-stream memory** (GDN state, PMA slots) already adapts to the context, but it is capacity-limited and overwritten.
- **Persistent plasticity**: a small set of *slow* fast-weights that accumulates across sessions (user vocabulary, names, recurring tool arguments). It's updated online by a local rule and persisted to disk between sessions.

## Candidate mechanisms (cheapest first)
1. **Hebbian / delta-rule persistent memory.** An extra GDN-style matrix per layer with a very small decay (hours/days timescale). It's written with the same delta rule, gated by *surprise* (per-token loss), and never reset between sessions. Cost: one extra d_k×d_v matrix per layer.
2. **Titans-style surprise-gated neural memory.** A 2-layer MLP memory updated by a gradient step on an associative loss ‖M(k) − v‖², with momentum and forgetting. Richer, but it needs per-token inner gradients.
3. **Test-time LoRA on the output head.** A low-rank update of the tied embedding and head from the next-token loss on the user's own stream (true TTT, but still local: only the head and a rank-4 adapter).

## Protocol
- **"User" streams:** held-out documents from one source or topic (e.g. one Wikipedia domain, or tool calls for one API family), split into sessions.
- **Metric:** bpb on session k+1 after adapting on sessions 1..k, compared with the frozen model. Also forgetting: bpb on general val after adaptation.
- **Budget:** plasticity state ≤ 5% of weights, update cost ≤ 20% of decode FLOPs.

Status: planned; runs after the main bake-off (the GPU is the shared bottleneck).
