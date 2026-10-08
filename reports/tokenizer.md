# Tokenizer study (Phase 0)

Source: `results/tokenizer_study/{metrics.json,log.txt}` produced by `scripts/tokenizer_study.py`.

**Setup.** BPE tokenizers were trained on a byte-balanced sample: 25 MB from each of 11 languages plus 20k tool-call samples (~290 MB total). Fertility is measured on held-out val text (1 MB per language, 1,000 tool samples). Two pre-tokenizers were compared:
- `bpe*`: the GPT-2 regex with individually split digits.
- `mbpe*`: my multilingual regex, which keeps Unicode combining marks (`\p{M}`) attached to letters and splits digits individually.

## Bytes per token (higher = better compression)

| tokenizer | en | de | fr | es | ru | zh | ja | ar | hi | sw | tr | tool |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bytes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bpe8k | 3.06 | 2.71 | 2.85 | 2.89 | 3.77 | 2.47 | 3.22 | 3.67 | 3.61 | 2.78 | 2.79 | 3.41 |
| mbpe8k | 3.07 | 2.71 | 2.86 | 2.91 | 3.75 | 2.46 | 3.21 | 3.68 | **5.15** | 2.78 | 2.80 | 3.42 |
| bpe16k | 3.44 | 3.07 | 3.17 | 3.24 | 4.34 | 2.86 | 3.70 | 4.19 | 3.73 | 3.17 | 3.17 | 3.78 |
| **mbpe16k** | 3.46 | 3.08 | 3.18 | 3.25 | 4.34 | 2.85 | 3.68 | 4.20 | **5.94** | 3.17 | 3.17 | 3.80 |
| bpe32k | 3.78 | 3.47 | 3.51 | 3.65 | 4.96 | 3.26 | 4.24 | 4.74 | 3.82 | 3.56 | 3.62 | 4.06 |
| mbpe32k | 3.80 | 3.48 | 3.52 | 3.66 | 4.96 | 3.25 | 4.22 | 4.77 | **6.71** | 3.57 | 3.63 | 4.10 |

## Fertility: tokens per word (zh, ja: tokens per character), GPT-2-regex tokenizers

| tokenizer | en | de | fr | es | ru | zh* | ja* | ar | hi | sw | tr | tool |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bytes | 6.16 | 7.24 | 6.38 | 6.17 | 12.58 | 2.62 | 2.72 | 10.52 | 12.83 | 6.58 | 8.21 | 7.29 |
| bpe8k | 2.01 | 2.67 | 2.24 | 2.13 | 3.34 | 1.06 | 0.85 | 2.86 | 3.55 | 2.36 | 2.94 | 2.14 |
| bpe16k | 1.79 | 2.36 | 2.01 | 1.90 | 2.90 | 0.92 | 0.74 | 2.51 | 3.44 | 2.07 | 2.59 | 1.93 |
| bpe32k | 1.63 | 2.09 | 1.82 | 1.69 | 2.54 | 0.80 | 0.64 | 2.22 | 3.36 | 1.84 | 2.27 | 1.79 |

For the `mbpe` variants, tokens/word = (bytes/word) ÷ (bytes/token). Hindi at 16k drops from 3.44 to about **2.16 tokens/word**; the other languages are within ±1% of the table above.

## Findings
1. **The GPT-2 regex is broken for Indic scripts.** It splits a word at every vowel sign (matra), because matras are `\p{M}`, not `\p{L}`. Hindi then barely benefits from a larger vocabulary (3.61 → 3.73 → 3.82 bytes/token from 8k to 32k). Keeping marks attached cuts Hindi token counts by 37% at 16k and changes no other language. The same issue would affect Bengali, Tamil, Thai, etc., which are likely future customer languages.
2. **Chinese is the least compressed** (≈1 character per token at 16k). This is expected with only 25 MB of zh in the tokenizer sample.
3. **Byte-level** input is 3–6× longer than BPE. At fixed context and compute that is a large cost for a small model. For tiny devices it removes the embedding table (~4M params at 16k×256), but costs ~3.4× more sequential steps per byte. I'm deferring it, and will revisit it as byte-level + hashed n-gram embeddings in Phase 3 if the embedding table turns out to dominate on-device memory.

## Decision
**`mbpe16k`** (16,384 vocab, multilingual regex, individual digits, 10 special tokens for chat and tool roles).
- 8k vs 16k compute per *byte* is roughly equal for a ~10M non-embedding model: the smaller head is offset by 11% more tokens.
- 16k compresses every language better and keeps the embedding at ~30% of a 14M model.
- All model comparisons use bits-per-byte, so tokenizer choice can't flatter any architecture.

Tokenized corpus: 1.12B tokens (en 346M, de 81M, fr 79M, es 78M, ru 58M, zh 95M, ja 70M, ar 60M, hi 42M, sw 77M, tr 80M, tool 56M).
