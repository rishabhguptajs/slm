"""Needle-in-a-stream: plant a fact, stream N filler tokens, then query it.

  prompt = NEEDLE(name, code) + filler[N tokens of held-out English] + QUERY(name)
  target = " d d d d d." (5 random digits; our tokenizer splits digits individually)

Metrics per N:
  exact   : greedy decode reproduces all 5 digits
  gain    : log2 p(target | needle ... query) - log2 p(target | decoy needle ... query), in bits.
            The decoy needle has the same name but a different random code, so the gain
            isolates "used the right fact" from "knows the format". Max ~16.6 bits.
"""
import numpy as np

from slm.data.shards import load_val
from slm.eval.generate import continuation_logprob, greedy

NAMES = ["Alvarez", "Okafor", "Lindqvist", "Tanaka", "Moreau", "Kowalski", "Haddad", "Ivanova",
         "Mensah", "Novak", "Castillo", "Bergstrom", "Nakamura", "Rossi", "Adeyemi", "Petrov"]


def _needle(name, code):
    return f"Note: the secret code for {name} is {code}.\n"


def _query(name):
    return f"\nQuestion: what is the secret code for {name}?\nAnswer: the secret code for {name} is"


def run_needle(model, tok, lengths=(0, 128, 512, 2048, 8192), trials=16, seed=0, filler_src="en"):
    rng = np.random.default_rng(seed)
    filler = load_val(tok.name if tok.name != "bytes" else "bytes", filler_src)
    results = {}
    for N in lengths:
        exact, gains = 0, []
        for t in range(trials):
            name, decoy = rng.choice(NAMES, 2, replace=False)
            code = "".join(str(d) for d in rng.integers(0, 10, 5))
            dcode = "".join(str(d) for d in rng.integers(0, 10, 5))
            off = int(rng.integers(0, len(filler) - N - 1))
            fill = list(filler[off:off + N])
            q = tok.encode(_query(name))
            tgt = tok.encode(" " + code)
            p_true = tok.encode(_needle(name, code)) + fill + q
            p_decoy = tok.encode(_needle(name, dcode)) + fill + q
            lp_t = continuation_logprob(model, p_true, tgt)
            lp_d = continuation_logprob(model, p_decoy, tgt)
            gains.append((lp_t - lp_d) / np.log(2))
            out = greedy(model, p_true, max_new=len(tgt))
            exact += int(out == tgt)
        results[N] = dict(exact=exact / trials, gain_bits=float(np.mean(gains)),
                          gain_bits_se=float(np.std(gains) / np.sqrt(trials)))
        print(f"  needle N={N:6d}: exact {exact}/{trials}  gain {np.mean(gains):.2f} bits", flush=True)
    return results
