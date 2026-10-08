"""Synthetic long-range recall rows, mixed into ordinary training batches.

A recall row is real filler text (from a training stream) with K key->value facts
planted at random positions and, later in the same row, queries asking for some
of them. The model must carry the right facts across hundreds or thousands of
tokens of unrelated text to predict the answer digits.

Templates and names are deliberately DIFFERENT from the needle eval
(src/slm/eval/needle.py uses "the secret code for <surname>"), so the eval
measures the recall skill, not memorised phrasing.
"""
import numpy as np

from slm.data.tokenizer import load_tokenizer

_SYL = ["ka", "ro", "mi", "tan", "lo", "ve", "su", "dar", "ni", "bel", "ko", "ra", "zu", "fen",
        "ta", "mor", "li", "gan", "pe", "shi", "vo", "ren", "da", "ul"]
_EVAL_NAMES = {"alvarez", "okafor", "lindqvist", "tanaka", "moreau", "kowalski", "haddad", "ivanova",
               "mensah", "novak", "castillo", "bergstrom", "nakamura", "rossi", "adeyemi", "petrov"}

# (fact, query prefix); the answer " <digits>." follows the query prefix
_TEMPLATES = [
    ("{n}'s locker number is {v}.", "\nQ: What is {n}'s locker number?\nA: {n}'s locker number is"),
    ("Remember: the PIN of {n} is {v}.", "\nWhat was the PIN of {n}? The PIN of {n} is"),
    ("[record] {n} -> {v}", "\n[lookup] {n} ->"),
    ("The room assigned to {n} is {v}.", "\nWhich room was assigned to {n}? The room assigned to {n} is"),
    ("{n} parked in spot {v}.", "\nWhere did {n} park? {n} parked in spot"),
]


def _name(rng):
    while True:
        n = "".join(rng.choice(_SYL, int(rng.integers(2, 4)))).capitalize()
        if n.lower() not in _EVAL_NAMES:
            return n


class RecallMixLoader:
    """Wraps a MixtureLoader: each row is replaced by a recall row with prob `frac`."""

    def __init__(self, base, tok_name, frac=0.25, k_range=(2, 6), seed=1):
        self.base, self.frac, self.k_range = base, frac, k_range
        self.tok = load_tokenizer(tok_name)
        self.filler = base.data["en"]
        self.rng = np.random.default_rng(seed)
        self.T = base.T

    def _row(self):
        rng, T, tok = self.rng, self.T, self.tok
        k = int(rng.integers(*self.k_range))
        names = []
        while len(names) < k:
            n = _name(rng)
            if n not in names:
                names.append(n)
        tmpl = [_TEMPLATES[i] for i in rng.integers(0, len(_TEMPLATES), k)]
        vals = ["".join(str(d) for d in rng.integers(0, 10, int(rng.integers(3, 7)))) for _ in range(k)]
        facts = [tok.encode(" " + f.format(n=n, v=v) + " ") for (f, _), n, v in zip(tmpl, names, vals)]
        q_idx = rng.permutation(k)[: int(rng.integers(1, k + 1))]
        queries = [tok.encode(tmpl[i][1].format(n=names[i]) + " " + vals[i] + ".\n") for i in q_idx]
        budget = T + 1 - sum(map(len, facts)) - sum(map(len, queries))
        if budget < 64:  # pathological: fall back to a normal row
            return None
        # facts in the first half of the filler, queries spread after the last fact
        cuts_f = np.sort(rng.integers(0, budget // 2, k))
        last_f = int(cuts_f[-1])
        cuts_q = np.sort(rng.integers(last_f, budget + 1, len(queries)))
        off = int(rng.integers(0, len(self.filler) - budget - 1))
        fill = np.asarray(self.filler[off:off + budget], dtype=np.int64).tolist()
        events = sorted([(int(c), 0, i) for i, c in enumerate(cuts_f)] +
                        [(int(c), 1, i) for i, c in enumerate(cuts_q)])
        out, pos = [], 0
        for c, kind, i in events:
            out += fill[pos:c]
            pos = c
            out += facts[i] if kind == 0 else queries[i]
        out += fill[pos:]
        return np.asarray(out[: T + 1], dtype=np.int32)

    def next(self):
        rows = self.base.next()
        for i in range(rows.shape[0]):
            if self.rng.random() < self.frac:
                r = self._row()
                if r is not None and len(r) == self.T + 1:
                    rows[i] = r
        return rows
