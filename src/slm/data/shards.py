"""Tokenize raw JSONL shards into flat token arrays and serve training batches.

data/tok/<tokenizer>/<source>/{train,val}.bin   uint16 (vocab <= 65535), docs separated by EOT
data/tok/<tokenizer>/token_bytes.npy            UTF-8 byte length of every token id (for bpb)
"""
import json
import os
import sys

import numpy as np

from slm.data.download import read_jsonl_zst
from slm.data.tokenizer import SPECIALS, load_tokenizer
from slm.data.tooldata import read_tool_jsonl

SOURCES = ["en", "de", "fr", "es", "ru", "zh", "ja", "ar", "hi", "sw", "tr", "tool"]


def token_byte_lengths(tok):
    """Bytes each token contributes to the text. Specials count as 1 byte (a separator)."""
    V = tok.vocab_size
    out = np.ones(V, dtype=np.int32)
    if tok.name == "bytes":
        return out
    specials = set(SPECIALS)
    for i in range(V):
        s = tok.tok.id_to_token(i)
        # byte-level BPE: each unicode char of the token string encodes exactly one byte
        out[i] = 1 if s in specials else len(s)
    return out


def iter_docs(src, split):
    if src == "tool":
        for r in read_tool_jsonl(f"data/raw/tool/{split}.jsonl.zst"):
            yield r["text"]
    else:
        yield from read_jsonl_zst(f"data/raw/{src}/{split}.jsonl.zst")


def tokenize_source(tok, src, split, out_dir, max_mb=None, batch=512):
    path = os.path.join(out_dir, src, f"{split}.bin")
    if os.path.exists(path + ".done"):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    n_bytes, n_tok = 0, 0
    tmp = path + ".partial"
    if os.path.exists(tmp):  # interrupted earlier: archive, never overwrite
        arch = os.path.join("archive", "tok_partial")
        os.makedirs(arch, exist_ok=True)
        os.rename(tmp, os.path.join(arch, f"{src}_{split}_{int(os.path.getmtime(tmp))}.bin"))
    with open(tmp, "wb") as f:
        buf = []

        def flush():
            nonlocal n_tok
            for ids in tok.encode_batch(buf):
                arr = np.array(ids + [tok.eot], dtype=dtype)
                f.write(arr.tobytes())
                n_tok += len(arr)
            buf.clear()

        for text in iter_docs(src, split):
            buf.append(text)
            n_bytes += len(text.encode("utf-8"))
            if len(buf) >= batch:
                flush()
            if max_mb and n_bytes >= max_mb * 1e6:
                break
        if buf:
            flush()
    os.rename(tmp, path)
    with open(path + ".done", "w") as f:
        json.dump({"bytes": n_bytes, "tokens": n_tok}, f)
    print(f"{src}/{split}: {n_bytes/1e6:.0f}MB -> {n_tok/1e6:.1f}M tokens", flush=True)


def build(tok_name, root="data/tok"):
    tok = load_tokenizer(tok_name)
    out = os.path.join(root, tok_name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "token_bytes.npy"), token_byte_lengths(tok))
    for src in SOURCES:
        for split in ("val", "train"):
            tokenize_source(tok, src, split, out)


class MixtureLoader:
    """Samples [B, T+1] windows from sources according to mixture weights.
    Deterministic given seed; each row picks a source, then a uniform offset."""

    def __init__(self, tok_name, weights, B, T, seed=0, split="train", root="data/tok"):
        self.srcs = list(weights)
        w = np.array([weights[s] for s in self.srcs], dtype=np.float64)
        self.p = w / w.sum()
        self.data = {}
        for s in self.srcs:
            p = os.path.join(root, tok_name, s, f"{split}.bin")
            dt = np.uint16 if os.path.getsize(p) and _is_u16(tok_name, root) else np.uint32
            self.data[s] = np.memmap(p, dtype=dt, mode="r")
        self.B, self.T = B, T
        self.rng = np.random.default_rng(seed)

    def next(self):
        src_idx = self.rng.choice(len(self.srcs), size=self.B, p=self.p)
        rows = np.empty((self.B, self.T + 1), dtype=np.int32)
        for i, si in enumerate(src_idx):
            d = self.data[self.srcs[si]]
            o = self.rng.integers(0, len(d) - self.T - 1)
            rows[i] = d[o:o + self.T + 1]
        return rows


def _is_u16(tok_name, root):
    tb = np.load(os.path.join(root, tok_name, "token_bytes.npy"))
    return len(tb) <= 65535


def load_val(tok_name, src, max_tokens=None, root="data/tok"):
    p = os.path.join(root, tok_name, src, "val.bin")
    dt = np.uint16 if _is_u16(tok_name, root) else np.uint32
    d = np.memmap(p, dtype=dt, mode="r")
    return np.array(d[:max_tokens] if max_tokens else d, dtype=np.int32)


if __name__ == "__main__":
    build(sys.argv[1])
