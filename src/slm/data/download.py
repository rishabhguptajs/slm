"""Stream text corpora from the HF hub into capped, zstd-compressed JSONL shards.

Each source writes data/raw/<lang>/{train,val}.jsonl.zst. The first `val_bytes`
of text go to val (held out, never trained on), then up to `cap_bytes` to train.
Streaming means we never download full datasets; the cap bounds disk use.
"""
import argparse
import json
import os
import sys
import time

import zstandard as zstd

# lang code -> (hf dataset, config, text field)
SOURCES = {
    "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "de": ("HuggingFaceFW/fineweb-2", "deu_Latn", "text"),
    "fr": ("HuggingFaceFW/fineweb-2", "fra_Latn", "text"),
    "es": ("HuggingFaceFW/fineweb-2", "spa_Latn", "text"),
    "ru": ("HuggingFaceFW/fineweb-2", "rus_Cyrl", "text"),
    "zh": ("HuggingFaceFW/fineweb-2", "cmn_Hani", "text"),
    "ja": ("HuggingFaceFW/fineweb-2", "jpn_Jpan", "text"),
    "ar": ("HuggingFaceFW/fineweb-2", "arb_Arab", "text"),
    "hi": ("HuggingFaceFW/fineweb-2", "hin_Deva", "text"),
    "sw": ("HuggingFaceFW/fineweb-2", "swh_Latn", "text"),
    "tr": ("HuggingFaceFW/fineweb-2", "tur_Latn", "text"),
}


class ZstdJsonlWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "wb")
        self.w = zstd.ZstdCompressor(level=6).stream_writer(self.f)
        self.bytes = 0
        self.docs = 0

    def write(self, text):
        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
        self.w.write(line.encode("utf-8"))
        self.bytes += len(text.encode("utf-8"))
        self.docs += 1

    def close(self):
        self.w.flush(zstd.FLUSH_FRAME)
        self.w.close()


def read_jsonl_zst(path):
    with open(path, "rb") as f:
        r = zstd.ZstdDecompressor().stream_reader(f)
        buf = b""
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if ln:
                    yield json.loads(ln)["text"]
        if buf.strip():
            yield json.loads(buf)["text"]


def download(lang, out_dir, cap_bytes, val_bytes, min_chars=200):
    from datasets import load_dataset

    name, cfg, field = SOURCES[lang]
    done_flag = os.path.join(out_dir, lang, "DONE.json")
    if os.path.exists(done_flag):
        print(f"[{lang}] already done", flush=True)
        return
    ds = load_dataset(name, cfg, split="train", streaming=True)
    val = ZstdJsonlWriter(os.path.join(out_dir, lang, "val.jsonl.zst"))
    train = ZstdJsonlWriter(os.path.join(out_dir, lang, "train.jsonl.zst"))
    t0 = time.time()
    for ex in ds:
        text = ex[field]
        if not text or len(text) < min_chars:
            continue
        if val.bytes < val_bytes:
            val.write(text)
        else:
            train.write(text)
            if train.docs % 5000 == 0:
                print(f"[{lang}] {train.bytes/1e6:.0f}MB {time.time()-t0:.0f}s", flush=True)
            if train.bytes >= cap_bytes:
                break
    val.close()
    train.close()
    meta = dict(lang=lang, source=f"{name}/{cfg}", train_bytes=train.bytes, train_docs=train.docs,
                val_bytes=val.bytes, val_docs=val.docs, seconds=round(time.time() - t0, 1))
    with open(done_flag, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{lang}] DONE {meta}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(SOURCES))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--cap_mb", type=float, default=250)
    ap.add_argument("--en_cap_mb", type=float, default=1200)
    ap.add_argument("--val_mb", type=float, default=3)
    a = ap.parse_args()
    for lang in a.langs.split(","):
        cap = (a.en_cap_mb if lang == "en" else a.cap_mb) * 1e6
        try:
            download(lang, a.out, cap, a.val_mb * 1e6)
        except Exception as e:  # keep going with other languages
            print(f"[{lang}] FAILED {type(e).__name__}: {e}", file=sys.stderr, flush=True)
