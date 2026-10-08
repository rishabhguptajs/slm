"""Train candidate tokenizers on a language-balanced sample and measure fertility
per language on held-out val text. Writes results/tokenizer_study/{metrics.json,log.txt}."""
import json
import os
import sys
import time

sys.path.insert(0, "src")
from slm.data.download import read_jsonl_zst  # noqa: E402
from slm.data.tokenizer import BPETokenizer, ByteTokenizer, train_bpe  # noqa: E402
from slm.data.tooldata import read_tool_jsonl  # noqa: E402

LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ar", "hi", "sw", "tr"]
NO_SPACE = {"zh", "ja"}
SAMPLE_MB = 25
VOCABS = [8192, 16384, 32768]
OUT = "results/tokenizer_study"


def sample_texts(lang, mb):
    out, n = [], 0
    for t in read_jsonl_zst(f"data/raw/{lang}/train.jsonl.zst"):
        out.append(t)
        n += len(t.encode("utf-8"))
        if n >= mb * 1e6:
            break
    return out


def val_texts(lang, mb=1.0):
    return sample_texts_path(f"data/raw/{lang}/val.jsonl.zst", mb)


def sample_texts_path(path, mb):
    out, n = [], 0
    for t in read_jsonl_zst(path):
        out.append(t)
        n += len(t.encode("utf-8"))
        if n >= mb * 1e6:
            break
    return out


def fertility(tok, texts, lang):
    ntok = sum(len(ids) for ids in tok.encode_batch(texts))
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    nchar = sum(len(t) for t in texts)
    nword = sum(len(t.split()) for t in texts)
    r = dict(bytes_per_token=nbytes / ntok, tokens_per_char=ntok / nchar)
    r["tokens_per_word"] = None if lang in NO_SPACE else ntok / nword
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(f"{OUT}/log.txt", "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")

    P(f"=== tokenizer study {time.ctime()} sample={SAMPLE_MB}MB/lang")
    corpus = []
    for lang in LANGS:
        corpus += sample_texts(lang, SAMPLE_MB)
    tool = [r["text"] for r in read_tool_jsonl("data/raw/tool/train.jsonl.zst")[:20000]]
    corpus += tool
    P(f"corpus docs={len(corpus)} bytes={sum(len(t.encode()) for t in corpus)/1e6:.0f}MB")

    toks = {"bytes": ByteTokenizer()}
    for V in VOCABS:
      for regex, prefix in (("gpt2", "bpe"), ("multi", "mbpe")):
        name = f"{prefix}{V // 1024}k"
        path = f"data/tokenizers/{name}/tokenizer.json"
        if not os.path.exists(path):
            t0 = time.time()
            train_bpe(iter(corpus), V, f"data/tokenizers/{name}", regex=regex)
            P(f"trained {name} in {time.time()-t0:.0f}s")
        toks[name] = BPETokenizer(path)

    vals = {lang: val_texts(lang) for lang in LANGS}
    vals["tool"] = [r["text"] for r in read_tool_jsonl("data/raw/tool/val.jsonl.zst")[:1000]]
    metrics = {}
    for name, tok in toks.items():
        metrics[name] = {"vocab_size": tok.vocab_size}
        for lang, texts in vals.items():
            metrics[name][lang] = fertility(tok, texts, lang)
        P(name, json.dumps({k: (round(v["bytes_per_token"], 2) if isinstance(v, dict) else v)
                            for k, v in metrics[name].items()}))
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
