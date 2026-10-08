"""Tokenizers: byte-level (vocab 256+specials) and byte-level BPE trained on a
language-balanced multilingual sample. Both expose the same small interface
(encode / decode / vocab_size / special ids) so the rest of the stack is agnostic.
"""
import json
import os

SPECIALS = ["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>",
            "<|tool_call|>", "<|/tool_call|>", "<|tool_result|>", "<|functions|>", "<|end|>"]


class ByteTokenizer:
    """Raw UTF-8 bytes. ids 0..255 are bytes, specials follow."""

    name = "bytes"

    def __init__(self):
        self.special = {s: 256 + i for i, s in enumerate(SPECIALS)}
        self.vocab_size = 256 + len(SPECIALS)
        self.eot = self.special["<|endoftext|>"]

    def encode(self, text):
        return list(text.encode("utf-8"))

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        inv = {v: k for k, v in self.special.items()}
        out, buf = [], []
        for i in ids:
            if i < 256:
                buf.append(i)
            else:
                out.append(bytes(buf).decode("utf-8", errors="replace"))
                buf = []
                out.append(inv.get(i, ""))
        out.append(bytes(buf).decode("utf-8", errors="replace"))
        return "".join(out)


class BPETokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(path)
        self.name = os.path.basename(os.path.dirname(path))
        self.vocab_size = self.tok.get_vocab_size()
        self.special = {s: self.tok.token_to_id(s) for s in SPECIALS}
        self.eot = self.special["<|endoftext|>"]

    def encode(self, text):
        return self.tok.encode(text).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)


# Multilingual pre-tokenization regex. Differs from GPT-2's in that combining marks
# (\p{M}: Devanagari/Arabic/Thai vowel signs etc.) stay attached to letters; GPT-2's
# regex splits every Hindi word at each matra. Digits are split individually
# (helps exact copying of numbers/IDs in tool arguments).
MULTI_REGEX = (r"'(?:[sdmt]|ll|ve|re)| ?[\p{L}\p{M}]+| ?\p{N}| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+")


def train_bpe(texts_iter, vocab_size, out_dir, regex="multi"):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    if regex == "multi":
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(MULTI_REGEX), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:  # v1: GPT-2 regex (kept for the record)
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
        max_token_length=16,
    )
    tok.train_from_iterator(texts_iter, trainer=trainer)
    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    return os.path.join(out_dir, "tokenizer.json")


def load_tokenizer(name, root="data/tokenizers"):
    if name == "bytes":
        return ByteTokenizer()
    return BPETokenizer(os.path.join(root, name, "tokenizer.json"))


def tokenizer_meta(tok):
    return dict(name=tok.name, vocab_size=tok.vocab_size, special=tok.special)


if __name__ == "__main__":
    print(json.dumps(tokenizer_meta(ByteTokenizer()), indent=1)[:200])
