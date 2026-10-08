"""Model assembly. A model is a stack of pre-norm residual blocks, each
    x = x + Mixer(norm(x));  x = x + SwiGLU(norm(x))
where the mixer type per layer is given by a pattern string:
    A = full causal attention (RoPE)       -> growing KV cache (baseline)
    W = sliding-window attention (RoPE)    -> KV capped at `window`
    G = gated linear attention (scalar decay)
    D = Gated DeltaNet
    P = Priority Memory Attention (M exact slots)
    C = gated short convolution (LFM2-style)
The backbone consumes embeddings [B,T,D] (modality-agnostic); `TokenLM` adds a
token embedding in and a tied linear head out. Other modalities only need their
own input adapter producing [B,T,D].
"""
import math
from dataclasses import asdict, dataclass, field

import mlx.core as mx
import mlx.nn as nn

from slm.models.layers import Attention, RMSNorm, ShortConv, SwiGLU
from slm.models.pma import PriorityMemoryAttention
from slm.models.recurrent import RecurrentMixer


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 256
    n_layers: int = 8
    pattern: str = "A"          # repeated/truncated to n_layers
    n_heads: int = 4
    mlp_hidden: int = 704
    window: int = 128
    n_slots: int = 64
    chunk: int = 64
    conv_k: int = 4
    tie_embeddings: bool = True
    extra: dict = field(default_factory=dict)

    def layer_types(self):
        p = self.pattern
        return (p * (self.n_layers // len(p) + 1))[: self.n_layers]

    def to_dict(self):
        return asdict(self)


class GatedShortConv(nn.Module):
    """LFM2-style block: y = C * conv(B * x)."""

    def __init__(self, d, k=3):
        super().__init__()
        self.w_in = nn.Linear(d, 3 * d, bias=False)
        self.conv = ShortConv(d, k)
        self.wo = nn.Linear(d, d, bias=False)

    def __call__(self, x, state=None):
        b, c, h = mx.split(self.w_in(x), 3, axis=-1)
        y, st = self.conv(b * h, None if state is None else state["conv"])
        out = self.wo(c * y)
        return out, (None if state is None else {"conv": st})

    def init_state(self, B):
        return {"conv": None}


def make_mixer(t, cfg):
    d, h = cfg.d_model, cfg.n_heads
    if t == "A":
        return Attention(d, h)
    if t == "W":
        return Attention(d, h, window=cfg.window)
    if t == "G":
        return RecurrentMixer(d, h, kind="gla", conv_k=cfg.conv_k, chunk=cfg.chunk)
    if t == "D":
        return RecurrentMixer(d, h, kind="gdn", conv_k=cfg.conv_k, chunk=cfg.chunk)
    if t == "P":
        return PriorityMemoryAttention(d, h, n_slots=cfg.n_slots, lams=cfg.extra.get("pma_lams"))
    if t == "C":
        return GatedShortConv(d, cfg.extra.get("conv_short_k", 3))
    raise ValueError(t)


class Block(nn.Module):
    def __init__(self, t, cfg):
        super().__init__()
        self.kind = t
        self.n1 = RMSNorm(cfg.d_model)
        self.mixer = make_mixer(t, cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def __call__(self, x, state=None):
        y, st = self.mixer(self.n1(x), state)
        x = x + y
        x = x + self.mlp(self.n2(x))
        return x, st


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = [Block(t, cfg) for t in cfg.layer_types()]
        self.norm = RMSNorm(cfg.d_model)

    def __call__(self, x, states=None):
        new = []
        for i, b in enumerate(self.blocks):
            x, st = b(x, None if states is None else states[i])
            new.append(st)
        return self.norm(x), (None if states is None else new)

    def init_state(self, B):
        return [b.mixer.init_state(B) for b in self.blocks]


class TokenLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.backbone = Backbone(cfg)
        if not cfg.tie_embeddings:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init()

    def _init(self):
        std = 0.02
        out_std = std / math.sqrt(2 * self.cfg.n_layers)
        self.embed.weight = mx.random.normal(self.embed.weight.shape) * std
        for b in self.backbone.blocks:
            for name, mod in b.named_modules():
                if isinstance(mod, nn.Linear):
                    is_out = name.split(".")[-1] in ("wo", "w_out")
                    mod.weight = mx.random.normal(mod.weight.shape) * (out_std if is_out else std)

    def __call__(self, tokens, states=None):
        h, states = self.backbone(self.embed(tokens), states)
        logits = self.embed.as_linear(h) if self.cfg.tie_embeddings else self.head(h)
        return logits, states

    def init_state(self, B):
        return self.backbone.init_state(B)


def count_params(model, exclude_embedding=False):
    from mlx.utils import tree_flatten

    n = 0
    for k, v in tree_flatten(model.parameters()):
        if exclude_embedding and k.startswith("embed"):
            continue
        n += v.size
    return n


def match_mlp_hidden(cfg: ModelConfig, target_params, multiple=32):
    """Pick mlp_hidden so total params ~= target (for equal-parameter comparisons)."""
    import copy

    best = None
    for hid in range(multiple, 8 * cfg.d_model + 1, multiple):
        c = copy.deepcopy(cfg)
        c.mlp_hidden = hid
        n = _analytic_params(c)
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (hid, n)
    cfg.mlp_hidden = best[0]
    return cfg


def _analytic_params(cfg):
    m = TokenLM.__new__(TokenLM)
    nn.Module.__init__(m)
    m.cfg = cfg
    m.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
    m.backbone = Backbone(cfg)
    return count_params(m)
