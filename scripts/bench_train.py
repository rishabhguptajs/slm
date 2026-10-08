"""Training-step throughput per architecture on random tokens (sizing the budget)."""
import sys, time, json
from functools import partial
sys.path.insert(0, "src")
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from slm.models.model import ModelConfig, TokenLM, count_params

def bench(pattern, B=16, T=512, d=256, L=12, steps=12, dtype="float32", chunk=64, **kw):
    cfg = ModelConfig(vocab_size=16384, d_model=d, n_layers=L, pattern=pattern, n_heads=4, mlp_hidden=704, chunk=chunk, **kw)
    m = TokenLM(cfg)
    if dtype != "float32": m.set_dtype(getattr(mx, dtype))
    opt = optim.AdamW(1e-3)
    lg = nn.value_and_grad(m, lambda m, x, y: nn.losses.cross_entropy(m(x)[0].astype(mx.float32), y, reduction="mean"))
    st = [m.state, opt.state]
    @partial(mx.compile, inputs=st, outputs=st)
    def step(x, y):
        l, g = lg(m, x, y); opt.update(m, g); return l
    x = mx.random.randint(0, 16384, (B, T)); y = mx.random.randint(0, 16384, (B, T))
    for _ in range(3): mx.eval(step(x, y), st)
    mx.reset_peak_memory(); t0 = time.time()
    for _ in range(steps): mx.eval(step(x, y), st)
    dt = (time.time() - t0) / steps
    return dict(pattern=pattern, dtype=dtype, params=count_params(m)/1e6, tok_s=B*T/dt, step_ms=dt*1000, peak_gb=mx.get_peak_memory()/1e9)

if __name__ == "__main__":
    for p, kw in [("A", {}), ("A", {"dtype": "bfloat16"}), ("D", {}), ("G", {}), ("DDDP", {}), ("DDDW", {}), ("C", {}), ("D", {"chunk": 32})]:
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in bench(p, **kw).items()} | ({"chunk": kw["chunk"]} if "chunk" in kw else {})), flush=True)
