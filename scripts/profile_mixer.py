"""Time forward+backward of single mixers and GDN sub-pieces (B=16,T=512,d=256,H=4)."""
import sys, time
sys.path.insert(0, "src")
import mlx.core as mx, mlx.nn as nn
from slm.models.layers import Attention
from slm.models.recurrent import RecurrentMixer, unit_lower_inverse, gdn_chunked, gla_chunked
from slm.models.pma import PriorityMemoryAttention

def timeit(f, *args, n=10):
    g = mx.compile(f)
    for _ in range(3): mx.eval(g(*args))
    t = time.time()
    for _ in range(n): mx.eval(g(*args))
    return (time.time() - t) / n * 1000

B, T, d, H = 16, 512, 256, 4
x = mx.random.normal((B, T, d))
for name, m in [("attn", Attention(d, H)), ("gdn", RecurrentMixer(d, H, "gdn")), ("gla", RecurrentMixer(d, H, "gla")),
                ("gdn_c32", RecurrentMixer(d, H, "gdn", chunk=32)), ("gdn_c128", RecurrentMixer(d, H, "gdn", chunk=128)),
                ("pma", PriorityMemoryAttention(d, H, 64))]:
    fb = nn.value_and_grad(m, lambda m, x: m(x)[0].sum())
    print(f"{name:10s} fwd {timeit(lambda x: m(x)[0], x):7.1f} ms   fwd+bwd {timeit(lambda x: fb(m, x)[1], x):7.1f} ms", flush=True)

# inverse pieces
for C in (32, 64):
    for base in (4, 8, 16):
        A = mx.tril(mx.random.normal((B, H, T // C, C, C)) * 0.1, k=-1) + mx.eye(C)
        f = lambda A: unit_lower_inverse(A, base)
        gf = mx.grad(lambda A: unit_lower_inverse(A, base).sum())
        print(f"inverse C={C} base={base}: fwd {timeit(f, A):6.1f} ms fwd+bwd {timeit(gf, A):6.1f} ms", flush=True)
