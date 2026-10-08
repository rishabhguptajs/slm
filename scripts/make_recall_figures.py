"""Figures for the long-text recall result (E10): a table image and a before/after chart.
Numbers are read from results/<run>/eval_suite_long.json (no hand-typed values).

python scripts/make_recall_figures.py  ->  reports/figures/recall_table.png, recall_chart.png
"""
import json
import os

import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = [  # (label, original run, long run, categorical slot colour)
    ("Transformer", "A_transformer_full", "A_transformer_long2k", "#2a78d6"),
    ("Mine (DeltaNet + Priority Memory)", "DP_gdn_pma_full", "DP_gdn_pma_long2k", "#eb6834"),
    ("DeltaNet only", "D_gdn_full", "D_gdn_long2k", "#1baf7a"),
]
DISTS = ["0", "512", "1024", "1800", "4096"]
OUT = "reports/figures"
plt.rcParams.update({"font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": 13, "text.color": INK})


def needle(run):
    return json.load(open(f"results/{run}/eval_suite_long.json"))["needle"]


data = {m[0]: (needle(m[1]), needle(m[2])) for m in MODELS}
os.makedirs(OUT, exist_ok=True)

# ---- 1) table image ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ax.axis("off")
fig.text(0.04, 0.93, "Recall before → after long-text training", fontsize=20, weight="bold")
fig.text(0.04, 0.875, "Bits of preference for the true planted code over a fake one (higher = better recall). "
         "14M-param models, 16 trials each.", fontsize=12, color=INK2)
col_x = [0.04, 0.27, 0.53, 0.79]
row_y0, dy = 0.74, 0.105
heads = ["Distance to fact"] + ["Transformer", "Mine", "DeltaNet only"]
for x, h, m in zip(col_x, heads, [None] + MODELS):
    fig.text(x, row_y0, h, fontsize=14, weight="bold")
    if m:
        fig.patches.append(plt.Rectangle((x, row_y0 - 0.025), 0.03, 0.012, transform=fig.transFigure,
                                         color=m[3], lw=0))
fig.add_artist(plt.Line2D([0.04, 0.96], [row_y0 - 0.04, row_y0 - 0.04], color=INK2, lw=1))
for i, d in enumerate(DISTS):
    y = row_y0 - dy * (i + 1)
    fig.text(col_x[0], y, f"{int(d):,} tokens", fontsize=14)
    for x, m in zip(col_x[1:], MODELS):
        before, after = data[m[0]][0][d]["gain_bits"], data[m[0]][1][d]["gain_bits"]
        delta = after - before
        fig.text(x, y, f"{before:.1f}  →  ", fontsize=14, color=INK2)
        fig.text(x + 0.085, y, f"{after:.1f}", fontsize=15, weight="bold")
        fig.text(x + 0.135, y, ("(±0.0)" if abs(delta) < 0.05 else f"({delta:+.1f})".replace("-", "−")), fontsize=12, color=INK2)
    fig.add_artist(plt.Line2D([0.04, 0.96], [y - 0.035, y - 0.035], color=GRID, lw=0.8))
fig.text(0.04, 0.06, "Long-text training: 12.3M tokens at 2,048-token length, 25% planted-fact practice "
         "(different wording/names from the test). Single seed.", fontsize=11, color=INK2)
fig.savefig(f"{OUT}/recall_table.png", facecolor=SURF)
plt.close(fig)

# ---- 2) before/after small multiples ----------------------------------------------
xs = [int(d) for d in DISTS[:4]]
fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), dpi=200, sharey=True)
fig.patch.set_facecolor(SURF)
for ax, idx, title in zip(axes, (0, 1), ("Before (trained on 512-token text)", "After long-text training (2,048)")):
    ax.set_facecolor(SURF)
    for label, _, _, c in MODELS:
        ys = [data[label][idx][d]["gain_bits"] for d in DISTS[:4]]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=8, mec=SURF, mew=2, zorder=3)
        short = label.split(" (")[0]
        ax.annotate(f"{short} {ys[0]:.1f}", (xs[0], ys[0]), xytext=(-12, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=12, color=INK, weight="bold")
    ax.set_xlim(-900, 1950)
    ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=12)
    ax.set_xticks(xs, [f"{x:,}" for x in xs])
    ax.set_xlabel("Tokens between the fact and the question", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
axes[0].set_ylabel("Recall (bits, higher = better)", color=INK2)
fig.suptitle("Long-text practice taught the transformer to remember. It made my fixed-memory model worse.",
             x=0.04, ha="left", fontsize=16, weight="bold", y=0.98)
fig.text(0.04, 0.015, "14M-param models trained from scratch on a MacBook Air (MLX). 16 trials per point, single seed.",
         fontsize=11, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(f"{OUT}/recall_chart.png", facecolor=SURF)
plt.close(fig)
print("wrote", f"{OUT}/recall_table.png", f"{OUT}/recall_chart.png")
