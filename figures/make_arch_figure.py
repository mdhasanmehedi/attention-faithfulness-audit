# Generates Figure 1 (PhishFormer architecture diagram).
#
# Run from the figures/ directory:
#   python3 make_arch_figure.py
#
# Output: fig1_architecture.png (200 dpi) and fig1_architecture.pdf (vector)
#
# Every hyperparameter shown is transcribed from src/models.py and matches the
# description in Section 4.1 of the paper: vocab 96, embedding 128, three Conv1D
# towers (k=3/4/5, 128 filters each, pad=1/2/2) concatenated to 384 channels,
# a 2-layer 4-head encoder with dim_feedforward=256 and norm_first=True
# (pre-norm), masked mean pooling, dropout 0.3, and a Linear(384, 4) head. The
# stated parameter total, 1,791,108, is reproduced exactly by summing those
# layer shapes. This is a diagram, not a plot of measured data: it has no
# inputs from results/.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9.2, 13.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 18.6); ax.axis("off")

C = dict(embed="#4C72B0", cnn3="#DD8452", cnn4="#C44E52", cnn5="#8172B3",
         concat="#937860", trans="#55A868", pool="#4C72B0", head="#CCB974",
         edge="#2f2f2f", txt="#1a1a1a")

def box(x, y, w, h, color, title, sub="", tsize=11, ssize=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=1.4, edgecolor=C["edge"], facecolor=color, alpha=0.92))
    ax.text(x+w/2, y+h/2 + (0.16 if sub else 0), title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color="white")
    if sub:
        ax.text(x+w/2, y+h/2 - 0.26, sub, ha="center", va="center",
                fontsize=ssize, color="white")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, linewidth=1.5, color=C["edge"]))

def tensor(x, y, txt):
    ax.text(x, y, txt, ha="center", va="center", fontsize=8.2,
            style="italic", color="#444", family="monospace")

ax.text(5, 18.1, "PhishFormer Architecture", ha="center", fontsize=15, fontweight="bold", color=C["txt"])
ax.text(5, 17.65, "character-level hybrid CNN\u2013Transformer, full-resolution fusion",
        ha="center", fontsize=9.5, color="#555")

# Input
box(3.2, 16.2, 3.6, 0.85, "#333333", "Input URL (character sequence)", "length 200, vocab 96", 10.5, 8.2)
tensor(5, 15.95, "(B, 200)")
arrow(5, 16.2, 5, 15.65)

# Embedding
box(3.0, 14.7, 4.0, 0.9, C["embed"], "Character Embedding", "nn.Embedding(96, 128)", 11, 8.5)
tensor(5, 14.45, "(B, 200, 128)  \u2192 transpose \u2192 (B, 128, 200)")
arrow(5, 14.7, 5, 14.05)

# CNN towers
tower_y = 12.45
box(1.05, tower_y, 2.5, 1.05, C["cnn3"], "Conv1D  k=3", "128 filters, pad=1", 10, 8)
box(3.75, tower_y, 2.5, 1.05, C["cnn4"], "Conv1D  k=4", "128 filters, pad=2", 10, 8)
box(6.45, tower_y, 2.5, 1.05, C["cnn5"], "Conv1D  k=5", "128 filters, pad=2", 10, 8)
ax.text(5, 13.75, "Parallel CNN towers  (ReLU, \u2018same\u2019 padding preserves all 200 positions)",
        ha="center", fontsize=8.6, color="#555")
for tx in (2.3, 5.0, 7.7):
    arrow(5, 13.95, tx, tower_y+1.05)
for tx in (2.3, 5.0, 7.7):
    tensor(tx, tower_y-0.28, "(B, 128, 200)")
    arrow(tx, tower_y, 5, 11.15)

# Concat
box(3.0, 10.25, 4.0, 0.9, C["concat"], "Concatenate along filter dim", "3 \u00d7 128 = 384 channels", 10.5, 8.3)
tensor(5, 10.0, "(B, 384, 200)  \u2192 transpose \u2192 (B, 200, 384)")
arrow(5, 10.25, 5, 9.65)
ax.text(9.35, 10.7, "full-resolution\nsequence retained", ha="right", fontsize=8, color="#C44E52", style="italic")

# ---- Transformer (FIXED: taller box, title at top, details stacked below) ----
tb_x, tb_y, tb_w, tb_h = 2.5, 7.35, 5.0, 2.05
ax.add_patch(FancyBboxPatch((tb_x, tb_y), tb_w, tb_h,
             boxstyle="round,pad=0.02,rounding_size=0.10",
             linewidth=1.4, edgecolor=C["edge"], facecolor=C["trans"], alpha=0.92))
ax.text(tb_x+tb_w/2, tb_y+tb_h-0.34, "Transformer Encoder", ha="center", va="center",
        fontsize=12, fontweight="bold", color="white")
ax.text(tb_x+tb_w/2, tb_y+tb_h-0.86, "2 layers  \u00b7  d_model = 384  \u00b7  4 heads",
        ha="center", va="center", fontsize=9, color="white")
ax.text(tb_x+tb_w/2, tb_y+tb_h-1.22, "dim_feedforward = 256  \u00b7  pre-norm",
        ha="center", va="center", fontsize=8.6, color="white")
ax.text(tb_x+tb_w/2, tb_y+tb_h-1.62, "self-attention over 200 positions",
        ha="center", va="center", fontsize=8.2, color="#eafaef", style="italic")
tensor(5, 7.12, "(B, 200, 384)")
arrow(5, 7.35, 5, 6.75)
# side note moved fully clear to the left of the box
ax.text(2.35, 8.35, "d_model = 384\nmatches CNN output\n(no projection needed)",
        ha="right", va="center", fontsize=7.8, color="#3a7d52", style="italic")

# Pooling
box(3.1, 5.85, 3.8, 0.85, C["pool"], "Masked Mean Pooling", "over sequence (ignores PAD)", 10.5, 8.2)
tensor(5, 5.6, "(B, 384)")
arrow(5, 5.85, 5, 5.25)

# Head
box(3.3, 4.35, 3.4, 0.85, C["head"], "Dropout (0.3) \u2192 Linear", "Linear(384, 4)", 10.5, 8.5)
tensor(5, 4.1, "(B, 4)")
arrow(5, 4.35, 5, 3.75)

# Output
box(3.35, 2.85, 3.3, 0.85, "#333333", "Softmax \u2192 4 classes",
    "benign / phishing / malware / defacement", 10, 7.4)

ax.text(5, 2.1, "Total trainable parameters: 1,791,108", ha="center", fontsize=9.5,
        fontweight="bold", color=C["txt"])
# Subtitle deliberately omits inference latency: the deployment-efficiency
# analysis was removed when the manuscript was refocused on the faithfulness
# audit, so a latency figure here would be a quantitative claim with no
# supporting method anywhere in the paper. This line reproduces the figure as
# published.
ax.text(5, 1.72, "checkpoint size 20.5 MB",
        ha="center", fontsize=8.3, color="#555")

plt.tight_layout()
plt.savefig("fig1_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("fig1_architecture.pdf", bbox_inches="tight", facecolor="white")
print("saved fig1_architecture.png and fig1_architecture.pdf")
