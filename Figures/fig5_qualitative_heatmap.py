# Generates Figure 5 (qualitative attention vs. gradient heatmap).
# Run: python3 fig5_qualitative_heatmap.py
# Output: fig5_qualitative_heatmap.png

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

chars = ["m","i","c","r","o","s","0","f","t","o","n","l","i","n","e",".","g","a"]

attention = np.array([
    0.39854106307029724, 0.07708936184644699, 0.011219048872590065, 0.18776634335517883,
    0.003134101629257202, 0.009992185048758984, 0.24194695055484772, 0.010463968850672245,
    1.3367292694965727e-07, 2.2448029994848184e-06, 0.03356580436229706, 0.001972552388906479,
    0.0025429127272218466, 0.0003734707133844495, 0.004554998129606247, 0.003956719767302275,
    0.012878041714429855, 3.787451063885783e-08
])

ig = np.array([
    4.754077911376953, 0.34217140078544617, 3.055217742919922, 0.45478522777557373,
    6.28214168548584, 0.26930108666419983, 4.071592807769775, 1.4013134241104126,
    0.32584595680236816, 1.1985840797424316, 0.8840401768684387, 4.454195499420166,
    6.859536647796631, 3.5849270820617676, 6.452025413513184, 8.965156555175781,
    7.4147114753723145, 4.765529155731201
])

# Normalize each row independently (min-max) for visual comparability -- raw scales differ
def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

attn_n = norm(attention)
ig_n = norm(ig)

n = len(chars)
fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)

rows = [("Integrated\nGradients", ig_n), ("Raw\nattention", attn_n)]
cmap = plt.cm.Reds

for row_i, (label, vals) in enumerate(rows):
    for i, v in enumerate(vals):
        ax.add_patch(plt.Rectangle((i, row_i), 1, 1, facecolor=cmap(0.15 + 0.75*v),
                                     edgecolor="white", linewidth=1.5))
    ax.text(-0.3, row_i + 0.5, label, ha="right", va="center", fontsize=9)

# Character labels
for i, c in enumerate(chars):
    ax.text(i + 0.5, -0.35, c, ha="center", va="center", fontsize=11, family="monospace")

# Highlight the homoglyph (position 6) and the TLD region (positions 15-17)
ax.add_patch(plt.Rectangle((6, 0), 1, 2, fill=False, edgecolor="#1565C0", linewidth=2.2))
ax.add_patch(plt.Rectangle((15, 0), 3, 2, fill=False, edgecolor="#1565C0", linewidth=2.2, linestyle="--"))

ax.text(6.5, 2.15, "homoglyph\n(0 for o)", ha="center", va="bottom", fontsize=7.5, color="#1565C0")
ax.text(16.5, 2.15, "TLD (.ga)", ha="center", va="bottom", fontsize=7.5, color="#1565C0")

ax.set_xlim(-1.2, n)
ax.set_ylim(-0.7, 2.6)
ax.axis("off")
ax.set_title("micros0ftonline.ga  (phishing, confidence=0.9999993)", fontsize=9.5, pad=2)

plt.tight_layout()
plt.savefig("fig5_qualitative_heatmap.png", dpi=200, bbox_inches="tight")
print("saved")
