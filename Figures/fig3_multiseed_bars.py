# Generates Figure 3 (excess confidence drop across 8 seeds).
# Run: python3 fig3_multiseed_bars.py
# Output: fig3_multiseed_bars.png

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

seeds      = [2024,   42,     88,     31337,  999,    7,      123,    456]
attn_ex    = [-0.0387,-0.0249,+0.0119,+0.0247,+0.1118,+0.1511,+0.2831,+0.3475]
ixg_ex     = [0.1173, 0.1839, 0.2200, 0.1495, 0.1959, 0.2702, 0.3626, 0.2591]
ig_ex      = [0.1047, 0.2816, 0.3681, 0.3322, 0.2754, 0.3643, 0.3955, 0.3899]
attn_p     = [0.9607, 0.9410, 0.4966, 0.2979, 3.2e-6, 3.6e-10, 8.37e-30, 6.39e-35]

x = np.arange(len(seeds))
w = 0.26
fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)

colors_attn = ["#B33A3A" if p >= 0.05 else "#2E7D32" for p in attn_p]

bars_attn = ax.bar(x - w, attn_ex, w, color=colors_attn, edgecolor="black", linewidth=0.6, label="Raw attention")
for b, p in zip(bars_attn, attn_p):
    if p >= 0.05:
        b.set_hatch("///")
ax.bar(x,      ixg_ex, w, color="#7FA8D0", edgecolor="black", linewidth=0.6, label="Input\u00d7Gradient")
ax.bar(x + w,  ig_ex,  w, color="#2C5F8A", edgecolor="black", linewidth=0.6, label="Integrated Gradients")

ax.axhline(0, color="black", linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=8.5, rotation=20)
ax.set_ylabel("Excess confidence drop\n(top-20% masking \u2212 random masking)", fontsize=9)
ax.set_title("Faithfulness across 8 independently trained models\n(same architecture, data, and protocol; differing only in random seed)",
              fontsize=10)
ax.legend(fontsize=8.5, loc="upper left", frameon=False, ncol=3)
ax.set_ylim(-0.08, 0.45)

for i, p in enumerate(attn_p):
    if p >= 0.05:
        ax.text(x[i] - w, attn_ex[i] - 0.028 if attn_ex[i] < 0 else -0.028,
                 "FAILS", ha="center", va="top", fontsize=7.5, fontweight="bold", color="#B33A3A")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("fig3_multiseed_bars.png", dpi=200, bbox_inches="tight")
print("saved")
