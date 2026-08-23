# Generates Figure 3 (excess confidence drop across 8 seeds, v2: 20-mask-averaged
# random control, with 95% bootstrap CI error bars).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ordered by v2 attention excess drop (ascending)
seeds   = [2024,    42,      88,     31337,  999,    7,      123,    456]
attn_ex = [-0.0400, -0.0299, +0.0044, +0.0261, +0.1089, +0.1331, +0.2974, +0.3074]
attn_lo = [-0.0738, -0.0660, -0.0319, -0.0089, +0.0746, +0.0961, +0.2618, +0.2757]
attn_hi = [-0.0068, +0.0063, +0.0407, +0.0613, +0.1441, +0.1702, +0.3319, +0.3393]
attn_p  = [0.9923,  0.9652,  0.5585,  0.3293,  9.19e-9, 9.68e-9, 1.17e-38, 1.72e-46]

ixg_ex  = [0.1159, 0.1788, 0.2125, 0.1509, 0.1930, 0.2522, 0.3769, 0.2190]
ixg_lo  = [0.0834, 0.1420, 0.1755, 0.1123, 0.1581, 0.2135, 0.3420, 0.1818]
ixg_hi  = [0.1478, 0.2158, 0.2497, 0.1904, 0.2272, 0.2913, 0.4109, 0.2562]

ig_ex   = [0.1034, 0.2766, 0.3607, 0.3335, 0.2725, 0.3463, 0.4098, 0.3498]
ig_lo   = [0.0705, 0.2451, 0.3311, 0.3000, 0.2392, 0.3127, 0.3763, 0.3160]
ig_hi   = [0.1359, 0.3078, 0.3899, 0.3670, 0.3051, 0.3791, 0.4428, 0.3833]

x = np.arange(len(seeds))
w = 0.26
fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=600)

colors_attn = ["#B33A3A" if p >= 0.05 else "#2E7D32" for p in attn_p]

def errbar(vals, lo, hi):
    return [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]

bars_attn = ax.bar(x - w, attn_ex, w, color=colors_attn, edgecolor="black", linewidth=0.6, label="Raw attention")
ax.errorbar(x - w, attn_ex, yerr=errbar(attn_ex, attn_lo, attn_hi), fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
for b, p in zip(bars_attn, attn_p):
    if p >= 0.05:
        b.set_hatch("///")

bars_ixg = ax.bar(x, ixg_ex, w, color="#7FA8D0", edgecolor="black", linewidth=0.6, label="Input\u00d7Gradient")
ax.errorbar(x, ixg_ex, yerr=errbar(ixg_ex, ixg_lo, ixg_hi), fmt="none", ecolor="black", elinewidth=0.8, capsize=2)

bars_ig = ax.bar(x + w, ig_ex, w, color="#2C5F8A", edgecolor="black", linewidth=0.6, label="Integrated Gradients")
ax.errorbar(x + w, ig_ex, yerr=errbar(ig_ex, ig_lo, ig_hi), fmt="none", ecolor="black", elinewidth=0.8, capsize=2)

ax.axhline(0, color="black", linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=8.5, rotation=20)
ax.set_ylabel("Excess confidence drop\n(top-20% masking \u2212 random-masking control)", fontsize=9)
ax.set_title("Faithfulness across 8 independently trained models\n(random-masking control: mean of 20 draws/URL; error bars = 95% bootstrap CI)",
              fontsize=9.5)
ax.legend(fontsize=8.5, loc="upper left", frameon=False, ncol=3)
ax.set_ylim(-0.12, 0.48)

for i, p in enumerate(attn_p):
    if p >= 0.05:
        ax.text(x[i] - w, attn_lo[i] - 0.018, "n.s.", ha="center", va="top", fontsize=7.5, fontweight="bold", color="#B33A3A")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("fig3_multiseed_bars.png", dpi=600, bbox_inches="tight")
print("saved")
