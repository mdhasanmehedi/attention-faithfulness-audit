# Generates Figure 4 (diagnostics vs. attention faithfulness scatter).
# Run: python3 fig4_diagnostics_scatter.py
# Output: fig4_diagnostics_scatter.png

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Verified data: seed -> (test_acc, test_f1, excess_drop, faithful)
data = {
    42:    dict(acc=0.9806, f1=0.9728, excess=-0.0249, faithful=False),
    2024:  dict(acc=0.9758, f1=0.9665, excess=-0.0387, faithful=False),
    88:    dict(acc=0.9796, f1=0.9711, excess=+0.0119, faithful=False),
    31337: dict(acc=0.9819, f1=0.9745, excess=+0.0247, faithful=False),
    999:   dict(acc=0.9780, f1=0.9691, excess=+0.1118, faithful=True),
    7:     dict(acc=0.9750, f1=0.9643, excess=+0.1511, faithful=True),
    123:   dict(acc=0.9740, f1=0.9664, excess=+0.2831, faithful=True),
    456:   dict(acc=0.9800, f1=0.9712, excess=+0.3475, faithful=True),
}

seeds = list(data.keys())
acc = np.array([data[s]['acc'] for s in seeds]) * 100
f1  = np.array([data[s]['f1']  for s in seeds]) * 100
ex  = np.array([data[s]['excess'] for s in seeds])
faith = np.array([data[s]['faithful'] for s in seeds])

r_acc, p_acc = stats.pearsonr(acc, ex)
r_f1, p_f1   = stats.pearsonr(f1, ex)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), dpi=200)

panels = [
    (axes[0], acc, "Test accuracy (%)", r_acc, p_acc),
    (axes[1], f1,  "Test macro-F1 (%)", r_f1, p_f1),
]

for ax, x, xlabel, r, p in panels:
    fail_mask = ~faith
    pass_mask = faith
    ax.scatter(x[fail_mask], ex[fail_mask], s=90, color="#B33A3A", edgecolor="black",
               linewidth=0.8, marker="X", label="Attention fails", zorder=3)
    ax.scatter(x[pass_mask], ex[pass_mask], s=90, color="#2E7D32", edgecolor="black",
               linewidth=0.8, marker="o", label="Attention passes", zorder=3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    for i, s in enumerate(seeds):
        ax.annotate(str(s), (x[i], ex[i]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points", color="#333333")

    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel("Attention excess confidence drop\n(top-20% masking \u2212 random masking)", fontsize=8.8)
    ax.set_title(f"Pearson r = {r:+.2f}  (p = {p:.2f})", fontsize=9.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].legend(fontsize=8, loc="lower left", frameon=False)

fig.suptitle("Attention faithfulness does not track model performance across 8 seeds", fontsize=10.5, y=1.02)
plt.tight_layout()
plt.savefig("fig4_diagnostics_scatter.png", dpi=200, bbox_inches="tight")
print("saved")
