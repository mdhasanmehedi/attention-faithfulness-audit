# Generates Figure 2 (faithfulness-audit protocol schematic).
# Run: python3 fig2_protocol_schematic.py
# Output: fig2_protocol_schematic.png

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=600)
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis("off")

def box(x, y, w, h, text, fc="#EDEDED", ec="#333333", fontsize=9.5, weight="normal"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=1.1, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
             weight=weight, wrap=True)
    return b

def arrow(x1, y1, x2, y2, color="#333333", style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                          linewidth=1.1, color=color)
    ax.add_patch(a)

# Step 1: input
box(3.0, 9.6, 4.0, 0.9, "500 correctly classified\nmalicious URLs (test set)", fc="#E3EDF7", fontsize=9)
arrow(5.0, 9.6, 5.0, 8.9)

# Step 2: three parallel explanation methods
box(0.6, 7.6, 2.6, 1.0, "Raw attention\nweights", fc="#F7DEDE", fontsize=9)
box(3.7, 7.6, 2.6, 1.0, "Input\u00d7Gradient", fc="#DCEFE0", fontsize=9)
box(6.8, 7.6, 2.6, 1.0, "Integrated\nGradients", fc="#DCEFE0", fontsize=9)

arrow(1.9, 7.6, 1.9, 6.9)
arrow(5.0, 7.6, 5.0, 6.9)
arrow(8.1, 7.6, 8.1, 6.9)

# Step 3: rank characters
box(0.6, 5.9, 2.6, 1.0, "Rank characters\nby importance", fontsize=9)
box(3.7, 5.9, 2.6, 1.0, "Rank characters\nby importance", fontsize=9)
box(6.8, 5.9, 2.6, 1.0, "Rank characters\nby importance", fontsize=9)

arrow(1.9, 5.9, 1.9, 5.2)
arrow(5.0, 5.9, 5.0, 5.2)
arrow(8.1, 5.9, 8.1, 5.2)

# Step 4: mask top 20%
box(0.6, 4.2, 2.6, 1.0, "Mask top 20%\nof characters", fontsize=9)
box(3.7, 4.2, 2.6, 1.0, "Mask top 20%\nof characters", fontsize=9)
box(6.8, 4.2, 2.6, 1.0, "Mask top 20%\nof characters", fontsize=9)

arrow(1.9, 4.2, 1.9, 3.5)
arrow(5.0, 4.2, 5.0, 3.5)
arrow(8.1, 4.2, 8.1, 3.5)

# Step 5: measure confidence drop
box(0.6, 2.5, 2.6, 1.0, "Measure drop in\npredicted-class\nconfidence", fontsize=8.7)
box(3.7, 2.5, 2.6, 1.0, "Measure drop in\npredicted-class\nconfidence", fontsize=8.7)
box(6.8, 2.5, 2.6, 1.0, "Measure drop in\npredicted-class\nconfidence", fontsize=8.7)

arrow(1.9, 2.5, 1.9, 1.8)
arrow(5.0, 2.5, 5.0, 1.8)
arrow(8.1, 2.5, 8.1, 1.8)

# Random masking control (side input feeding all three comparisons).
# The control is the MEAN of 20 independent random masks per URL, not a single
# draw; this is the protocol described in Section 6.1 and Table 1's caption, and
# the schematic must state it or it understates what was run.
box(0.6, 0.8, 8.8, 1.0,
    "Compare against random 20% masking control \u2014 mean of 20 independent\n"
    "draws per URL; one-sided Wilcoxon signed-rank test, \u03b1=0.05, n=500",
    fc="#F5EEDD", fontsize=8.8)

plt.tight_layout()
plt.savefig("fig2_protocol_schematic.png", dpi=600, bbox_inches="tight")
print("saved")
