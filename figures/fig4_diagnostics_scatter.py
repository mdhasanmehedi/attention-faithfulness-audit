# Generates Figure 4 (diagnostics vs. attention faithfulness scatter).
#
# Run from the project root, with the venv active:
#   source venv/bin/activate
#   python3 src/fig4_diagnostics_scatter.py
#
# Output: results/fig4_diagnostics_scatter.png
#
# All plotted values are read from the canonical per-seed result files rather
# than hardcoded. An earlier revision of this script carried a hardcoded copy
# of the excess drops that predated the switch to a 20-draw-averaged
# random-masking control, and consequently disagreed with Table 1 of the
# manuscript. The assertions below exist to make that class of drift fail
# loudly instead of silently producing a plausible-looking wrong figure.

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_DIR = "results"
OUT_PATH = os.path.join(RESULTS_DIR, "fig4_diagnostics_scatter.png")

# Seed order is presentation-only; correlations are order-invariant.
SEEDS = [42, 2024, 88, 31337, 999, 7, 123, 456]

# The manuscript's protocol. Any audit file not matching this is pre-revision.
EXPECTED_RANDOM_MASKS = 20
EXPECTED_N_SAMPLES = 500
EXPECTED_MASK_RATIO = 0.20

# Candidate filename patterns, tried in order. Adjust if your layout differs.
AUDIT_PATTERNS = [
    "attribution_faithfulness_seed{seed}.json",
    "seed{seed}/attribution_faithfulness.json",
    "attribution_faithfulness_{seed}.json",
]
METRIC_PATTERNS = [
    "metrics_seed{seed}.json",
    "seed{seed}/metrics.json",
    "eval_seed{seed}.json",
    "test_metrics_seed{seed}.json",
]


def _resolve(patterns, seed, kind):
    """Return the first existing path among `patterns`, or exit with guidance."""
    tried = []
    for pat in patterns:
        path = os.path.join(RESULTS_DIR, pat.format(seed=seed))
        tried.append(path)
        if os.path.exists(path):
            return path
    sys.exit(
        f"\nERROR: no {kind} file found for seed {seed}. Tried:\n  "
        + "\n  ".join(tried)
        + f"\n\nEdit the {kind.upper()}_PATTERNS list at the top of this script "
          "to match your filenames.\n"
    )


def _dig(d, *keys, required=True, label=""):
    """Fetch the first present key from a dict, with a clear error if absent."""
    for k in keys:
        if k in d:
            return d[k]
    if required:
        sys.exit(
            f"\nERROR: none of {keys} found in {label}.\n"
            f"Available keys: {sorted(d)}\n"
        )
    return None


def load_audit(seed):
    """Attention excess drop and pass/fail verdict, from the canonical audit JSON."""
    path = _resolve(AUDIT_PATTERNS, seed, "audit")
    with open(path) as f:
        j = json.load(f)

    n_masks = j.get("n_random_masks_per_url")
    if n_masks is None:
        sys.exit(
            f"\nERROR: {path} has no 'n_random_masks_per_url' field.\n"
            "This file predates the 20-draw-averaged random-masking control and "
            "does not match the protocol described in the manuscript. Re-run "
            "faithfulness_audit.py for this seed before plotting.\n"
        )
    if n_masks != EXPECTED_RANDOM_MASKS:
        sys.exit(
            f"\nERROR: {path} used {n_masks} random mask(s) per URL, expected "
            f"{EXPECTED_RANDOM_MASKS}. The manuscript reports the {EXPECTED_RANDOM_MASKS}-draw "
            "averaged control; plotting this file would disagree with Table 1.\n"
        )

    n_samples = j.get("n_samples")
    if n_samples is not None and n_samples != EXPECTED_N_SAMPLES:
        print(f"  WARNING: seed {seed} audited on n={n_samples}, expected {EXPECTED_N_SAMPLES}")
    mask_ratio = j.get("mask_ratio")
    if mask_ratio is not None and abs(mask_ratio - EXPECTED_MASK_RATIO) > 1e-9:
        print(f"  WARNING: seed {seed} used mask_ratio={mask_ratio}, expected {EXPECTED_MASK_RATIO}")

    att = _dig(j["methods"], "attention", label=f"{path}:methods")
    excess = float(_dig(att, "excess_drop_mean", label=f"{path}:attention"))
    faithful = bool(_dig(att, "faithful", label=f"{path}:attention"))
    p_value = float(_dig(att, "p_value", label=f"{path}:attention"))

    # The verdict must follow from the p-value; a mismatch means the file was
    # edited by hand or written by a different criterion.
    if faithful != (p_value < 0.05):
        sys.exit(
            f"\nERROR: {path} is internally inconsistent — faithful={faithful} "
            f"but p={p_value:.4g} (criterion is p<0.05).\n"
        )
    return excess, faithful, p_value, path


def load_metrics(seed):
    """Test accuracy and test macro-F1 for one seed, as fractions in [0, 1]."""
    path = _resolve(METRIC_PATTERNS, seed, "metrics")
    with open(path) as f:
        j = json.load(f)
    acc = float(_dig(j, "test_accuracy", "accuracy", "test_acc", label=path))
    f1 = float(_dig(j, "test_macro_f1", "macro_f1", "test_f1", "macro_f1_score", label=path))
    # Accept either fraction or percentage and normalise to a fraction.
    if acc > 1.0:
        acc /= 100.0
    if f1 > 1.0:
        f1 /= 100.0
    return acc, f1, path


def load_all():
    data = {}
    print("Loading canonical results:")
    for seed in SEEDS:
        excess, faithful, p_value, apath = load_audit(seed)
        acc, f1, mpath = load_metrics(seed)
        data[seed] = dict(acc=acc, f1=f1, excess=excess,
                          faithful=faithful, p_value=p_value)
        print(f"  seed {seed:<6} excess={excess:+.4f}  p={p_value:.3g}  "
              f"acc={acc:.4f}  f1={f1:.4f}  {'PASS' if faithful else 'fail'}")
    return data


def report(data):
    """Print the derived quantities the manuscript cites, for cross-checking."""
    ex = np.array([data[s]["excess"] for s in SEEDS])
    npass = sum(data[s]["faithful"] for s in SEEDS)
    print("\nDerived quantities (cross-check against the manuscript):")
    print(f"  attention excess drop: mean={ex.mean():+.4f}  SD={ex.std(ddof=1):.4f}")
    print(f"  range: {ex.min():+.4f} to {ex.max():+.4f}")
    print(f"  pass / fail split: {npass} / {len(SEEDS) - npass}")

    fails = sorted(data[s]["excess"] for s in SEEDS if not data[s]["faithful"])
    passes = sorted(data[s]["excess"] for s in SEEDS if data[s]["faithful"])
    if fails and passes:
        print(f"  non-passing: {fails[0]:+.4f} to {fails[-1]:+.4f}")
        print(f"  passing:     {passes[0]:+.4f} to {passes[-1]:+.4f}")
        if passes[0] > fails[-1]:
            print(f"  gap: {passes[0] - fails[-1]:+.4f} "
                  f"({passes[0] / fails[-1]:.1f}x the largest non-passing drop)"
                  if fails[-1] > 0 else f"  gap: {passes[0] - fails[-1]:+.4f}")
        else:
            print("  NOTE: pass/fail groups overlap in excess drop — no clean gap.")

    ps = [data[s]["p_value"] for s in SEEDS]
    span = np.log10(max(ps) / min(ps))
    print(f"  p-value span: {span:.1f} orders of magnitude")


def make_figure(data, out_path=OUT_PATH):
    acc = np.array([data[s]["acc"] for s in SEEDS]) * 100
    f1 = np.array([data[s]["f1"] for s in SEEDS]) * 100
    ex = np.array([data[s]["excess"] for s in SEEDS])
    faith = np.array([data[s]["faithful"] for s in SEEDS])

    r_acc, p_acc = stats.pearsonr(acc, ex)
    r_f1, p_f1 = stats.pearsonr(f1, ex)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), dpi=200)
    panels = [
        (axes[0], acc, "Test accuracy (%)", r_acc, p_acc),
        (axes[1], f1, "Test macro-F1 (%)", r_f1, p_f1),
    ]

    for ax, x, xlabel, r, p in panels:
        ax.scatter(x[~faith], ex[~faith], s=90, color="#B33A3A", edgecolor="black",
                   linewidth=0.8, marker="X", label="Attention fails", zorder=3)
        ax.scatter(x[faith], ex[faith], s=90, color="#2E7D32", edgecolor="black",
                   linewidth=0.8, marker="o", label="Attention passes", zorder=3)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

        for i, s in enumerate(SEEDS):
            ax.annotate(str(s), (x[i], ex[i]), fontsize=7, xytext=(4, 4),
                        textcoords="offset points", color="#333333")

        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylabel("Attention excess confidence drop\n"
                      "(top-20% masking \u2212 random masking)", fontsize=8.8)
        ax.set_title(f"Pearson r = {r:+.2f}  (p = {p:.2f})", fontsize=9.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Legend below the axes: placing it inside the lower-left of the left panel
    # overlapped the seed-2024 point label.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Attention faithfulness does not track model performance across 8 seeds",
                 fontsize=10.5, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPearson correlations printed on the figure:")
    print(f"  test accuracy : r = {r_acc:+.2f}  (p = {p_acc:.2f})")
    print(f"  test macro-F1 : r = {r_f1:+.2f}  (p = {p_f1:.2f})")
    print(f"\nSaved {out_path}")
    return r_acc, p_acc, r_f1, p_f1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=RESULTS_DIR,
                    help="Directory holding the per-seed JSON files (default: results)")
    ap.add_argument("--out", default=None, help="Output PNG path")
    args = ap.parse_args()

    RESULTS_DIR = args.results_dir
    out = args.out or os.path.join(RESULTS_DIR, "fig4_diagnostics_scatter.png")

    data = load_all()
    report(data)
    make_figure(data, out)
