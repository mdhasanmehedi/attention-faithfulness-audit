# Generates Figure 3 (excess confidence drop across 8 seeds, three explanation
# methods, with 95% bootstrap CI error bars).
#
# Run from the project root, with the venv active:
#   source venv/bin/activate
#   python3 src/fig3_multiseed_bars.py
#
# Output: results/fig3_multiseed_bars.png
#
# All 48 plotted values are read from the canonical per-seed audit files rather
# than hardcoded. An earlier sibling script (fig4_diagnostics_scatter.py) carried
# a hardcoded copy of the excess drops that predated the switch to a 20-draw
# averaged random-masking control, and silently disagreed with Table 1. The
# assertions below make that class of drift fail loudly instead of producing a
# plausible-looking wrong figure.

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"

# Seeds are re-sorted by attention excess drop before plotting; this list only
# declares which seeds must be present.
SEEDS = [7, 42, 88, 123, 456, 999, 2024, 31337]

# The manuscript's protocol. Any audit file not matching this is pre-revision.
EXPECTED_RANDOM_MASKS = 20
EXPECTED_N_SAMPLES = 500
EXPECTED_MASK_RATIO = 0.20
ALPHA = 0.05

# Method keys as written by faithfulness_audit.py, in plotting order.
METHODS = [
    ("attention", "Raw attention"),
    ("input_x_gradient", "Input\u00d7Gradient"),
    ("integrated_gradients", "Integrated Gradients"),
]

AUDIT_PATTERNS = [
    "attribution_faithfulness_seed{seed}.json",
    "seed{seed}/attribution_faithfulness.json",
    "attribution_faithfulness_{seed}.json",
]


def _resolve(seed):
    tried = []
    for pat in AUDIT_PATTERNS:
        path = os.path.join(RESULTS_DIR, pat.format(seed=seed))
        tried.append(path)
        if os.path.exists(path):
            return path
    sys.exit(
        f"\nERROR: no audit file found for seed {seed}. Tried:\n  "
        + "\n  ".join(tried)
        + "\n\nEdit AUDIT_PATTERNS at the top of this script to match your filenames.\n"
    )


def _need(d, key, label):
    if key not in d:
        sys.exit(f"\nERROR: '{key}' missing from {label}.\nAvailable keys: {sorted(d)}\n")
    return d[key]


def load_seed(seed):
    """Return {method_key: dict(excess, lo, hi, p, faithful)} for one seed."""
    path = _resolve(seed)
    with open(path) as f:
        j = json.load(f)

    n_masks = j.get("n_random_masks_per_url")
    if n_masks is None:
        sys.exit(
            f"\nERROR: {path} has no 'n_random_masks_per_url' field.\n"
            "This file predates the 20-draw-averaged random-masking control and does "
            "not match the protocol described in the manuscript. Re-run "
            "faithfulness_audit.py for this seed before plotting.\n"
        )
    if n_masks != EXPECTED_RANDOM_MASKS:
        sys.exit(
            f"\nERROR: {path} used {n_masks} random mask(s) per URL, expected "
            f"{EXPECTED_RANDOM_MASKS}. Plotting this file would disagree with Table 1.\n"
        )

    if j.get("n_samples") not in (None, EXPECTED_N_SAMPLES):
        print(f"  WARNING: seed {seed} audited on n={j['n_samples']}, expected {EXPECTED_N_SAMPLES}")
    mr = j.get("mask_ratio")
    if mr is not None and abs(mr - EXPECTED_MASK_RATIO) > 1e-9:
        print(f"  WARNING: seed {seed} used mask_ratio={mr}, expected {EXPECTED_MASK_RATIO}")

    methods = _need(j, "methods", path)
    out = {}
    for key, _ in METHODS:
        m = _need(methods, key, f"{path}:methods")
        lab = f"{path}:{key}"
        rec = dict(
            excess=float(_need(m, "excess_drop_mean", lab)),
            lo=float(_need(m, "excess_drop_ci_low", lab)),
            hi=float(_need(m, "excess_drop_ci_high", lab)),
            p=float(_need(m, "p_value", lab)),
            faithful=bool(_need(m, "faithful", lab)),
        )
        if rec["faithful"] != (rec["p"] < ALPHA):
            sys.exit(
                f"\nERROR: {path} ({key}) is internally inconsistent — "
                f"faithful={rec['faithful']} but p={rec['p']:.4g} (criterion p<{ALPHA}).\n"
            )
        if not (rec["lo"] <= rec["excess"] <= rec["hi"]):
            sys.exit(
                f"\nERROR: {path} ({key}) has a point estimate outside its own CI: "
                f"{rec['excess']:.4f} not in [{rec['lo']:.4f}, {rec['hi']:.4f}].\n"
            )
        out[key] = rec
    return out, path


def load_all():
    print("Loading canonical results:")
    data = {}
    for s in SEEDS:
        data[s], path = load_seed(s)
        a = data[s]["attention"]
        print(f"  seed {s:<6} attention excess={a['excess']:+.4f} "
              f"[{a['lo']:+.4f},{a['hi']:+.4f}] p={a['p']:.3g} "
              f"{'PASS' if a['faithful'] else 'fail'}")
    return data


def report(data, order):
    """Print the quantities the manuscript cites, for cross-checking."""
    att = np.array([data[s]["attention"]["excess"] for s in order])
    ps = [data[s]["attention"]["p"] for s in order]
    faith = [data[s]["attention"]["faithful"] for s in order]

    print("\nDerived quantities (cross-check against the manuscript):")
    print(f"  attention excess drop: mean={att.mean():+.4f}  SD={att.std(ddof=1):.4f}")
    print(f"  range: {att.min():+.4f} to {att.max():+.4f}")
    print(f"  pass / fail split: {sum(faith)} / {len(faith) - sum(faith)}")
    print(f"  p-value span: {np.log10(max(ps) / min(ps)):.1f} orders of magnitude")

    fails = sorted(data[s]["attention"]["excess"] for s in order if not data[s]["attention"]["faithful"])
    passes = sorted(data[s]["attention"]["excess"] for s in order if data[s]["attention"]["faithful"])
    if fails and passes:
        print(f"  non-passing: {fails[0]:+.4f} to {fails[-1]:+.4f}")
        print(f"  passing:     {passes[0]:+.4f} to {passes[-1]:+.4f}")
        if passes[0] > fails[-1]:
            gap = passes[0] - fails[-1]
            ratio = f", {passes[0] / fails[-1]:.1f}x the largest non-passing drop" if fails[-1] > 0 else ""
            print(f"  gap: {gap:+.4f}{ratio}")
        else:
            print("  NOTE: pass/fail groups overlap — no clean gap.")

    n_zero = sum(1 for s in order
                 if not data[s]["attention"]["faithful"]
                 and data[s]["attention"]["lo"] < 0 < data[s]["attention"]["hi"])
    n_fail = len(fails)
    print(f"  non-passing CIs containing zero: {n_zero}/{n_fail}")
    below = [s for s in order if data[s]["attention"]["hi"] < 0]
    if below:
        print(f"  CIs lying entirely below zero: seeds {below}")

    for key, label in METHODS[1:]:
        v = [data[s][key]["excess"] for s in order]
        allpass = all(data[s][key]["faithful"] for s in order)
        worst = max(data[s][key]["p"] for s in order)
        above = all(data[s][key]["lo"] > 0 for s in order)
        print(f"  {label}: range {min(v):+.4f} to {max(v):+.4f}; "
              f"passes all {len(order)}: {allpass}; weakest p={worst:.2g}; "
              f"all CIs above zero: {above}")


def make_figure(data, order, out_path):
    n = len(order)
    x = np.arange(n)
    w = 0.26

    def series(key):
        e = np.array([data[s][key]["excess"] for s in order])
        lo = np.array([data[s][key]["lo"] for s in order])
        hi = np.array([data[s][key]["hi"] for s in order])
        return e, [e - lo, hi - e]

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=600)

    att_e, att_err = series("attention")
    att_p = [data[s]["attention"]["p"] for s in order]
    att_lo = [data[s]["attention"]["lo"] for s in order]
    colors = ["#B33A3A" if p >= ALPHA else "#2E7D32" for p in att_p]

    bars = ax.bar(x - w, att_e, w, color=colors, edgecolor="black",
                  linewidth=0.6, label="Raw attention")
    ax.errorbar(x - w, att_e, yerr=att_err, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=2)
    for b, p in zip(bars, att_p):
        if p >= ALPHA:
            b.set_hatch("///")

    ixg_e, ixg_err = series("input_x_gradient")
    ax.bar(x, ixg_e, w, color="#7FA8D0", edgecolor="black", linewidth=0.6,
           label="Input\u00d7Gradient")
    ax.errorbar(x, ixg_e, yerr=ixg_err, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=2)

    ig_e, ig_err = series("integrated_gradients")
    ax.bar(x + w, ig_e, w, color="#2C5F8A", edgecolor="black", linewidth=0.6,
           label="Integrated Gradients")
    ax.errorbar(x + w, ig_e, yerr=ig_err, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=2)

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in order], fontsize=8.5, rotation=20)
    ax.set_ylabel("Excess confidence drop\n"
                  "(top-20% masking \u2212 random-masking control)", fontsize=9)
    ax.set_title(f"Faithfulness across {n} independently trained models\n"
                 f"(random-masking control: mean of {EXPECTED_RANDOM_MASKS} draws/URL; "
                 "error bars = 95% bootstrap CI)", fontsize=9.5)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False, ncol=3)

    # Bounds derived from the data, with headroom for the legend and n.s. labels.
    all_hi = max(max(data[s][k]["hi"] for k, _ in METHODS) for s in order)
    all_lo = min(min(data[s][k]["lo"] for k, _ in METHODS) for s in order)
    ax.set_ylim(min(all_lo - 0.08, -0.12), all_hi + 0.07)

    for i, p in enumerate(att_p):
        if p >= ALPHA:
            ax.text(x[i] - w, att_lo[i] - 0.018, "n.s.", ha="center", va="top",
                    fontsize=7.5, fontweight="bold", color="#B33A3A")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=RESULTS_DIR,
                    help="Directory holding the per-seed audit JSON files (default: results)")
    ap.add_argument("--out", default=None, help="Output PNG path")
    args = ap.parse_args()

    RESULTS_DIR = args.results_dir
    out = args.out or os.path.join(RESULTS_DIR, "fig3_multiseed_bars.png")

    data = load_all()
    # Bars are ordered by attention excess drop so the bimodal split reads as a gap.
    order = sorted(SEEDS, key=lambda s: data[s]["attention"]["excess"])
    report(data, order)
    make_figure(data, order, out)
