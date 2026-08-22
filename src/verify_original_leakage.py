"""
verify_original_leakage.py -- PhishFormer

Reconstructs the ORIGINAL naive split (the one dedup_group_split.py was
built to fix) by forcing data.py's load_and_split() into its fallback
path -- the plain train_test_split-based split with no duplicate-URL
grouping -- and then directly measures train/test URL overlap.

This exists to verify two specific numbers currently in the manuscript
that predate this session and have no confirmed source:
  - "1.79% of test-set rows shared a URL string with a training-set row"
  - "30.5% of malware test rows specifically affected"

Usage:
    python3 verify_original_leakage.py
    python3 verify_original_leakage.py --seed 42 --csv_path ../data/raw/malicious_phish.csv
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from data import load_and_split, LABEL2IDX, IDX2LABEL
from utils import get_logger

logger = get_logger()


def main(seed: int = 42, csv_path: str = "data/raw/malicious_phish.csv"):
    # Force the fallback path: point splits_dir at a directory that does not
    # exist, so load_and_split() cannot find pre-generated dedup-safe splits
    # and falls back to the original in-function train_test_split logic.
    fake_splits_dir = "__force_fallback_do_not_create__"
    if os.path.exists(fake_splits_dir):
        logger.error(
            f"'{fake_splits_dir}' unexpectedly exists -- this would defeat the "
            f"purpose of this script. Remove it or rename it, then rerun."
        )
        sys.exit(1)

    logger.info(f"Reconstructing the ORIGINAL naive split (seed={seed})...")
    logger.info("Forcing fallback path (no dedup-safe grouping) via a nonexistent splits_dir.\n")

    train_ds, val_ds, test_ds, _ = load_and_split(
        csv_path, seed=seed, splits_dir=fake_splits_dir
    )

    train_urls = set(train_ds.urls)
    test_urls_list = test_ds.urls
    test_labels_list = test_ds.labels

    n_test = len(test_urls_list)
    leak_mask = [u in train_urls for u in test_urls_list]
    n_leak = sum(leak_mask)
    pct_leak = 100.0 * n_leak / n_test

    logger.info("=" * 70)
    logger.info("OVERALL TEST-SET LEAKAGE (naive split, no dedup grouping)")
    logger.info("=" * 70)
    logger.info(f"Total test rows                         : {n_test:,}")
    logger.info(f"Test rows sharing a URL with train       : {n_leak:,}")
    logger.info(f"Percentage of test rows leaked           : {pct_leak:.2f}%")
    logger.info("(Manuscript currently states: 1.79%)")

    # Per-class breakdown, with special attention to malware
    logger.info("\n" + "=" * 70)
    logger.info("PER-CLASS LEAKAGE BREAKDOWN")
    logger.info("=" * 70)
    for label_name, idx in LABEL2IDX.items():
        class_indices = [i for i, lab in enumerate(test_labels_list) if lab == idx]
        n_class = len(class_indices)
        if n_class == 0:
            continue
        n_class_leak = sum(leak_mask[i] for i in class_indices)
        pct_class_leak = 100.0 * n_class_leak / n_class
        marker = "  <-- check against manuscript's 30.5% claim" if label_name == "malware" else ""
        logger.info(
            f"  {label_name:>12s}: {n_class_leak:>6,} / {n_class:>7,} leaked "
            f"({pct_class_leak:>5.1f}%){marker}"
        )

    logger.info("\nDone. Compare the two flagged numbers above against the manuscript's")
    logger.info("§3.3 leakage disclosure (currently: 1.79% overall, 30.5% for malware).")
    logger.info("If they match closely, the manuscript's numbers are verified.")
    logger.info("If they differ substantially, the manuscript's numbers need correcting")
    logger.info("or softening to remove the specific percentages.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the naive split (default 42, matching the "
                             "project's primary seed).")
    parser.add_argument("--csv_path", type=str, default="data/raw/malicious_phish.csv")
    args = parser.parse_args()
    main(seed=args.seed, csv_path=args.csv_path)
