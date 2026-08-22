"""
find_example_urls.py -- PhishFormer

Finds candidate example URLs for the qualitative attention-vs-gradient
figure, by ranking correctly-classified malicious test-set URLs by how
much raw attention and Integrated Gradients DISAGREE on which characters
matter most.

Reuses the existing attribution functions from integrated_gradients.py
unchanged -- this script only adds a ranking step on top.

Usage:
    python3 src/find_example_urls.py
    python3 src/find_example_urls.py --n_candidates 200 --top_k 15
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, load_checkpoint, device_info
from data import (
    load_and_split, tokenize_url,
    IDX2LABEL, LABEL2IDX, NUM_CLASSES, MAX_LEN, PAD_IDX
)
from models import PhishFormer
from integrated_gradients import (
    attribution_attention,
    attribution_input_x_gradient,
    attribution_integrated_gradients,
    MASK_RATIO, IG_STEPS, CKPT_DIR,
)

logger = get_logger()


def top_k_positions(attr: np.ndarray, seq_len: int, mask_ratio: float = MASK_RATIO) -> set:
    """Same top-k selection logic as run_faithfulness: highest-attributed
    character positions, count = max(1, int(seq_len * mask_ratio))."""
    n_mask = max(1, int(seq_len * mask_ratio))
    order = np.argsort(attr[:seq_len])  # ascending
    top = order[-n_mask:]
    return set(int(i) for i in top)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main(seed: int = 42, n_candidates: int = 200, top_k: int = 15,
          csv_path: str = "data/raw/malicious_phish.csv",
          dump_full: bool = False, dump_urls: str = ""):
    set_seed(seed)
    device = torch.device("cpu")  # gradients needed; CPU for stability, same as integrated_gradients.py
    logger.info(f"Device: {device_info(device)} (CPU forced for gradient stability)")

    # Load model -- identical to integrated_gradients.py
    model = PhishFormer().to(device)
    ckpt = os.path.join(CKPT_DIR, f"phishformer_seed{seed}_best.pt")
    if not os.path.exists(ckpt):
        logger.error(f"Checkpoint not found: {ckpt}")
        sys.exit(1)
    load_checkpoint(model, ckpt, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    logger.info(f"Loaded {ckpt}")

    # Build candidate pool: correctly classified malicious URLs (same logic as main())
    _, _, test_ds, _ = load_and_split(csv_path, seed=seed)
    logger.info("Finding correctly classified malicious URLs...")
    malicious_indices = []
    bs = 512
    for start in range(0, len(test_ds), bs):
        end = min(start + bs, len(test_ds))
        xs = torch.tensor(
            [tokenize_url(test_ds.urls[i]) for i in range(start, end)],
            dtype=torch.long
        ).to(device)
        ys = [test_ds.labels[i] for i in range(start, end)]
        with torch.no_grad():
            preds = model(xs).argmax(1).cpu().numpy()
        for li, (p, t) in enumerate(zip(preds, ys)):
            if t != LABEL2IDX["benign"] and p == t:
                malicious_indices.append(start + li)
        if len(malicious_indices) >= n_candidates * 3:
            break

    rng = np.random.default_rng(seed)
    candidates = rng.choice(
        malicious_indices, size=min(n_candidates, len(malicious_indices)), replace=False
    ).tolist()
    logger.info(f"Evaluating {len(candidates)} candidate URLs\n")

    results = []
    for count, idx in enumerate(candidates):
        url = test_ds.urls[idx]
        label = test_ds.labels[idx]
        tokens = torch.tensor([tokenize_url(url)], dtype=torch.long).to(device)
        seq_len = int((tokens[0] != PAD_IDX).sum().item())
        if seq_len < 8:
            continue  # too short to be a useful illustrative example

        with torch.no_grad():
            probs = torch.softmax(model(tokens), dim=1)
            conf = probs[0, label].item()

        attn = attribution_attention(model, tokens, label, device)
        ig = attribution_integrated_gradients(model, tokens, label, device, IG_STEPS)
        ixg = attribution_input_x_gradient(model, tokens, label, device)

        attn_top = top_k_positions(attn, seq_len)
        ig_top = top_k_positions(ig, seq_len)
        ixg_top = top_k_positions(ixg, seq_len)

        overlap_attn_ig = jaccard(attn_top, ig_top)
        overlap_attn_ixg = jaccard(attn_top, ixg_top)

        results.append({
            "idx": int(idx),
            "url": url,
            "label": IDX2LABEL[label],
            "confidence": conf,
            "seq_len": seq_len,
            "overlap_attn_vs_ig": overlap_attn_ig,
            "overlap_attn_vs_ixg": overlap_attn_ixg,
            "attn_top_positions": sorted(attn_top),
            "ig_top_positions": sorted(ig_top),
        })

        if (count + 1) % 50 == 0:
            logger.info(f"  processed {count+1}/{len(candidates)}")

    # Sort by lowest attention-vs-IG overlap first -- strongest disagreement
    results.sort(key=lambda r: r["overlap_attn_vs_ig"])

    logger.info("\n" + "=" * 90)
    logger.info(f"TOP {top_k} CANDIDATES -- LOWEST ATTENTION vs INTEGRATED-GRADIENTS OVERLAP")
    logger.info("=" * 90)
    for r in results[:top_k]:
        logger.info(
            f"\nURL: {r['url']}\n"
            f"  label={r['label']}  confidence={r['confidence']:.4f}  len={r['seq_len']}\n"
            f"  Jaccard(attention, IG)={r['overlap_attn_vs_ig']:.3f}   "
            f"Jaccard(attention, IxG)={r['overlap_attn_vs_ixg']:.3f}\n"
            f"  attention top chars (positions): {r['attn_top_positions']}\n"
            f"  IG top chars (positions):        {r['ig_top_positions']}"
        )
    logger.info("\n" + "=" * 90)
    logger.info("Pick 1-2 of the above with clear, human-recognizable structure "
                 "(e.g. brand impersonation, homoglyphs) for the qualitative figure.")

    # --- Optional: dump full per-character arrays for specific URLs ---
    # Use this on a second run once you've picked which URL(s) to use in the figure.
    if dump_full:
        targets = []
        if dump_urls:
            requested = [u.strip() for u in dump_urls.split(",") if u.strip()]
            for u in requested:
                match = next((r for r in results if r["url"] == u), None)
                if match is not None:
                    targets.append(match["idx"])
                else:
                    # Not in the candidate pool -- look it up directly in the test set,
                    # or fall back to tokenizing it standalone with predicted label as target.
                    found_idx = next((i for i in range(len(test_ds)) if test_ds.urls[i] == u), None)
                    if found_idx is None:
                        logger.warning(f"URL not found in test set, skipping: {u}")
                        continue
                    targets.append(found_idx)
        else:
            # No specific URLs given -- dump the single best (lowest-overlap) candidate
            if results:
                targets.append(results[0]["idx"])

        dump_out = []
        for idx in targets:
            url = test_ds.urls[idx]
            label = test_ds.labels[idx]
            tokens = torch.tensor([tokenize_url(url)], dtype=torch.long).to(device)
            seq_len = int((tokens[0] != PAD_IDX).sum().item())

            with torch.no_grad():
                probs = torch.softmax(model(tokens), dim=1)
                conf = probs[0, label].item()
                pred = int(probs.argmax(1).item())

            attn = attribution_attention(model, tokens, label, device)
            ig = attribution_integrated_gradients(model, tokens, label, device, IG_STEPS)
            ixg = attribution_input_x_gradient(model, tokens, label, device)

            dump_out.append({
                "url": url,
                "characters": list(url[:seq_len]),
                "true_label": IDX2LABEL[label],
                "predicted_label": IDX2LABEL[pred],
                "confidence": conf,
                "seq_len": seq_len,
                "attention": [float(x) for x in attn[:seq_len]],
                "input_x_gradient": [float(x) for x in ixg[:seq_len]],
                "integrated_gradients": [float(x) for x in ig[:seq_len]],
            })
            logger.info(f"Dumped full per-character arrays for: {url}")

        out_path = os.path.join("results", "example_urls_full.json")
        with open(out_path, "w") as f:
            json.dump(dump_out, f, indent=2)
        logger.info(f"\nFull per-character arrays saved to: {out_path}")
        logger.info("Send this file's contents back for the qualitative figure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_candidates", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--csv_path", type=str, default="data/raw/malicious_phish.csv")
    parser.add_argument("--dump_full", action="store_true",
                         help="Save full per-character attention/IxG/IG arrays to JSON.")
    parser.add_argument("--dump_urls", type=str, default="",
                         help="Comma-separated exact URL(s) to dump full arrays for. "
                              "If omitted with --dump_full, dumps the top-ranked candidate.")
    args = parser.parse_args()
    main(seed=args.seed, n_candidates=args.n_candidates, top_k=args.top_k,
         csv_path=args.csv_path, dump_full=args.dump_full, dump_urls=args.dump_urls)
