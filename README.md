# Gradient-Based Attributions Pass, Attention Does Not Always

Code and data for the paper *"Gradient-Based Attributions Pass, Attention Does Not Always: A Multi-Seed Faithfulness Audit of Malicious URL Detection."*

## What this repository contains

This repository is the faithfulness audit only: a perturbation-based test of whether attention weights and gradient-based attribution (Input×Gradient, Integrated Gradients) faithfully explain the decisions of PhishFormer, a character-level CNN-Transformer malicious URL detector, replicated across **8 independently trained models** that share architecture, hyperparameters, and splitting procedure and differ in the random seed governing initialization, batch ordering, and split assignment.

The central finding: both evaluated gradient-based attribution methods pass a predefined perturbation-based faithfulness criterion in all 8 trained models; raw attention passes in exactly half and does not pass in the other half, with no observable predictor (training diagnostics, validation metrics, test accuracy) separating the two groups.

## Repository structure

```
attention-faithfulness-audit/
├── src/            model, data pipeline, training, and audit code
├── results/        training and audit output (JSON) for all 8 seeds
├── figures/        scripts that regenerate every figure in the paper
├── requirements.txt
├── .gitignore
└── LICENSE
```

**Checkpoints are not included** (8 models × ~20.5MB exceeds what is practical to host without Git LFS). Every result in this repository is reproducible from scratch by retraining — see below.

## Setup

```bash
git clone https://github.com/mdhasanmehedi/attention-faithfulness-audit.git
cd attention-faithfulness-audit
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

This project uses the **Malicious-Phish** dataset (651,191 URLs, 4 classes: benign, defacement, phishing, malware), publicly available on Kaggle:
[https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)

Download it and place it at:
```
<repo-root>/data/raw/malicious_phish.csv
```
(the `data/` folder is not tracked in this repository; create it locally)

## Reproducing the paper

All commands below are run from inside `src/`, with the dataset in place as described above.

### 1. Generate dedup-safe splits (one per seed)

The raw dataset contains 10,072 exact-duplicate URLs across 2,207 groups. `dedup_group_split.py` guarantees no duplicate URL crosses a train/val/test boundary, using `StratifiedGroupKFold`.

```bash
for S in 42 123 456 7 88 999 2024 31337; do
  python3 dedup_group_split.py --seed $S
done
```

Each run prints a leakage check and must end with `PASS: zero exact-duplicate leakage across splits.` before proceeding.

### 2. Train PhishFormer for each seed

```bash
python3 train.py --model phishformer --seeds 42 123 456 7 88 999 2024 31337 \
  --epochs 30 --batch_size 512 --lr 0.001 --patience 5 \
  --csv_path ../data/raw/malicious_phish.csv
```

This can be split across multiple invocations (e.g., a few seeds at a time). Each run saves a checkpoint to `checkpoints/phishformer_seed{N}_best.pt` and a training-history JSON to `results/`.

### 3. Run the faithfulness audit for each seed

```bash
for S in 42 123 456 7 88 999 2024 31337; do
  python3 faithfulness_audit.py --n_samples 500 --seed $S \
    --csv_path ../data/raw/malicious_phish.csv
done
```

Each run writes `results/attribution_faithfulness_seed{N}.json`. (Earlier versions of the script wrote to a fixed filename and required each seed's output to be copied out before the next run; that is no longer necessary. The eight result files committed here are named `audit_seed{N}.json` from that earlier convention, and the figure scripts read either name.)

The random-masking control draws 20 independent random masks per URL (`--n_random_masks 20`, the default) and averages the resulting confidence drops, rather than relying on a single draw — this reduces the sampling variance of the control relative to a naive single-draw baseline. Because the control's random number generator is re-seeded identically for each attribution method, all three methods are compared against the **same** random masks. The script also computes a 95% percentile bootstrap confidence interval (10,000 resamples) on the excess confidence drop (top-masking minus random-masking) for each method, and saves the raw per-URL drop arrays — including the full (n_urls × 20) matrix of individual random-mask drops before averaging — so that a different interval estimator could be applied later without rerunning the audit.

### 4. Select and dump the qualitative example (Figure 5)

```bash
python3 find_example_urls.py --seed 42
python3 find_example_urls.py --seed 42 --dump_full --dump_urls "micros0ftonline.ga"
```
The first command ranks candidate URLs by attention/gradient disagreement; the second dumps full per-character attribution arrays for the chosen example to `results/example_urls_full.json`.

### 5. Regenerate the figures

```bash
cd ../figures
python3 fig2_protocol_schematic.py
python3 fig3_multiseed_bars.py --results_dir ../results
python3 fig4_diagnostics_scatter.py --results_dir ../results
python3 fig5_qualitative_heatmap.py
```

`fig3` and `fig4` read every plotted value from the JSON files in `results/`, so a figure cannot silently disagree with the published numbers. Both refuse to run on audit output that does not match the protocol reported in the paper — a file recording fewer than 20 random masks per URL, a pass/fail verdict inconsistent with its own p-value, or a point estimate outside its own confidence interval will stop the script with an explanatory error rather than produce a plausible-looking wrong plot. Both also print the derived quantities cited in the paper (mean and SD of the attention excess drop, the pass/fail split, the p-value span, the gradient-method ranges), so regenerating a figure doubles as a cross-check against Table 1 and §6.2–6.4.

`fig2` is a schematic with no data inputs. `fig5` plots the per-character arrays for the single qualitative example, transcribed from `results/example_urls_full.json`. Figure 1 (architecture diagram) is a static illustration, not a generated plot.

## What's in `results/`

| File | Contents |
|---|---|
| `train_seed{42,123,456}.json` | Individual training runs, original 3 seeds |
| `train_seed7_999.json` | Combined training run, seeds 7 and 999 (`per_seed` array) |
| `train_seed2024_31337_88.json` | Combined training run, seeds 2024, 31337, 88 (`per_seed` array) |
| `audit_seed{N}.json` (×8) | Faithfulness-audit output for each seed — Wilcoxon statistics, p-values, top/random/bottom-masking confidence drops, 95% bootstrap confidence intervals on the excess drop, and raw per-URL drop arrays, for all three explanation methods. The random-masking control is the mean of 20 independent draws per URL. |
| `example_urls_full.json` | Full per-character attention and Integrated Gradients scores for the Figure 5 qualitative example, from real inference on the seed-42 checkpoint |

## Mapping paper items to source files

| Paper item | Source |
|---|---|
| Table 1 | `results/audit_seed*.json` (via `src/faithfulness_audit.py`) |
| Figure 2 | `figures/fig2_protocol_schematic.py` (schematic; no data inputs) |
| Figure 3 | `figures/fig3_multiseed_bars.py`, reads `results/audit_seed*.json` |
| Figure 4 | `figures/fig4_diagnostics_scatter.py`, reads `results/audit_seed*.json` and `results/train_seed*.json` |
| Figure 5 | `figures/fig5_qualitative_heatmap.py`, data from `results/example_urls_full.json` |
| §3.3 leakage disclosure (1.79%, 30.5%) | Reproducible via `src/verify_original_leakage.py` |
| §4.3 class weights | Reproducible via `src/data.py`'s `load_and_split()` (see docstring for usage) |
| §6.2 gradient-method ranges | Printed by `fig3_multiseed_bars.py`; from `results/audit_seed*.json` |
| §6.3 excess-drop mean, SD, split, p-value span | Printed by `fig3_multiseed_bars.py` and `fig4_diagnostics_scatter.py` |
| §6.4 diagnostics correlations | Computed by `fig4_diagnostics_scatter.py` from `results/train_seed*.json` and `results/audit_seed*.json` |
| §6.4 multiplicity correction (Holm-Bonferroni) | Computed from the p-values in `results/audit_seed*.json` |

## Citation

If you use this code or data, please cite:

```bibtex
@article{hasan2026gradientbased,
  title   = {Gradient-Based Attributions Pass, Attention Does Not
             Always: A Multi-Seed Faithfulness Audit of Malicious
             URL Detection},
  author  = {Hasan, Md Mehedi},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
