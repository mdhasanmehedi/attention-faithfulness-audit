# Gradients Are Reliable, Attention Is Not Always

Code and data for the paper *"Gradients Are Reliable, Attention Is Not Always: A Multi-Seed Faithfulness Audit of Explainable Malicious URL Detection."*

## What this repository contains

This repository is the faithfulness audit only: a perturbation-based causal test of whether attention weights and gradient-based attribution (Input×Gradient, Integrated Gradients) faithfully explain the decisions of PhishFormer, a character-level CNN-Transformer malicious URL detector, replicated across **8 independently trained models** that differ only in random seed.

The central finding: gradient-based attribution is causally faithful in all 8 trained models; raw attention is faithful in exactly half and fails outright in the other half, with no observable predictor (training diagnostics, validation metrics, test accuracy) separating the two groups.

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
python3 dedup_group_split.py --seed 42
python3 dedup_group_split.py --seed 123
python3 dedup_group_split.py --seed 456
python3 dedup_group_split.py --seed 7
python3 dedup_group_split.py --seed 88
python3 dedup_group_split.py --seed 999
python3 dedup_group_split.py --seed 2024
python3 dedup_group_split.py --seed 31337
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
  python3 integrated_gradients.py --n_samples 500 --seed $S \
    --csv_path ../data/raw/malicious_phish.csv
  cp results/attribution_faithfulness.json results/audit_seed${S}.json
done
```

The `cp` step is important: the script writes to a fixed filename each run, so each seed's result must be copied out immediately or it will be overwritten by the next seed.

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
python3 fig3_multiseed_bars.py
python3 fig4_diagnostics_scatter.py
python3 fig5_qualitative_heatmap.py
```

Each script is self-contained (the underlying numbers are embedded directly in the script, extracted from the `results/` files) and saves its output PNG in the same folder. Figure 1 (architecture diagram) is a static illustration, not a generated plot.

## What's in `results/`

| File | Contents |
|---|---|
| `train_seed{42,123,456}.json` | Individual training runs, original 3 seeds |
| `train_seed7_999.json` | Combined training run, seeds 7 and 999 (`per_seed` array) |
| `train_seed2024_31337_88.json` | Combined training run, seeds 2024, 31337, 88 (`per_seed` array) |
| `audit_seed{N}.json` (×8) | Faithfulness-audit output for each seed — Wilcoxon statistics, p-values, top/random/bottom-masking confidence drops for all three explanation methods |
| `example_urls_full.json` | Full per-character attention and Integrated Gradients scores for the Figure 5 qualitative example, from real inference on the seed-42 checkpoint |

## Mapping paper items to source files

| Paper item | Source |
|---|---|
| Table 1 | `results/audit_seed*.json` |
| Figure 2 | `figures/fig2_protocol_schematic.py` |
| Figure 3 | `figures/fig3_multiseed_bars.py`, data from `results/audit_seed*.json` |
| Figure 4 | `figures/fig4_diagnostics_scatter.py`, data from `results/audit_seed*.json` and `results/train_seed*.json` |
| Figure 5 | `figures/fig5_qualitative_heatmap.py`, data from `results/example_urls_full.json` |
| §6.4 diagnostics correlations | Computed from `results/train_seed*.json` and `results/audit_seed*.json` |
| §3.3 leakage disclosure (1.79%, 30.5%) | Reproducible via the naive-split reconstruction in `src/verify_original_leakage.py` |

## Citation

If you use this code or data, please cite:

```bibtex
@article{hasan2026gradients,
  title   = {Gradients Are Reliable, Attention Is Not Always: A Multi-Seed
             Faithfulness Audit of Explainable Malicious URL Detection},
  author  = {Hasan, Md Mehedi},
  journal = {Journal of Network and Computer Applications},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
