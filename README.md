# LLMDroid

**APK-Free screening for LLM-integrated Android apps.**

LLMDroid flags likely LLM-powered apps using only public app-store metadata: text descriptions and promotional screenshots.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-under%20review-orange)]()

---

## How it works

```
App listing (title + description + screenshots)
        │
        ├─ Text branch ──── BGE-large-en-v1.5 (1024-d SBERT)
        │                   + keyword features (13-d)
        │                   + metadata features (21-d)
        │
        └─ Image branch ─── CLIP ViT-L/14-336 mean+max (1536-d)
                            + zero-shot chat-UI score (1-d)
                            + OCR keyword features (15-d)
                                    │
                                    ▼
                    LightGBM + SelectKBest (k=200)
                    Fusion: Score-Max / Soft Voting / Stacking / Early Fusion
```

**5-fold CV on 298 apps · Independent test N=110 · ROC-AUC up to 0.948**

---

## Setup

```bash
git clone https://github.com/Tran-Trung-Nhut/LLMDroid
cd LLMDroid

conda create -n llmdroid python=3.10 -y && conda activate llmdroid
pip install -r requirements.txt

# System deps (Ubuntu / Lightning AI)
sudo apt-get install -y tesseract-ocr default-jre
```

---

## Run

### 1 — Training pipeline

```bash
python src/train_pipeline.py
```

| Step | What happens | Skip flag |
|------|-------------|-----------|
| 0 | Download missing screenshots from Google Play | `--skip-image-download` |
| 1 | Preprocess text (clean HTML, dedup images) | — |
| 2 | OCR screenshots via Tesseract | `--skip-ocr` |
| 3 | Create stratified 5-fold splits | — |
| 4a | Extract text features (BGE-large + keywords + meta) | `--skip-features` |
| 4b | Extract image features (CLIP + zero-shot + OCR) | `--skip-features` |
| 4c | Extract SLM features (Qwen2.5-1.5B, ablation only) | `--skip-features` |
| — | k-sensitivity for SelectKBest | `--skip-k-sensitivity` |
| 5 | Train & evaluate all fusion strategies | — |

Downloaded screenshots are deleted automatically after training unless `--keep-images` is passed.

If features are already cached (`data/features/`), jump straight to training:

```bash
python src/train_pipeline.py --train-only
```

### 2 — Evaluate on independent test set

This is a **required** pre-step before running analysis. Run it once after training:

```bash
python src/steps/independent_test_eval.py
```

Outputs predictions for all strategies (Text-Only, Image-Only, Early Fusion, Soft Voting, Stacking, Score-Max) to `runs/feature_fusion/independent_test/`.

### 3 — Post-training analysis

```bash
python src/run_analysis.py
```

| Step | What |
|------|------|
| 6.1 | Statistical significance tests (DeLong + bootstrap F1) |
| 6.2 | Branch complementarity |
| 6.3 | Disagreement accuracy |
| 6.4 | Per-category performance |
| 6.5 | Prior-corrected precision |
| 6.6 | Temporal split |
| 6.7 | Inference latency benchmark |
| 6.8 | Probability calibration |
| 6.9 | Robustness to missing modalities |
| 6.10 | Per-label-criterion evaluation (construct validity) |
| 6.11 | Image branch helps/hurts qualitative analysis |
| 6.12 | Keyword-drift robustness (remove model-name keywords) |
| 6.13 | SHAP feature importance (Text-Only + Early Fusion) |

### 4 — Baselines

```bash
# All local baselines (TF-IDF+SVM + Qwen2.5-7B + E2E transformer)
python src/run_baselines.py

# Include API baselines (GPT-4o-mini zero-shot + GPT-4o 6-shot)
export OPENAI_API_KEY=sk-...
python src/run_baselines.py --all
```

Flags: `--skip-tfidf-svm`, `--skip-qwen`, `--skip-e2e`, `--skip-local`, `--no-latency`.

| Baseline | Type | Output |
|----------|------|--------|
| TF-IDF + linear SVM | description-only, 1–2 grams, max 20k features, L2 LinearSVC | `baseline_tfidf_svm.json` |
| Qwen2.5-7B | description-only, zero-shot | `baseline_qwen.json` |
| E2E transformer | BGE + CLIP, fine-tuned, 5-fold | `baseline_e2e_transformer.json` |
| GPT-4o-mini | multimodal, zero-shot (`--all`) | `baseline_mllm_zeroshot_*.json` |
| GPT-4o | multimodal, 6-shot (`--all`) | `baseline_mllm_fewshot_*.json` |

### 5 — Inference on new apps

```bash
# From raw JSONL (runs full pipeline: preprocess → OCR → features → predict)
python src/run_inference.py --input-raw data/apps_inference_raw.jsonl --output results/

# From pre-extracted features
python src/run_inference.py --input data/features_test --output results/
```

Flags: `--skip-preprocessing`, `--skip-ocr`, `--keep-images`, `--keep-artifacts`.

Output per strategy: `early_fusion_inference.csv`, `stacking_inference.csv`, `soft_voting_inference.csv`, `score_max_inference.csv`.

---

## Annotation pipelines

### Inter-annotator agreement

```bash
# Requires: data/inter_annotator.csv
python src/run_iaa.py
# → runs/cohen_kappa/iaa.txt
```

### Code-level validation (Table 2)

Runs AI Discriminator on decompiled APKs and compares against listing labels and LLMDroid Early Fusion predictions.

```bash
# One-time: download Androzoo metadata (~3 GB)
wget https://androzoo.uni.lu/static/lists/latest.csv.gz
gunzip latest.csv.gz && mv latest.csv data/androzoo_latest.csv

export ANDROZOO_API_KEY=your_key

# Phase 1: code-validation set (N=80)
python src/run_code_validation.py --phases 1

# Phase 2: independent test set (N=110) + F1 comparison vs LLMDroid Early Fusion
python src/run_code_validation.py --phases 2

# Both phases
python src/run_code_validation.py --phases 1 2
```

Phase 2 requires `runs/feature_fusion/independent_test/predictions_early_fusion.csv` — run `independent_test_eval.py` first.

Checkpoints auto-saved to `data/code_validation_checkpoint.json` and `data/ai_disc_independent_checkpoint.json` — safe to interrupt and resume.

---

## Data files required

| File | Description |
|------|-------------|
| `data/apps_raw.jsonl` | Training apps with labels and image paths |
| `data/apps_inference_raw.jsonl` | Apps for inference / independent test set |
| `data/inference_manual.csv` | Ground-truth labels for 110-app test set (`pkg_name`, `label`) |
| `data/inter_annotator.csv` | IAA data (`app_id`, `annotator1`, `annotator2`) |
| `data/code_validation_apps.csv` | 80-app code-validation set (`pkg_name`, `listing_label`) |
| `data/androzoo_latest.csv` | Androzoo metadata for APK lookup (code-level validation only) |
| `data/images/{app_id}/*.png` | Screenshots (auto-downloaded by Step 0 if missing) |

---

## Output structure

```
runs/feature_fusion/
├── k_sensitivity/                ← SelectKBest k sweep
├── text_only/                    ← text-branch CV results
├── image_only/                   ← image-branch CV results
├── ablation/                     ← feature ablation (text modalities)
├── fusion/
│   ├── base_models_saved/        ← LightGBM models (×5 folds, text + image)
│   ├── early_fusion/
│   ├── late_fusion_stacking/
│   ├── late_fusion_soft_voting/
│   └── late_fusion_score_max/
├── independent_test/             ← predictions_*.csv, independent_test.json, baseline_*.json
├── statistical_tests/            ← summary.csv, results.json
├── branch_complementarity/
├── disagreement_accuracy/
├── per_category/
├── prior_correction/
├── temporal_split/
├── latency/
├── calibration/
├── robustness/
├── per_label_criterion/
├── image_analysis/
├── keyword_drift/
└── shap_analysis/

runs/cohen_kappa/
├── iaa.txt
└── validation.txt
```

---

## Reproduce from cached features

```bash
python src/train_pipeline.py --train-only
python src/steps/independent_test_eval.py
python src/run_analysis.py
python src/run_baselines.py
```
