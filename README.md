# LLM Detector

Multimodal LLM integration detection in Android apps using text (SBERT + keywords + SLM reasoning) and image (CLIP + OCR) features with Early/Late Fusion strategies.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train Pipeline (Full CV)
```bash
python src/train_pipeline.py
```

Runs complete pipeline:
- Preprocessing (text cleaning, image deduplication)
- Create 5-fold stratified splits
- OCR extraction from screenshots
- Extract text features (SBERT embeddings, keywords, metadata)
- Extract image features (CLIP embeddings, zero-shot classification, OCR)
- Extract SLM reasoning scores (Qwen2.5-1.5B)
- Train & evaluate Early Fusion + Late Fusion (Stacking, Max Voting, Soft Voting)

### 2. Inference on New Data
```bash
# Use default test set from config
python src/run_inference.py

# Or specify custom test features directory
python src/run_inference.py --test_features_dir ./data/features_test_100
```

## Configuration

Edit `src/config.py` to customize:
- Models: Text encoder (SBERT), Image encoder (CLIP), SLM (Qwen/Gemma)
- Hyperparameters: LightGBM params, feature selection, thresholds
- Paths: Data, features, runs, inference directories

## Project Structure

```
LLM_Detector/
├── data/
│   ├── apps_raw.jsonl          # Raw dataset
│   ├── apps.jsonl              # Preprocessed dataset
│   ├── images/                 # App screenshots
│   ├── splits/                 # CV fold splits
│   └── features_v2/            # Cached features
├── src/
│   ├── config.py               # Central configuration
│   ├── train_pipeline.py       # Full training pipeline
│   ├── run_inference.py        # Inference script
│   ├── steps/                  # Pipeline steps
│   │   ├── preprocessing.py
│   │   ├── make_splits.py
│   │   ├── run_ocr.py
│   │   ├── extract_text_features.py
│   │   ├── extract_image_features.py
│   │   ├── extract_slm_features.py
│   │   └── train_evaluate.py
│   └── utils/                  # Utilities
│       ├── io.py
│       ├── metrics.py
│       └── seed.py
├── runs/                       # Training outputs
└── inference_results/          # Inference predictions
```

## Features

**Text Features:**
- SBERT embeddings (BAAI/bge-large-en-v1.5)
- Keyword matching (LLM-related terms)
- Metadata (category, ratings, installs)
- SLM reasoning scores (Qwen2.5-1.5B)

**Image Features:**
- CLIP embeddings (mean, max pooling)
- Zero-shot classification (LLM vs non-LLM UI)
- OCR text extraction

**Fusion Strategies:**
- Early Fusion: Concatenate all features → LightGBM
- Late Fusion (Stacking): Text/Image branches → Meta-learner
- Late Fusion (Max/Soft Voting): Ensemble predictions

## Results

Output CSV includes:
- `app_id`: Application identifier
- `y_prob`: Prediction probability
- `prediction_label`: Binary label (0/1)
- `y_true`: Ground truth (if available)
- `correct`: Prediction correctness (if labels available)