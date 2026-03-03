# LLM Detector

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python -m src.train_paligemma_lora_single_image \
  --dataset_path data/apps.jsonl \
  --split_path data/splits/fold_0.json \
  --run_dir runs/paligemma_single_image/fold_0
```

## Project Structure

```
project/
├── data/
│   ├── apps.jsonl
│   ├── images/
│   └── splits/
│       ├── fold_0.json
│       ├── fold_1.json
│       ├── fold_2.json
│       ├── fold_3.json
│       └── fold_4.json
├── src/
│   ├── config.py
│   ├── utils_seed.py
│   ├── utils_io.py
│   ├── utils_text.py
│   ├── utils_metrics.py
│   ├── make_splits.py
│   ├── dataset_apps.py
│   ├── prompts.py
│   ├── train_paligemma_lora_single_image.py
│   ├── infer_paligemma_multi_image.py
│   └── run_cv.py
├── runs/
├── requirements.txt
└── README.md
```