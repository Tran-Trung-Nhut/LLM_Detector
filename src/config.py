from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    # -------------------------
    # Repro / folds
    # -------------------------
    seed: int = 42
    n_folds: int = 5

    # -------------------------
    # Data
    # -------------------------
    raw_dataset_path: str = "data/apps_raw.jsonl"
    dataset_path: str = "data/apps.jsonl"
    splits_dir: str = "data/splits"

    # -------------------------
    # Model
    # -------------------------
    base_model: str = "google/paligemma-3b-pt-224"

    # -------------------------
    # Training (single-image LoRA)
    # -------------------------
    num_epochs: int = 5
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_text_len: int = 1024 + 512

    # Which single image to pick per app during training/eval:
    # - "best": OCR-based if available, else middle image
    # - "first": first screenshot
    # - "random": random screenshot
    image_strategy: str = "best"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Output dirs
    runs_dir: str = "runs"
    train_run_name: str = "paligemma_single_image"   # adapters saved per fold under runs_dir/train_run_name/fold_{i}
    infer_run_name: str = "paligemma_multi_image"    # predictions saved per fold under runs_dir/infer_run_name/fold_{i}

    # -------------------------
    # Multi-image inference
    # -------------------------
    # pooling over per-image probabilities:
    # - "max" is usually best for "evidence in any screenshot"
    # - "mean" is more conservative
    multi_image_pooling: str = "max"

    # API Config
    hf_token: str = os.environ.get("HF_TOKEN", None)


CFG = Config()