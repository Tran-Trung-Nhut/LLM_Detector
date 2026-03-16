"""
Centralized configuration for V2 pipeline.
Reuses data paths and seed from V1; adds model configs for SBERT, CLIP, classifiers.
"""
from dataclasses import dataclass, field
import os
from typing import List


@dataclass(frozen=True)
class Config:
    # ── Reproducibility ──
    seed: int = 42
    n_folds: int = 5

    # ── Data ──
    raw_dataset_path: str = "data/apps_raw.jsonl"
    dataset_path: str = "data/apps.jsonl"
    splits_dir: str = "data/splits"

    # ── Text encoder ──
    # BGE-large works well for semantic similarity; fits easily on L4
    text_model: str = "BAAI/bge-large-en-v1.5"
    text_embed_dim: int = 1024
    text_batch_size: int = 32

    # ── Image encoder ──
    # CLIP ViT-L/14 @ 336px — higher resolution captures screenshot text
    clip_model: str = "openai/clip-vit-large-patch14-336"
    clip_embed_dim: int = 768
    clip_batch_size: int = 16

    # ── CLIP zero-shot prompts (carefully crafted for LLM detection) ──
    clip_positive_prompts: tuple = (
        "a screenshot of an AI chatbot conversation",
        "a mobile app with AI chat assistant interface",
        "a screenshot showing AI-generated text responses",
        "an app with a large language model powered chat",
        "a conversational AI interface on a phone screen",
    )
    clip_negative_prompts: tuple = (
        "a mobile app screenshot with no AI features",
        "a standard mobile application interface",
        "a photo editing or camera app screenshot",
        "a settings or profile page of a mobile app",
    )

    # ── Feature cache ──
    features_dir: str = "data/features_v2"

    # ── Classifier ──
    classifier_type: str = "lightgbm"   # "lightgbm" or "xgboost"
    fusion_strategy: str = "stacking"   # "stacking" or "max_voting" or "soft_voting"
    # LightGBM training parameters
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": None,
        "n_jobs": -1,
    })

    # ── Output ──
    runs_dir: str = "runs"
    run_name: str = "v2_feature_fusion"

    # ── Misc ──
    hf_token: str = os.environ.get("HF_TOKEN", None)


CFG = Config()
