"""
Centralized configuration for V2 pipeline.
Reuses data paths and seed from V1; adds model configs for SBERT, CLIP, classifiers.
All constants are centralized here for easier maintenance.
"""
from dataclasses import dataclass, field
import os


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
    text_max_length: int = 512  # Tokenizer max length

    # ── SLM Reasoning Module ──
    # Recommend Qwen2.5-1.5B or Gemma-2-2B as they run smoothly on L4 GPU with excellent reasoning
    slm_model: str = "Qwen/Qwen2.5-1.5B-Instruct" 
    slm_max_new_tokens: int = 10
    slm_batch_size: int = 8
    slm_text_max_length: int = 1500  # Max characters from description for SLM input
    
    # SLM prompts for LLM detection reasoning
    slm_system_prompt: str = "You are an expert AI software architecture reviewer."
    slm_user_prompt_template: str = """Analyze the following Android application description. Does this app integrate a Large Language Model (LLM) like ChatGPT, GPT-4, Claude, or similar AI chat technologies? 
Consider implied features like 'conversational agent', 'smart ai writer', or 'ai chat'.

App Description:
{text}

Respond ONLY with a single confidence score between 0 and 100 indicating the probability of LLM integration. Do not output any other text, explanations, or words.
Score:"""

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
    features_dir: str = "data/features"

    # ── Classifier ──
    classifier_type: str = "lightgbm"   # "lightgbm" or "xgboost"
    fusion_strategy: list[str] = field(default_factory=lambda: ["stacking", "max_voting", "soft_voting"])   # "stacking" or "max_voting" or "soft_voting"
    
    # LightGBM training parameters
    lgbm_num_rounds: int = 500
    lgbm_early_stopping_rounds: int = 50
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
    
    # Feature selection
    feature_selection_k: int = 50  # Number of top features to select
    classification_threshold: float = 0.5  # Binary classification threshold
    
    # Meta-learner (stacking fusion)
    meta_learner_C: float = 1.0
    meta_learner_max_iter: int = 1000

    # ── Preprocessing ──
    image_dedup_max_dist: int = 4  # Max hash distance for image deduplication
    footer_markers: tuple = (
        r"privacy\s*policy",
        r"terms\s*(of\s*(use|service))?",
        r"contact\s*us",
        r"follow\s*us",
        r"connect\s*with\s*us",
        r"subscriptions?",
        r"in[-\s]*app\s*purchases?",
        r"need\s*help",
        r"feedback",
        r"refund",
    )

    # ── OCR ──
    ocr_lang: str = "eng"

    # ── Keywords (LLM-related) ──
    keywords: tuple = (
        # Model names
        "chatgpt", "gpt-4", "gpt-3", "claude", "gemini", "copilot", "llama", "mistral",
        # Core LLM terms
        "llm", "large language model", "ai chat", "chatbot", "ai assistant",
        # Generation features
        "generate text", "text generation", "ai writing", "ai writer", "ai compose",
        "ai draft", "smart reply", "auto-reply", "rewrite", "paraphrase",
        "summar", "ai summary",
        # Interaction patterns
        "ask ai", "ai answer", "talk to ai", "chat with ai", "prompt",
        "conversational ai", "ai-powered chat", "ai response",
        # Content creation
        "content generat", "essay generator", "article generator", "story generator",
        "ai copywriting", "ai content",
    )
    
    keyword_categories: dict = field(default_factory=lambda: {
        "model_name": ["chatgpt", "gpt-4", "gpt-3", "claude", "gemini", "copilot", "llama", "mistral"],
        "core_llm": ["llm", "large language model", "ai chat", "chatbot", "ai assistant"],
        "generation": ["generate text", "text generation", "ai writing", "ai writer", "ai compose",
                       "ai draft", "smart reply", "auto-reply", "rewrite", "paraphrase", "summar", "ai summary"],
        "interaction": ["ask ai", "ai answer", "talk to ai", "chat with ai", "prompt",
                        "conversational ai", "ai-powered chat", "ai response"],
        "content": ["content generat", "essay generator", "article generator", "story generator",
                    "ai copywriting", "ai content"],
    })
    
    top_categories: tuple = (
        "Education", "Communication", "Business", "Productivity", "Health & Fitness",
        "Tools", "Entertainment", "Lifestyle", "Social", "Finance",
        "Shopping", "Travel & Local", "Medical", "Music & Audio", "Photography",
    )

    # ── Output ──
    runs_dir: str = "runs"
    run_name: str = "feature_fusion"

    # ── Inference ──
    inference_test_features_dir: str = "data/features_test"
    inference_output_dir: str = "inference_results"
    inference_default_threshold: float = 0.5

    # ── Misc ──
    hf_token: str = os.environ.get("HF_TOKEN", None)


CFG = Config()
