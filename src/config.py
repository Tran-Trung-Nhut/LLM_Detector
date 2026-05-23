from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_folds: int = 5

    raw_dataset_path: str = "data/apps_raw.jsonl"
    raw_inference_dataset_path: str = "data/apps_inference_raw.jsonl"
    inference_dataset_path: str = "data/apps_inference.jsonl"
    dataset_path: str = "data/apps.jsonl"
    splits_dir: str = "data/splits"
    images_dir: str = "data/images"

    inference_apps_csv_path: str = "data/inference_apps.csv"
    failed_apps_filename: str = "failed_apps.txt"

    androzoo_api_url: str = "https://androzoo.uni.lu/api/search"
    androzoo_api_timeout: int = 10
    gplay_lang: str = "en"
    gplay_country: str = "us"
    api_request_delay: float = 1.0
    screenshot_download_timeout: int = 10
    max_failed_apps_display: int = 10
    image_format: str = "png"

    text_model: str = "BAAI/bge-large-en-v1.5"
    text_embed_dim: int = 1024
    text_batch_size: int = 32
    text_max_length: int = 512

    slm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    slm_max_new_tokens: int = 10
    slm_batch_size: int = 8
    slm_text_max_length: int = 1500
    slm_system_prompt: str = "You are an expert AI software architecture reviewer."
    slm_user_prompt_template: str = """Analyze the following Android application description. Does this app integrate a Large Language Model (LLM) like ChatGPT, GPT-4, Claude, or similar AI chat technologies? 
Consider implied features like 'conversational agent', 'smart ai writer', or 'ai chat'.

App Description:
{text}

Respond ONLY with a single confidence score between 0 and 100 indicating the probability of LLM integration. Do not output any other text, explanations, or words.
Score:"""

    # CLIP ViT-L/14 @ 336px — higher resolution captures on-screen text in screenshots
    clip_model: str = "openai/clip-vit-large-patch14-336"
    clip_embed_dim: int = 768
    clip_batch_size: int = 16

    clip_positive_prompts: tuple = (
        "a screenshot of a chatbot conversation interface",
        "a screenshot of an AI assistant",
        "a mobile app conversation between user and AI",
        "a chat bubble interface for messaging an AI",
        "a text input box at the bottom of a chat thread",
    )
    clip_negative_prompts: tuple = (
        "a screenshot of a settings or preferences page",
        "a screenshot of a calendar or scheduling app",
        "a screenshot of a photo gallery or media viewer",
        "a screenshot of a login or onboarding screen",
        "a screenshot of a list of products in an e-commerce app",
    )

    features_dir: str = "data/features"

    classifier_type: str = "lightgbm"
    fusion_strategy: list[str] = field(default_factory=lambda: ["stacking", "score_max", "soft_voting"])

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
    
    feature_selection_k: int = 200
    soft_voting_alpha_candidates: tuple = (0.3, 0.4, 0.5, 0.6, 0.7)
    k_sensitivity_values: tuple = (20, 50, 100, 200, 500)
    inner_val_ratio: float = 0.2
    stacking_inner_cv_folds: int = 4
    classification_threshold: float = 0.5
    threshold_search_min: float = 0.01
    threshold_search_max: float = 0.99
    threshold_search_step: float = 0.01

    meta_learner_C: float = 1.0
    meta_learner_max_iter: int = 1000

    image_dedup_max_dist: int = 4
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

    ocr_lang: str = "eng"

    keywords: tuple = (
        "chatgpt", "gpt-4", "gpt-3", "claude", "gemini", "copilot", "llama", "mistral",
        "llm", "large language model", "ai chat", "chatbot", "ai assistant",
        "generate text", "text generation", "ai writing", "ai writer", "ai compose",
        "ai draft", "smart reply", "auto-reply", "rewrite", "paraphrase",
        "summar", "ai summary",
        "ask ai", "ai answer", "talk to ai", "chat with ai", "prompt",
        "conversational ai", "ai-powered chat", "ai response",
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

    iaa_csv: str = "data/inter_annotator.csv"
    code_validation_csv: str = "data/code_validation.csv"
    ai_disc_independent_csv: str = "data/ai_disc_independent.csv"
    n_bootstrap: int = 2000
    disagree_threshold: float = 0.3
    prior_pi_values: tuple = (0.005, 0.01, 0.05)
    ece_n_bins: int = 10

    inference_manual_csv: str = "data/inference_manual.csv"
    features_test_dir: str = "data/features_test"
    features_test_trunc50_dir: str = "data/features_test_trunc50"
    trunc_desc_chars: int = 50
    temporal_d_cut: str = "2026-04-30"

    latency_n_screenshots: int = 4
    latency_n_runs: int = 3

    runs_dir: str = "runs"
    run_name: str = "feature_fusion"

    inference_features_dir: str = "data/features_test"
    inference_output_dir: str = "inference_results"
    inference_default_threshold: float = 0.5

    hf_token: str = os.environ.get("HF_TOKEN", None)


CFG = Config()
