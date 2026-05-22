"""report_paper_results.py — Persist paper-published result tables to JSON for downstream comparison."""
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json


def main():
    out_dir = Path(CFG.runs_dir) / CFG.run_name / "paper_reported_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    table7 = {
        "note": "5-fold CV, threshold=0.5, Table 7 of paper",
        "Text-Only":              {"accuracy": 0.809, "precision": 0.816, "recall": 0.704, "f1": 0.753, "roc_auc": 0.872, "pr_auc": 0.869},
        "Image-Only":             {"accuracy": 0.742, "precision": 0.741, "recall": 0.592, "f1": 0.654, "roc_auc": 0.825, "pr_auc": 0.776},
        "Early Fusion":           {"accuracy": 0.792, "precision": 0.793, "recall": 0.688, "f1": 0.732, "roc_auc": 0.873, "pr_auc": 0.867},
        "Late Fusion - Score-Max":{"accuracy": 0.792, "precision": 0.733, "recall": 0.808, "f1": 0.766, "roc_auc": 0.894, "pr_auc": 0.872},
        "Late Fusion - Soft Voting":{"accuracy": 0.809,"precision": 0.881,"recall": 0.632, "f1": 0.727, "roc_auc": 0.904, "pr_auc": 0.886},
        "Late Fusion - Stacking": {"accuracy": 0.802, "precision": 0.809, "recall": 0.696, "f1": 0.742, "roc_auc": 0.902, "pr_auc": 0.885},
    }
    write_json(out_dir / "table7_cv_results.json", table7)

    table8 = {
        "note": "Threshold-optimized on pooled inner-validation (N=240), Table 8 of paper",
        "Text-Only":    {"threshold": 0.41, "accuracy": 0.842, "precision": 0.816, "recall": 0.800, "f1": 0.808, "roc_auc": 0.906, "pr_auc": 0.898},
        "Image-Only":   {"threshold": 0.41, "accuracy": 0.804, "precision": 0.773, "recall": 0.750, "f1": 0.761, "roc_auc": 0.850, "pr_auc": 0.814},
        "Early Fusion": {"threshold": 0.58, "accuracy": 0.879, "precision": 0.928, "recall": 0.770, "f1": 0.842, "roc_auc": 0.929, "pr_auc": 0.923},
        "Score-Max":    {"threshold": 0.48, "accuracy": 0.850, "precision": 0.786, "recall": 0.880, "f1": 0.830, "roc_auc": 0.918, "pr_auc": 0.895},
        "Soft Voting":  {"threshold": 0.38, "accuracy": 0.867, "precision": 0.833, "recall": 0.850, "f1": 0.842, "roc_auc": 0.925, "pr_auc": 0.908},
        "Stacking":     {"threshold": 0.36, "accuracy": 0.858, "precision": 0.806, "recall": 0.870, "f1": 0.837, "roc_auc": 0.924, "pr_auc": 0.898},
    }
    write_json(out_dir / "table8_threshold_optimized.json", table8)

    table12 = {
        "note": "Independent test set N=110 (44 positives). Table 12 of paper. Recall CI is Clopper-Pearson 95%.",
        "External baselines": {
            "Qwen2.5-7B (desc only, 0-shot)":        {"accuracy": 0.755, "precision": 0.692, "recall": 0.614, "recall_ci": [0.456, 0.756], "f1": 0.651, "roc_auc": 0.812, "ap": 0.731},
            "GPT-4o-mini (multimodal, 0-shot)":       {"accuracy": 0.800, "precision": 0.795, "recall": 0.745, "recall_ci": [0.595, 0.861], "f1": 0.769, "roc_auc": 0.872, "ap": 0.788},
            "Gemini-1.5-Flash (multimodal, 0-shot)":  {"accuracy": 0.791, "precision": 0.683, "recall": 0.795, "recall_ci": [0.647, 0.901], "f1": 0.735, "roc_auc": 0.864, "ap": 0.776},
            "GPT-4o (multimodal, 6-shot)":            {"accuracy": 0.818, "precision": 0.864, "recall": 0.727, "recall_ci": [0.574, 0.848], "f1": 0.792, "roc_auc": 0.902, "ap": 0.831},
            "E2E multimodal transformer (BGE+CLIP+xattn)": {"accuracy": 0.755, "precision": 0.682, "recall": 0.682, "recall_ci": [0.524, 0.815], "f1": 0.682, "roc_auc": 0.853, "ap": 0.776},
        },
        "LLMDroid (ours)": {
            "Text-Only":    {"accuracy": 0.836, "precision": 0.756, "recall": 0.818, "recall_ci": [0.673, 0.918], "f1": 0.786, "roc_auc": 0.918, "ap": 0.842},
            "Score-Max":    {"accuracy": 0.836, "precision": 0.717, "recall": 0.977, "recall_ci": [0.877, 0.999], "f1": 0.827, "roc_auc": 0.918, "ap": 0.870},
            "Soft Voting":  {"accuracy": 0.864, "precision": 0.754, "recall": 0.977, "recall_ci": [0.877, 0.999], "f1": 0.851, "roc_auc": 0.930, "ap": 0.846},
            "Stacking":     {"accuracy": 0.855, "precision": 0.741, "recall": 0.977, "recall_ci": [0.877, 0.999], "f1": 0.843, "roc_auc": 0.937, "ap": 0.859},
            "Early Fusion": {"accuracy": 0.780, "precision": 0.800, "recall": 0.886, "recall_ci": [0.755, 0.961], "f1": 0.830, "roc_auc": 0.938, "ap": 0.870},
        },
    }
    write_json(out_dir / "table12_external_baselines.json", table12)

    table19 = {
        "note": "LLMAID code-level validation on N_code=80 apps. Table 19 of paper. LLMAID is an external tool (Liu et al.).",
        "Listing label":         {"accuracy": 0.875, "precision": 0.886, "recall": 0.886, "f1": 0.886},
        "LLMAID code-level label": {"accuracy": 0.862, "precision": 0.864, "recall": 0.864, "f1": 0.864},
        "Difference |Δ|":        {"accuracy": 0.013, "precision": 0.022, "recall": 0.022, "f1": 0.022},
        "interpretation": "F1 gap 0.022 is within pre-specified 'small gap' threshold of 0.05. Listing label is reliable.",
    }
    write_json(out_dir / "table19_llmaid_validation.json", table19)

    table16 = {
        "note": "Mean wall-clock time per app on 110-app test set. API cost at published token rates. Table 16 of paper.",
        "LLMDroid components": {
            "BGE-large-en-v1.5 encode":      {"latency_s": 0.013, "api_cost_usd": None},
            "CLIP ViT-L/14 (4 screenshots)": {"latency_s": 0.097, "api_cost_usd": None},
            "OCR Tesseract (4 screenshots)": {"latency_s": 0.082, "api_cost_usd": None},
            "LightGBM inference":            {"latency_s": 0.001, "api_cost_usd": None},
            "Fusion + threshold":            {"latency_s": 0.001, "api_cost_usd": None},
            "LLMDroid (full pipeline)":      {"latency_s": 0.21,  "api_cost_usd": 0.000},
        },
        "External baselines": {
            "Qwen2.5-7B (desc only, 0-shot)":       {"latency_s": 0.81, "api_cost_usd": 0.000, "note": "local"},
            "GPT-4o-mini (multimodal, 0-shot)":      {"latency_s": 2.46, "api_cost_usd": 0.0017},
            "Gemini-1.5-Flash (multimodal, 0-shot)": {"latency_s": 2.21, "api_cost_usd": 0.0013},
            "GPT-4o (multimodal, 6-shot)":           {"latency_s": 5.47, "api_cost_usd": 0.0420},
            "E2E multimodal transformer":             {"latency_s": 0.08, "api_cost_usd": 0.000},
        },
    }
    write_json(out_dir / "table16_latency.json", table16)

    table3 = {
        "note": "Text branch ablation. 5-fold CV, k=200. Table 3 of paper.",
        "BGE only":              {"dims": 1024, "roc_auc": "0.8079±0.0695", "f1": "0.6820±0.0842"},
        "Handcrafted only":      {"dims": 34,   "roc_auc": "0.7904±0.0494", "f1": "0.7020±0.0532"},
        "SLM only":              {"dims": 1,    "roc_auc": "0.5227±0.0421", "f1": "0.0000±0.0000"},
        "BGE + Handcrafted":     {"dims": 1058, "roc_auc": "0.8723±0.0343", "f1": "0.7531±0.0499"},
        "BGE + SLM":             {"dims": 1025, "roc_auc": "0.8079±0.0695", "f1": "0.6820±0.0842"},
        "Handcrafted + SLM":     {"dims": 35,   "roc_auc": "0.7762±0.0589", "f1": "0.6832±0.0883"},
        "BGE + Handcrafted + SLM": {"dims": 1059, "roc_auc": "0.8723±0.0343", "f1": "0.7531±0.0499"},
    }
    write_json(out_dir / "table3_text_ablation.json", table3)

    table5 = {
        "note": "Image branch leave-one-out ablation. 5-fold CV, k=200. Table 5 of paper.",
        "Full image branch (CLIP + chat + OCR)": {"dims": 1552, "roc_auc": "0.8246±0.0413", "delta_roc_auc": None},
        "-CLIP pooled embeddings":               {"dims": 16,   "roc_auc": "0.7321±0.0497", "delta_roc_auc": -0.092},
        "-Zero-shot chatbot-UI score":           {"dims": 1551, "roc_auc": "0.8073±0.0432", "delta_roc_auc": -0.017},
        "-OCR keyword features":                 {"dims": 1537, "roc_auc": "0.7958±0.0451", "delta_roc_auc": -0.029},
    }
    write_json(out_dir / "table5_image_ablation.json", table5)

    table15 = {
        "note": "Temporal split D_cut=2025-06-01. Training=218 apps, Test=82 apps (33 positives). Table 15 of paper.",
        "Text-Only":    {"random_f1": 0.753, "temporal_f1": 0.708, "delta": -0.045, "roc_auc_temporal": 0.841},
        "Early Fusion": {"random_f1": 0.732, "temporal_f1": 0.694, "delta": -0.038, "roc_auc_temporal": 0.851},
        "Score-Max":    {"random_f1": 0.766, "temporal_f1": 0.726, "delta": -0.040, "roc_auc_temporal": 0.868},
        "Stacking":     {"random_f1": 0.742, "temporal_f1": 0.705, "delta": -0.037, "roc_auc_temporal": 0.866},
        "Soft Voting":  {"random_f1": 0.727, "temporal_f1": 0.692, "delta": -0.035, "roc_auc_temporal": 0.868},
    }
    write_json(out_dir / "table15_temporal.json", table15)

    table17 = {
        "note": "Brier score and ECE on pooled CV held-out predictions (N=300). Table 17 of paper.",
        "Text-Only":    {"brier_raw": 0.158, "ece_raw": 0.094, "brier_platt": 0.149, "ece_platt": 0.041},
        "Early Fusion": {"brier_raw": 0.156, "ece_raw": 0.098, "brier_platt": 0.146, "ece_platt": 0.039},
        "Score-Max":    {"brier_raw": 0.171, "ece_raw": 0.121, "brier_platt": 0.158, "ece_platt": 0.052},
        "Soft Voting":  {"brier_raw": 0.139, "ece_raw": 0.082, "brier_platt": 0.133, "ece_platt": 0.034},
        "Stacking":     {"brier_raw": 0.142, "ece_raw": 0.085, "brier_platt": 0.135, "ece_platt": 0.036},
    }
    write_json(out_dir / "table17_calibration_paper.json", table17)

    table18 = {
        "note": "Soft Voting on independent test set (N=110, 44 positives). Table 18 of paper.",
        "Full listing (baseline)":              {"recall": 0.977, "precision": 0.754, "f1": 0.851, "delta_f1": None},
        "Drop screenshots":                     {"recall": 0.864, "precision": 0.745, "f1": 0.800, "delta_f1": -0.051},
        "Truncate description to 50 chars":     {"recall": 0.795, "precision": 0.660, "f1": 0.722, "delta_f1": -0.129},
        "Drop screenshots and truncate text":   {"recall": 0.682, "precision": 0.612, "f1": 0.645, "delta_f1": -0.206},
    }
    write_json(out_dir / "table18_robustness_paper.json", table18)

    print("Paper results saved to:", out_dir)
    for f in sorted(out_dir.glob("*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
