#!/usr/bin/env python3
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils.runner import run_step, print_summary


def main():
    results = {}

    results["6.1 Statistical tests"] = run_step(
        "6.1 — Statistical tests (Table 9)",
        "steps.statistical_tests",
    )
    results["6.2 Branch complementarity"] = run_step(
        "6.2 — Branch complementarity (Table 10 + Figure 4)",
        "steps.branch_complementarity",
    )
    results["6.3 Disagreement accuracy"] = run_step(
        "6.3 — Disagreement accuracy (Table 11)",
        "steps.disagreement_accuracy",
    )
    results["6.4 Per-category"] = run_step(
        "6.4 — Per-category performance (Table 13)",
        "steps.per_category",
    )
    results["6.5 Prior correction"] = run_step(
        "6.5 — Prior-corrected precision (Table 14 + Figure 5)",
        "steps.prior_correction",
    )
    results["6.6 Temporal split"] = run_step(
        "6.6 — Temporal split (Table 15)",
        "steps.temporal_split",
    )
    results["6.7 Latency benchmark"] = run_step(
        "6.7 — Inference latency (Table 16)",
        "steps.latency_benchmark",
    )
    results["6.8 Calibration"] = run_step(
        "6.8 — Probability calibration (Table 17)",
        "steps.calibration",
    )

    trunc50_path = Path(CFG.features_test_trunc50_dir) / "text" / "features.npz"
    if not trunc50_path.exists():
        run_step(
            "6.9 — Extract truncated text features (pre-step)",
            "steps.extract_text_features_trunc50",
        )
    results["6.9 Robustness"] = run_step(
        "6.9 — Robustness to missing modalities (Table 18)",
        "steps.robustness",
    )

    results["6.10 Per-label-criterion"] = run_step(
        "6.10 — Per-label-criterion evaluation (construct validity)",
        "steps.per_label_criterion",
    )
    results["6.11 Image analysis"] = run_step(
        "6.11 — Image branch helps/hurts qualitative analysis",
        "steps.image_analysis_examples",
    )
    results["6.12 Keyword drift"] = run_step(
        "6.12 — Keyword-drift robustness (remove model-name keywords)",
        "steps.keyword_drift",
    )
    results["6.13 SHAP analysis"] = run_step(
        "6.13 — SHAP feature importance (Text-Only + Early Fusion)",
        "steps.shap_analysis",
    )

    print_summary(results)


if __name__ == "__main__":
    main()
