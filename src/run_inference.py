import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import lightgbm as lgb
import json
import argparse


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils.io import write_predictions_csv
from utils.metrics import compute_binary_metrics

# ── TEST PATH CONFIGURATION ──
MODELS_DIR = Path(CFG.runs_dir) / CFG.run_name
OUTPUT_DIR = Path(CFG.inference_output_dir)

def load_test_features(features_dir: Path):
    """Load features from Hold-out test set."""
    print(f"Loading test features from {features_dir} ...")
    td = np.load(features_dir / "text" / "features.npz", allow_pickle=True)
    imd = np.load(features_dir / "image" / "features.npz", allow_pickle=True)
    slmd = np.load(features_dir / "slm" / "features.npz", allow_pickle=True)

    app_ids = list(td["app_ids"])
    
    labels = td["labels"]
    has_real_labels = len(np.unique(labels)) > 1 

    # Text
    text_feats = np.concatenate([td["sbert"], td["keywords"], td["meta"], slmd["slm_score"]], axis=1)
    # Image
    image_feats = np.concatenate([imd["clip_mean"], imd["clip_max"], imd["zeroshot"], imd["ocr"]], axis=1)
    # All
    all_feats = np.concatenate([text_feats, image_feats], axis=1)

    assert list(td["app_ids"]) == list(imd["app_ids"]), "Feature files must have same app order"
    
    return app_ids, labels, text_feats, image_feats, all_feats, has_real_labels


def print_and_save_report(name, app_ids, y_true, y_prob, threshold, output_csv, has_real_labels=True):
    """Calculate metrics (if labels exist), print report and save CSV file."""
    print(f"\n[{name}] Threshold = {threshold}")
    
    # Create prediction results CSV
    preds = []
    for i, aid in enumerate(app_ids):
        is_llm = int(y_prob[i] >= threshold)
        row = {
            "app_id": aid,
            "y_prob": float(y_prob[i]),
            "prediction_label": is_llm,
        }
        if has_real_labels:
            row["y_true"] = int(y_true[i])
            row["correct"] = int(is_llm == int(y_true[i]))
        preds.append(row)
    
    write_predictions_csv(output_csv, preds)
    print(f"  -> Saved detailed predictions to: {output_csv}")

    # Only calculate metrics if real labels are available
    if has_real_labels:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        print(f"  Accuracy  : {metrics['accuracy']:.3f}")
        print(f"  Precision : {metrics['precision_pos']:.3f}")
        print(f"  Recall    : {metrics['recall_pos']:.3f}")
        print(f"  F1-Score  : {metrics['f1_pos']:.3f}")
    else:
        print("  [INFO] No ground truth labels found for accuracy evaluation.")
        print(f"  System predicted {sum([p['prediction_label'] for p in preds])} apps with LLM integration.")

def ensemble_early_fusion(X_all, num_folds=None):
    """Predict by averaging predictions from all folds of Early Fusion."""
    if num_folds is None:
        num_folds = CFG.n_folds
    early_fusion_dir = MODELS_DIR / "early_fusion" / "saved_models"
    fold_probs = []

    for fold in range(num_folds):
        # 1. Load SelectKBest
        selector = joblib.load(early_fusion_dir / f"selector_fold_{fold}.joblib")
        X_sel = selector.transform(X_all)
        
        # 2. Load LightGBM
        lgbm = lgb.Booster(model_file=str(early_fusion_dir / f"lgbm_fold_{fold}.txt"))
        
        # 3. Predict
        prob = lgbm.predict(X_sel)
        fold_probs.append(prob)
        
    # Average (Ensemble)
    ensemble_prob = np.mean(fold_probs, axis=0)
    return ensemble_prob

def ensemble_late_fusion_stacking(X_text, X_image, num_folds=None):
    """Predict using Late Fusion - Stacking strategy across all folds."""
    if num_folds is None:
        num_folds = CFG.n_folds
    base_models_dir = MODELS_DIR / "fusion" / "base_models_saved"
    stacking_dir = MODELS_DIR / "fusion" / "late_fusion_stacking" / "saved_models"
    
    fold_probs = []

    for fold in range(num_folds):
        # -- TEXT BRANCH --
        text_selector = joblib.load(base_models_dir / f"text_selector_fold_{fold}.joblib")
        X_text_sel = text_selector.transform(X_text)
        text_lgbm = lgb.Booster(model_file=str(base_models_dir / f"text_lgbm_fold_{fold}.txt"))
        text_prob = text_lgbm.predict(X_text_sel)

        # -- IMAGE BRANCH --
        img_selector = joblib.load(base_models_dir / f"img_selector_fold_{fold}.joblib")
        X_img_sel = img_selector.transform(X_image)
        img_lgbm = lgb.Booster(model_file=str(base_models_dir / f"img_lgbm_fold_{fold}.txt"))
        img_prob = img_lgbm.predict(X_img_sel)

        # -- STACKING (META-LEARNER) --
        meta_clf = joblib.load(stacking_dir / f"meta_clf_fold_{fold}.joblib")
        scaler = joblib.load(stacking_dir / f"scaler_fold_{fold}.joblib")
        
        X_meta = np.column_stack([text_prob, img_prob])
        X_meta_scaled = scaler.transform(X_meta)
        
        final_prob = meta_clf.predict_proba(X_meta_scaled)[:, 1]
        fold_probs.append(final_prob)

    # Average Meta-learner predictions
    ensemble_prob = np.mean(fold_probs, axis=0)
    return ensemble_prob



def get_optimal_threshold(model_dir: Path, default_threshold: float = None) -> float:
    """Automatically find and read optimal threshold from JSON file saved during training."""
    if default_threshold is None:
        default_threshold = CFG.inference_default_threshold
    json_path = model_dir / "best_threshold_metrics.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            best_t = float(data.get("best_threshold", default_threshold))
            print(f"Loaded optimal threshold {best_t:.2f} from {json_path.parent.name}")
            return best_t
    else:
        print(f"[WARNING] File {json_path} not found. Using default: {default_threshold}")
        return default_threshold

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference on test dataset for LLM detection")
    parser.add_argument(
        "--file_path", "-f",
        type=str,
        default=CFG.inference_test_features_dir,
        help=f"Path to test features directory (default: {CFG.inference_test_features_dir})"
    )
    args = parser.parse_args()
    
    # Use command-line argument or default from config
    TEST_FEATURES_DIR = Path(args.file_path)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  INDEPENDENT HOLD-OUT EVALUATION   ")
    print("=" * 60)

    if not TEST_FEATURES_DIR.exists():
        print(f"[ERROR] Test features directory not found at: {TEST_FEATURES_DIR}")
        print("Please run extract_text/image/slm_features.py on the dataset first!")
        return

    # 1. Load Data
    app_ids, labels, text_feats, image_feats, all_feats, has_real_labels = load_test_features(TEST_FEATURES_DIR)
    print(f"Loaded {len(app_ids)} test apps.")

    # ── AUTO-LOAD OPTIMAL THRESHOLD FROM JSON FILE ──
    early_fusion_dir = MODELS_DIR / "fusion" / "early_fusion"
    stacking_dir = MODELS_DIR / "fusion" / "late_fusion_stacking"
    
    ef_opt_threshold = get_optimal_threshold(early_fusion_dir)
    stack_opt_threshold = get_optimal_threshold(stacking_dir)

    # 2. Run Early Fusion Inference
    ef_prob = ensemble_early_fusion(all_feats)
    print_and_save_report(
        name="EARLY FUSION (Recall Optimized)", 
        app_ids=app_ids, 
        y_true=labels, 
        y_prob=ef_prob, 
        threshold=ef_opt_threshold,
        output_csv=OUTPUT_DIR / "early_fusion_inference.csv",
        has_real_labels=has_real_labels
    )

    # 3. Run Late Fusion - Stacking Inference
    stack_prob = ensemble_late_fusion_stacking(text_feats, image_feats)
    print_and_save_report(
        name="STACKING (Precision Optimized)", 
        app_ids=app_ids, 
        y_true=labels, 
        y_prob=stack_prob, 
        threshold=stack_opt_threshold,
        output_csv=OUTPUT_DIR / "stacking_inference.csv",
        has_real_labels=has_real_labels
    )

    print("\n" + "=" * 60)
    print("Hold-out evaluation completed!")

if __name__ == "__main__":
    main()