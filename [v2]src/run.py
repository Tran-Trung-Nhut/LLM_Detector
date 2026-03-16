"""
run.py — One-command orchestrator for the full V2 pipeline.

Usage (from project root):
    python "[v2]src/run.py"                 # Full pipeline
    python "[v2]src/run.py" --skip-ocr      # Skip OCR step
    python "[v2]src/run.py" --skip-features # Skip feature extraction
    python "[v2]src/run.py" --train-only    # Only train & evaluate
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is CWD and [v2]src is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG


def step_preprocess():
    """Step 1: Clean text and deduplicate images."""
    if Path(CFG.dataset_path).exists():
        print(f"[skip] {CFG.dataset_path} already exists")
        return
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)
    import preprocessing
    preprocessing.main()


def step_make_splits():
    """Step 2: Create stratified k-fold splits."""
    split_file = Path(CFG.splits_dir) / f"fold_{CFG.n_folds - 1}.json"
    if split_file.exists():
        print(f"[skip] Splits already exist in {CFG.splits_dir}")
        return
    print("\n" + "=" * 60)
    print("STEP 2: Create Splits")
    print("=" * 60)
    import make_splits
    make_splits.main()


def step_ocr():
    """Step 3: Run Tesseract OCR on screenshots."""
    print("\n" + "=" * 60)
    print("STEP 3: OCR")
    print("=" * 60)
    import run_ocr
    run_ocr.main()


def step_extract_text_features():
    """Step 4a: Extract text features (SBERT + keywords + meta)."""
    feat_path = Path(CFG.features_dir) / "text" / "features.npz"
    if feat_path.exists():
        print(f"[skip] Text features already cached at {feat_path}")
        return
    print("\n" + "=" * 60)
    print("STEP 4a: Extract Text Features")
    print("=" * 60)
    import extract_text_features
    extract_text_features.main()


def step_extract_image_features():
    """Step 4b: Extract image features (CLIP + zero-shot + OCR keywords)."""
    feat_path = Path(CFG.features_dir) / "image" / "features.npz"
    if feat_path.exists():
        print(f"[skip] Image features already cached at {feat_path}")
        return
    print("\n" + "=" * 60)
    print("STEP 4b: Extract Image Features")
    print("=" * 60)
    import extract_image_features
    extract_image_features.main()

def step_extract_slm_features():
    """Step 4c: Extract SLM reasoning score."""
    feat_path = Path(CFG.features_dir) / "slm" / "features.npz"
    if feat_path.exists():
        print(f"[skip] SLM features already cached at {feat_path}")
        return
    print("\n" + "=" * 60)
    print("STEP 4c: Extract SLM Reasoning Features")
    print("=" * 60)
    import extract_slm_features
    extract_slm_features.main()


def step_train_evaluate():
    """Step 5: Train classifiers and evaluate with 5-fold CV."""
    print("\n" + "=" * 60)
    print("STEP 5: Train & Evaluate")
    print("=" * 60)
    import train_evaluate
    train_evaluate.main()


def main():
    parser = argparse.ArgumentParser(description="V2 LLM Detector Pipeline")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR step")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction")
    parser.add_argument("--train-only", action="store_true", help="Only train & evaluate")
    args = parser.parse_args()

    print("=" * 60)
    print("  V2 LLM Detector — Feature Fusion Pipeline")
    print("=" * 60)

    if not args.train_only:
        step_preprocess()

        if not args.skip_ocr:
            step_ocr()

        step_make_splits()

        if not args.skip_features:
            step_extract_text_features()
            step_extract_image_features()
            step_extract_slm_features()
    step_train_evaluate()

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
