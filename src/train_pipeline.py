import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG


def _resolve_abs_path(path_str: str) -> Path:
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def _is_within_dir(path: Path, root_dir: Path) -> bool:
    try:
        path.relative_to(root_dir)
        return True
    except ValueError:
        return False


def step_download_training_images() -> list[Path]:
    """
    Step 0: Ensure screenshot files exist locally.
    Downloads missing images for each app from Play Store metadata and updates raw dataset paths.
    Returns list of files newly downloaded in this run for optional cleanup.
    """
    raw_path = _resolve_abs_path(CFG.raw_dataset_path)
    if not raw_path.exists():
        print(f"[skip] Raw dataset not found at {raw_path}")
        return []

    print("\n[STEP 0] Ensure Training Images")

    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    from fetch_app_metadata import AppMetadataFetcher

    fetcher = AppMetadataFetcher()
    images_root = _resolve_abs_path(CFG.images_dir)

    downloaded_abs_paths = set()
    updated = 0
    skipped = 0
    attempted = 0
    failed_fetch = 0

    for row in rows:
        app_id = row.get("app_id")
        if not app_id:
            skipped += 1
            continue

        original_paths = row.get("image_paths", []) or []
        existing_paths = []
        missing_detected = False

        for rel_path in original_paths:
            abs_path = _resolve_abs_path(rel_path)
            if abs_path.exists() and abs_path.is_file():
                existing_paths.append(rel_path)
            else:
                missing_detected = True

        if original_paths and not missing_detected:
            skipped += 1
            continue

        attempted += 1

        app_image_dir = images_root / app_id
        existed_before = set()
        if app_image_dir.exists() and app_image_dir.is_dir():
            for existing in app_image_dir.iterdir():
                if existing.is_file():
                    existed_before.add(existing.resolve())

        play_data = fetcher.fetch_play_store_metadata(app_id)
        if not play_data:
            failed_fetch += 1
            new_paths = existing_paths
        else:
            screenshot_urls = play_data.get("screenshots", [])
            downloaded_paths = fetcher.download_screenshots(app_id, screenshot_urls)
            if downloaded_paths:
                new_paths = downloaded_paths
            else:
                new_paths = existing_paths

            for rel_path in downloaded_paths:
                abs_path = _resolve_abs_path(rel_path)
                if abs_path.exists() and abs_path.is_file() and abs_path not in existed_before:
                    downloaded_abs_paths.add(abs_path)

        if new_paths != original_paths:
            row["image_paths"] = new_paths
            updated += 1

    if updated > 0:
        with open(raw_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Apps checked: {len(rows)}")
    print(f"Already complete: {skipped}")
    print(f"Download attempts: {attempted}")
    print(f"Metadata fetch failed: {failed_fetch}")
    print(f"New files downloaded: {len(downloaded_abs_paths)}")
    if updated > 0:
        print(f"Updated image_paths in: {updated} records")
    else:
        print("No raw dataset rows needed updating")

    return sorted(downloaded_abs_paths)


def step_cleanup_downloaded_images(downloaded_paths: list[Path]) -> None:
    """Remove images downloaded in current run and prune now-empty app folders."""
    if not downloaded_paths:
        print("\n[skip] Cleanup downloaded images: nothing new was downloaded")
        return

    print("\n[CLEANUP] Remove Downloaded Training Images")

    images_root = _resolve_abs_path(CFG.images_dir)
    removed_files = 0
    missing_files = 0
    skipped_outside_root = 0
    removed_parent_dirs = set()

    for img_abs in downloaded_paths:
        abs_path = img_abs.resolve()
        if not _is_within_dir(abs_path, images_root):
            skipped_outside_root += 1
            continue

        if abs_path.exists() and abs_path.is_file():
            abs_path.unlink()
            removed_files += 1
            removed_parent_dirs.add(abs_path.parent)
        else:
            missing_files += 1

    for parent in sorted(removed_parent_dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            if parent.exists() and parent != images_root and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass

    print(f"Removed files: {removed_files}")
    if missing_files:
        print(f"Already missing: {missing_files}")
    if skipped_outside_root:
        print(f"Skipped outside {images_root}: {skipped_outside_root}")


def step_preprocess():
    """Step 1: Clean text and deduplicate images."""
    if Path(CFG.dataset_path).exists():
        print(f"[skip] {CFG.dataset_path} already exists")
        return
    print("\n[STEP 1] Preprocessing")
    from steps import preprocessing
    preprocessing.main()


def step_make_splits():
    """Step 3: Create stratified k-fold splits."""
    split_file = Path(CFG.splits_dir) / f"fold_{CFG.n_folds - 1}.json"
    if split_file.exists():
        print(f"[skip] Splits already exist in {CFG.splits_dir}")
        return
    print("\n[STEP 3] Create Splits")
    from steps import make_splits
    make_splits.main()


def step_ocr():
    """Step 2: Run Tesseract OCR on screenshots."""
    print("\n[STEP 2] OCR")
    from steps import run_ocr
    run_ocr.main()


def step_extract_text_features():
    """Step 4a: Extract text features (SBERT + keywords + meta)."""
    feat_path = Path(CFG.features_dir) / "text" / "features.npz"
    if feat_path.exists():
        print(f"[skip] Text features already cached at {feat_path}")
        return
    print("\n[STEP 4a] Extract Text Features")
    from steps import extract_text_features
    extract_text_features.main()


def step_extract_image_features():
    """Step 4b: Extract image features (CLIP + zero-shot + OCR keywords)."""
    feat_path = Path(CFG.features_dir) / "image" / "features.npz"
    if feat_path.exists():
        print(f"[skip] Image features already cached at {feat_path}")
        return
    print("\n[STEP 4b] Extract Image Features")
    from steps import extract_image_features
    extract_image_features.main()

def step_extract_slm_features():
    """Step 4c: Extract SLM reasoning score."""
    feat_path = Path(CFG.features_dir) / "slm" / "features.npz"
    if feat_path.exists():
        print(f"[skip] SLM features already cached at {feat_path}")
        return
    print("\n[STEP 4c] Extract SLM Reasoning Features")
    from steps import extract_slm_features
    extract_slm_features.main()


def step_train_evaluate():
    """Step 5: Train classifiers, ablation study, and α grid search."""
    print("\n[STEP 5] Train & Evaluate (+ Ablation + α grid search)")
    from steps import train_evaluate
    train_evaluate.main()


def step_k_sensitivity():
    """Step 6: k sensitivity analysis for SelectKBest.

    Finds optimal k, updates config.py on disk, and patches CFG in-memory
    so the subsequent train_evaluate step uses the correct k without restart.
    """
    summary_path = Path(CFG.runs_dir) / CFG.run_name / "k_sensitivity" / "summary.json"
    if summary_path.exists():
        print(f"\n[skip] k sensitivity already done at {summary_path.parent}")
        return
    print("\n[STEP 6] k Sensitivity Analysis")
    from steps import k_sensitivity
    best_k = k_sensitivity.main()
    if best_k is not None and best_k != CFG.feature_selection_k:
        object.__setattr__(CFG, "feature_selection_k", best_k)
        print(f"  → CFG.feature_selection_k patched to {best_k} for this run")


def step_statistical_tests():
    """Step 7: McNemar + Bootstrap AUC significance tests."""
    results_path = Path(CFG.runs_dir) / CFG.run_name / "statistical_tests" / "results.json"
    if results_path.exists():
        print(f"\n[skip] Statistical tests already done at {results_path.parent}")
        return
    print("\n[STEP 7] Statistical Significance Tests")
    from steps import statistical_tests
    statistical_tests.main()


def main():
    parser = argparse.ArgumentParser(description="V2 LLM Detector Pipeline")
    parser.add_argument("--skip-image-download", action="store_true", help="Skip downloading missing training images")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR step")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction")
    parser.add_argument("--keep-images", action="store_true", help="Keep newly downloaded images after training")
    parser.add_argument("--train-only", action="store_true", help="Only train & evaluate (includes ablation + α search)")
    parser.add_argument("--skip-k-sensitivity", action="store_true", help="Skip k sensitivity analysis")
    args = parser.parse_args()

    print("V2 LLM Detector — Feature Fusion Pipeline")

    downloaded_paths = []
    try:
        if not args.train_only:
            if not args.skip_image_download:
                downloaded_paths = step_download_training_images()
            else:
                print("\n[skip] STEP 0: Download training images")

            step_preprocess()

            if not args.skip_ocr:
                step_ocr()

            step_make_splits()

            if not args.skip_features:
                step_extract_text_features()
                step_extract_image_features()
                # SLM features are only used in the ablation study (not main pipeline).
                # Run step_extract_slm_features() manually if data/features/slm/ is missing.
                step_extract_slm_features()

        if not args.skip_k_sensitivity:
            step_k_sensitivity()

        step_train_evaluate()
        step_statistical_tests()
    finally:
        if downloaded_paths:
            if args.keep_images:
                print("\n[skip] Cleanup downloaded images because --keep-images was set")
            else:
                step_cleanup_downloaded_images(downloaded_paths)

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
