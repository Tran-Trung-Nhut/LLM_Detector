"""
One-command runner.

Order:
1) make_splits (only if not exist)
2) train (5 folds)
3) infer multi-image (5 folds)

Run:
  python -m src.run_experiment
"""
from pathlib import Path
from huggingface_hub import login
from src.config import CFG
import os

from config import CFG

def _splits_exist() -> bool:
    return all(Path(CFG.splits_dir, f"fold_{i}.json").exists() for i in range(CFG.n_folds))

def main():

    if CFG.hf_token:
        print("[INFO] Logging in to Hugging Face...")
        login(token=CFG.hf_token)

    if not _splits_exist():
        os.system("python -m src.make_splits")

    os.system("python -m src.train_paligemma_lora_single_image")
    os.system("python -m src.infer_paligemma_multi_image")

if __name__ == "__main__":
    main()