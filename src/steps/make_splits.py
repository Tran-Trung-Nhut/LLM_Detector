"""
Create stratified K-fold splits. Reused from V1.
"""
import json
import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG


def main():
    rows = []
    with open(CFG.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    out = Path(CFG.splits_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = df["app_id"].tolist()
    y = df["label_binary"].tolist()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_ids = [X[i] for i in train_idx]
        test_ids = [X[i] for i in test_idx]
        split = {"fold": fold, "train_ids": train_ids, "test_ids": test_ids}
        with open(out / f"fold_{fold}.json", "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"Wrote {CFG.n_folds} folds to {out}")


if __name__ == "__main__":
    main()
