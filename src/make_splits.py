import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def main(
    dataset_path="data/apps.jsonl",
    out_dir="data/splits",
    n_splits=5,
    seed=42,
):
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    assert "app_id" in df.columns and "label_binary" in df.columns

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = df["app_id"].tolist()
    y = df["label_binary"].tolist()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_ids = [X[i] for i in train_idx]
        test_ids = [X[i] for i in test_idx]
        split = {"fold": fold, "train_ids": train_ids, "test_ids": test_ids}
        with open(out / f"fold_{fold}.json", "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"Wrote {n_splits} folds to {out}")

if __name__ == "__main__":
    main()