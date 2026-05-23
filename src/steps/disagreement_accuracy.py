"""
disagreement_accuracy.py — Table 11.

Accuracy on apps where |s_text - s_img| > 0.3 (branch disagreement subset).
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json


def load_fusion_pred(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "app_id":     row["app_id"],
                "y_true":     int(row["y_true"]),
                "y_prob":     float(row["y_prob"]),
                "text_prob":  float(row.get("text_prob", row["y_prob"])),
                "image_prob": float(row.get("image_prob", row["y_prob"])),
            })
    return rows


def accuracy_on_subset(rows, mask, tau=0.5):
    subset = [r for r, m in zip(rows, mask) if m]
    if not subset:
        return None, 0
    y_true = np.array([r["y_true"] for r in subset])
    y_pred = (np.array([r["y_prob"] for r in subset]) >= tau).astype(int)
    return round(float(np.mean(y_true == y_pred)), 3), len(subset)


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir = base_dir / "disagreement_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)

    sv_path = base_dir / "fusion" / "late_fusion_soft_voting" / "predictions.csv"
    if not sv_path.exists():
        print(f"[error] {sv_path} not found")
        return

    sv_rows = load_fusion_pred(sv_path)
    sv_app_order = [r["app_id"] for r in sv_rows]

    disagree_mask = [
        abs(r["text_prob"] - r["image_prob"]) > CFG.disagree_threshold
        for r in sv_rows
    ]
    n_disagree = sum(disagree_mask)
    n_pos_disagree = sum(r["y_true"] for r, m in zip(sv_rows, disagree_mask) if m)
    print(f"Disagreement subset: {n_disagree} apps ({n_pos_disagree} positives, {n_disagree - n_pos_disagree} negatives)")

    strategies = {
        "Text-Only":    base_dir / "text_only" / "predictions.csv",
        "Image-Only":   base_dir / "image_only" / "predictions.csv",
        "Early Fusion": base_dir / "fusion" / "early_fusion" / "predictions.csv",
        "Score-Max":    base_dir / "fusion" / "late_fusion_score_max" / "predictions.csv",
        "Soft Voting":  base_dir / "fusion" / "late_fusion_soft_voting" / "predictions.csv",
        "Stacking":     base_dir / "fusion" / "late_fusion_stacking" / "predictions.csv",
    }

    results = {}
    print(f"\n{'Strategy':<20} {'Disagree-set size':>18} {'Disagree-Accuracy':>18}")
    print("-" * 60)

    for name, csv_path in strategies.items():
        if not csv_path.exists():
            continue
        rows = load_fusion_pred(csv_path)
        row_by_id = {r["app_id"]: r for r in rows}
        aligned_rows = [row_by_id.get(aid, {"y_true": 0, "y_prob": 0.5}) for aid in sv_app_order]
        acc, n = accuracy_on_subset(aligned_rows, disagree_mask)
        results[name] = {"disagree_set_size": n, "disagree_accuracy": acc}
        print(f"  {name:<20} {n:>18} {acc if acc is not None else 'N/A':>18}")

    write_json(out_dir / "table11_disagree_accuracy.json", results)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
