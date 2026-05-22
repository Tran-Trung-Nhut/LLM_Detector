"""
per_category.py — Table 13: F1 by app category for Soft Voting vs Text-Only.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json, read_jsonl


def load_pred_with_id(csv_path: Path):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["app_id"]] = {
                "y_true": int(row["y_true"]),
                "y_pred": int(float(row["y_prob"]) >= 0.5),
            }
    return rows


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "per_category"
    out_dir.mkdir(parents=True, exist_ok=True)

    app_category = {}
    test_dataset_path = Path(CFG.raw_inference_dataset_path)
    if not test_dataset_path.exists():
        test_dataset_path = Path(CFG.inference_dataset_path)
    if not test_dataset_path.exists():
        print(f"[error] Test dataset not found. Tried: {test_dataset_path}")
        return
    for rec in read_jsonl(str(test_dataset_path)):
        cat = (rec.get("category") or "Other").strip()
        app_category[rec["app_id"]] = cat

    test_dir = base_dir / "independent_test"
    sv_csv   = test_dir / "predictions_soft_voting.csv"
    if not sv_csv.exists():
        print(f"[error] Run independent_test_eval.py first to generate {sv_csv}")
        return

    sv_preds = load_pred_with_id(sv_csv)

    to_preds = {}
    with open(sv_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            to_preds[row["app_id"]] = {
                "y_true": int(row["y_true"]),
                "y_pred": int(float(row.get("text_prob", row["y_prob"])) >= 0.5),
            }

    cat_groups = {
        "Productivity":  ["Productivity"],
        "Education":     ["Education"],
        "Tools":         ["Tools"],
        "Communication": ["Communication"],
        "Lifestyle":     ["Lifestyle"],
    }
    all_cats_covered = {c for cats in cat_groups.values() for c in cats}

    results = {}
    for group, cats in list(cat_groups.items()) + [("Other", None)]:
        sv_yt, sv_yp, to_yp = [], [], []
        for app_id, cat in app_category.items():
            in_group = (cats is None and cat not in all_cats_covered) or (cats and cat in cats)
            if not in_group or app_id not in sv_preds:
                continue
            sv_yt.append(sv_preds[app_id]["y_true"])
            sv_yp.append(sv_preds[app_id]["y_pred"])
            to_yp.append(to_preds.get(app_id, {"y_pred": 0})["y_pred"])

        if not sv_yt:
            continue
        sv_yt = np.array(sv_yt)
        f1_sv = round(float(f1_score(sv_yt, sv_yp, zero_division=0)), 3)
        f1_to = round(float(f1_score(sv_yt, to_yp, zero_division=0)), 3)
        results[group] = {
            "n": len(sv_yt),
            "n_pos": int(sv_yt.sum()),
            "f1_soft_voting": f1_sv,
            "f1_text_only":   f1_to,
            "delta_f1_vs_text_only": round(f1_sv - f1_to, 2),
        }

    write_json(out_dir / "table13_per_category.json", results)

    print("\nTable 13 (Per-Category F1, Soft Voting — Independent Test Set):")
    print(f"  {'Category':<16} {'N':>4} {'N+':>4} {'F1(SV)':>8} {'ΔF1':>7}")
    print("  " + "-" * 50)
    for cat, r in results.items():
        print(f"  {cat:<16} {r['n']:>4} {r['n_pos']:>4} {r['f1_soft_voting']:>8.3f} {r['delta_f1_vs_text_only']:>+7.2f}")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
