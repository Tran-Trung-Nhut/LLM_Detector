"""
calibration.py — Table 17: Brier score + ECE (raw and Platt-scaled).
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json


def load_pred_csv(path: Path):
    y_true, y_prob = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["y_true"]))
            y_prob.append(float(row["y_prob"]))
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float)


def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def ece(y_true, y_prob, n_bins: int = None):
    if n_bins is None:
        n_bins = CFG.ece_n_bins
    bins = np.linspace(0, 1, n_bins + 1)
    total_ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        total_ece += (mask.sum() / n) * abs(acc - conf)
    return round(total_ece, 4)


def platt_scale(y_true_val, y_prob_val, y_prob_test):
    clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    clf.fit(y_prob_val.reshape(-1, 1), y_true_val)
    return clf.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_dirs = {
        "Text-Only":    base_dir / "text_only",
        "Early Fusion": base_dir / "fusion" / "early_fusion",
        "Score-Max":    base_dir / "fusion" / "late_fusion_score_max",
        "Soft Voting":  base_dir / "fusion" / "late_fusion_soft_voting",
        "Stacking":     base_dir / "fusion" / "late_fusion_stacking",
    }

    results = {}
    print(f"\n{'Strategy':<20} {'Brier(raw)':>10} {'ECE(raw)':>9} {'Brier(Platt)':>13} {'ECE(Platt)':>11}")
    print("-" * 70)

    for name, strat_dir in strategy_dirs.items():
        test_csv = strat_dir / "predictions.csv"
        val_csv  = strat_dir / "validation_predictions.csv"
        if not test_csv.exists() or not val_csv.exists():
            print(f"  [skip] {name}")
            continue

        test_rows = []
        with open(test_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                test_rows.append({
                    "fold":   int(row["fold"]),
                    "y_true": int(row["y_true"]),
                    "y_prob": float(row["y_prob"]),
                })

        val_rows = []
        with open(val_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val_rows.append({
                    "fold":   int(row["fold"]),
                    "y_true": int(row["y_true"]),
                    "y_prob": float(row["y_prob"]),
                })

        y_test_all, p_raw_all, p_platt_all = [], [], []

        for fold in range(CFG.n_folds):
            val_fold  = [r for r in val_rows  if r["fold"] == fold]
            test_fold = [r for r in test_rows if r["fold"] == fold]
            if not val_fold or not test_fold:
                continue

            y_val = np.array([r["y_true"] for r in val_fold],  dtype=int)
            p_val = np.array([r["y_prob"] for r in val_fold],  dtype=float)
            y_te  = np.array([r["y_true"] for r in test_fold], dtype=int)
            p_te  = np.array([r["y_prob"] for r in test_fold], dtype=float)

            p_te_platt = platt_scale(y_val, p_val, p_te)

            y_test_all.extend(y_te.tolist())
            p_raw_all.extend(p_te.tolist())
            p_platt_all.extend(p_te_platt.tolist())

        y_test_all  = np.array(y_test_all,  dtype=int)
        p_raw_all   = np.array(p_raw_all,   dtype=float)
        p_platt_all = np.array(p_platt_all, dtype=float)

        b_raw   = brier_score(y_test_all, p_raw_all)
        e_raw   = ece(y_test_all, p_raw_all)
        b_platt = brier_score(y_test_all, p_platt_all)
        e_platt = ece(y_test_all, p_platt_all)

        results[name] = {
            "brier_raw":   round(b_raw,   3),
            "ece_raw":     e_raw,
            "brier_platt": round(b_platt, 3),
            "ece_platt":   e_platt,
            "protocol":    "per-fold Platt: fit on inner-val_k, eval on outer-test_k",
        }
        print(f"  {name:<20} {b_raw:>10.3f} {e_raw:>9.3f} {b_platt:>13.3f} {e_platt:>11.3f}")

    write_json(out_dir / "table17_calibration.json", results)
    print(f"\nSaved: {out_dir}")
    print("Protocol: per-fold Platt — inner-val and outer-test are guaranteed disjoint per fold.")


if __name__ == "__main__":
    main()
