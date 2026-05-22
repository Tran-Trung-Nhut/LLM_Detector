"""prior_correction.py — Prior-corrected precision (Table 14, Figure 5)."""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json


def load_predictions(csv_path: Path):
    y_true, y_prob = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["y_true"]))
            y_prob.append(float(row["y_prob"]))
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float)


def prior_corrected_precision(tpr: float, fpr: float, pi: float) -> float:
    num = pi * tpr
    denom = pi * tpr + (1.0 - pi) * fpr
    return float(num / denom) if denom > 1e-12 else 0.0


def tpr_fpr_at_threshold(y_true, y_prob, tau: float):
    y_pred = (y_prob >= tau).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr


def find_best_threshold_f1(y_true, y_prob):
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "prior_correction"
    out_dir.mkdir(parents=True, exist_ok=True)

    pi_values = list(CFG.prior_pi_values)
    pi_range  = np.logspace(-3, -1, 200)

    val_paths = {
        "Text-Only":    base_dir / "text_only" / "validation_predictions.csv",
        "Early Fusion": base_dir / "fusion" / "early_fusion" / "validation_predictions.csv",
        "Score-Max":    base_dir / "fusion" / "late_fusion_score_max" / "validation_predictions.csv",
        "Soft Voting":  base_dir / "fusion" / "late_fusion_soft_voting" / "validation_predictions.csv",
        "Stacking":     base_dir / "fusion" / "late_fusion_stacking" / "validation_predictions.csv",
    }
    test_dir = base_dir / "independent_test"
    test_paths = {
        "Text-Only":    test_dir / "predictions_text_only.csv",
        "Early Fusion": test_dir / "predictions_early_fusion.csv",
        "Score-Max":    test_dir / "predictions_score_max.csv",
        "Soft Voting":  test_dir / "predictions_soft_voting.csv",
        "Stacking":     test_dir / "predictions_stacking.csv",
    }
    cv_test_paths = {
        "Text-Only":    base_dir / "text_only" / "predictions.csv",
        "Early Fusion": base_dir / "fusion" / "early_fusion" / "predictions.csv",
        "Score-Max":    base_dir / "fusion" / "late_fusion_score_max" / "predictions.csv",
        "Soft Voting":  base_dir / "fusion" / "late_fusion_soft_voting" / "predictions.csv",
        "Stacking":     base_dir / "fusion" / "late_fusion_stacking" / "predictions.csv",
    }

    colors = {"Text-Only": "black", "Early Fusion": "green",
              "Score-Max": "orange", "Soft Voting": "blue", "Stacking": "red"}

    table14_rows = []
    results = {}
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in val_paths:
        val_csv = val_paths[name]
        if not val_csv.exists():
            continue

        y_val, p_val = load_predictions(val_csv)
        tau_opt = find_best_threshold_f1(y_val, p_val)

        if test_paths[name].exists():
            y_tpr_fpr, p_tpr_fpr = load_predictions(test_paths[name])
            source = "independent test set (N=110)"
        elif cv_test_paths[name].exists():
            y_tpr_fpr, p_tpr_fpr = load_predictions(cv_test_paths[name])
            source = "CV test predictions (fallback)"
        else:
            continue

        tpr_opt, fpr_opt = tpr_fpr_at_threshold(y_tpr_fpr, p_tpr_fpr, tau_opt)

        row = {"strategy": name, "threshold": round(tau_opt, 2),
               "tpr": round(tpr_opt, 3), "fpr": round(fpr_opt, 3), "source": source}
        for pi in pi_values:
            row[f"prec_pi_{pi}"] = round(prior_corrected_precision(tpr_opt, fpr_opt, pi), 3)
        table14_rows.append(row)

        curve = [prior_corrected_precision(tpr_opt, fpr_opt, pi) for pi in pi_range]
        ax.plot(pi_range, curve, color=colors.get(name, "gray"), label=name, linewidth=2)
        results[name] = row

    ax.set_xscale("log")
    ax.axvspan(0.005, 0.05, alpha=0.15, color="gray", label="Realistic deploy range")
    ax.set_xlabel("Deployment positive prior π_deploy (log scale)")
    ax.set_ylabel("Prior-corrected precision Prec_π(τ)")
    ax.set_title("Figure 5: Prior-corrected precision vs deployment prior")
    ax.legend(fontsize=9)
    ax.set_xlim([1e-3, 1e-1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(out_dir / "figure5_prior_corrected_precision.png", dpi=150)
    plt.close(fig)

    write_json(out_dir / "table14_prior_corrected.json", results)
    if table14_rows:
        with open(out_dir / "table14.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table14_rows[0].keys()))
            writer.writeheader()
            writer.writerows(table14_rows)

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
