"""k_sensitivity.py — Grid search over SelectKBest k on text features to justify the chosen k value."""
import os
import re
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
from utils.seed import set_seed
from steps.train_evaluate import load_features, run_single_experiment, aggregate_metrics


def main():
    set_seed(CFG.seed)
    base_dir = Path(CFG.runs_dir) / CFG.run_name / "k_sensitivity"
    base_dir.mkdir(parents=True, exist_ok=True)

    data = load_features()
    n_dims = data["text_feats"].shape[1]
    print(f"  Text feature dims: {n_dims}, current k: {CFG.feature_selection_k}")

    k_candidates = list(CFG.k_sensitivity_values) + [n_dims]
    k_candidates = sorted(set(k_candidates))

    summary = {}
    for k in k_candidates:
        actual_k = min(k, n_dims)
        label = f"all_{n_dims}" if k == n_dims else str(k)
        exp_dir = base_dir / f"k_{label}"
        print(f"\n  [k={label}]  actual_k={actual_k}")
        _, fold_metrics = run_single_experiment(
            f"k_{label}", data["text_feats"], data, exp_dir, k_features=actual_k
        )
        agg = aggregate_metrics(fold_metrics)
        summary[label] = {
            "k": k,
            "actual_k": actual_k,
            "roc_auc_mean": round(agg["roc_auc_mean"], 4),
            "roc_auc_std": round(agg["roc_auc_std"], 4),
            "f1_pos_mean": round(agg["f1_pos_mean"], 4),
            "f1_pos_std": round(agg["f1_pos_std"], 4),
        }

    write_json(base_dir / "summary.json", summary)

    print("\nK SENSITIVITY — Text Features (TextOnly classifier)")
    print(f"  {'k':>12}  {'ROC-AUC':>14}  {'F1':>14}")
    print("  " + "-" * 60)
    for label, r in summary.items():
        marker = " ← current" if r["k"] == CFG.feature_selection_k else ""
        print(
            f"  {label:>12}  "
            f"{r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}  "
            f"{r['f1_pos_mean']:.4f}±{r['f1_pos_std']:.4f}"
            f"{marker}"
        )

    best_label = max(summary, key=lambda lbl: summary[lbl]["roc_auc_mean"])
    best_k = summary[best_label]["k"]
    old_k = CFG.feature_selection_k
    print(f"\n  Best k by ROC-AUC: {best_label}  (AUC={summary[best_label]['roc_auc_mean']:.4f})")

    if best_k != old_k:
        _update_config_k(best_k)
        print(f"  ✓ config.py updated: feature_selection_k {old_k} → {best_k}")
    else:
        print(f"  Current k={old_k} is already optimal — config unchanged.")

    print(f"\nResults saved to: {base_dir}")
    return best_k


def _update_config_k(new_k: int) -> None:
    config_path = _PROJECT_ROOT / "src" / "config.py"
    content = config_path.read_text(encoding="utf-8")
    new_content = re.sub(
        r"(feature_selection_k\s*:\s*int\s*=\s*)\d+",
        rf"\g<1>{new_k}",
        content,
    )
    config_path.write_text(new_content, encoding="utf-8")


if __name__ == "__main__":
    main()
