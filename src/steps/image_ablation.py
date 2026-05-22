"""image_ablation.py — Image branch leave-one-out ablation under 5-fold CV."""
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
from utils.seed import set_seed
from steps.train_evaluate import load_features, run_single_experiment, aggregate_metrics


def main():
    set_seed(CFG.seed)
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "image_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_features()

    clip_mean = data["clip_mean"]
    clip_max  = data["clip_max"]
    zeroshot  = data["zeroshot"]
    ocr       = data["ocr"]

    ablation_configs = {
        "full_image_branch":   np.concatenate([clip_mean, clip_max, zeroshot, ocr], axis=1),
        "minus_clip_pooled":   np.concatenate([zeroshot, ocr], axis=1),
        "minus_zeroshot_chat": np.concatenate([clip_mean, clip_max, ocr], axis=1),
        "minus_ocr_features":  np.concatenate([clip_mean, clip_max, zeroshot], axis=1),
    }

    summary = {}
    full_auc = None

    for name, X in ablation_configs.items():
        marker = " ← full" if name == "full_image_branch" else ""
        print(f"\n  [Image Ablation] {name}{marker}  ({X.shape[1]} dims)")
        _, fold_metrics = run_single_experiment(name, X, data, out_dir / name)
        agg = aggregate_metrics(fold_metrics)
        summary[name] = {
            "n_dims":       int(X.shape[1]),
            "roc_auc_mean": round(agg["roc_auc_mean"], 4),
            "roc_auc_std":  round(agg["roc_auc_std"],  4),
            "f1_pos_mean":  round(agg["f1_pos_mean"],  4),
            "f1_pos_std":   round(agg["f1_pos_std"],   4),
        }
        if name == "full_image_branch":
            full_auc = agg["roc_auc_mean"]

    if full_auc is not None:
        for name in summary:
            delta = round(summary[name]["roc_auc_mean"] - full_auc, 3)
            summary[name]["delta_roc_auc"] = delta if name != "full_image_branch" else None

    write_json(out_dir / "image_ablation_summary.json", summary)

    print("\nIMAGE ABLATION — Image Branch Components (Table 5)")
    print(f"  {'Config':<30} {'Dims':>5}  {'ROC-AUC':>14}  {'ΔROC-AUC':>10}")
    print("  " + "-" * 65)
    for name, r in summary.items():
        delta_str = f"{r['delta_roc_auc']:+.3f}" if r.get("delta_roc_auc") is not None else "    —"
        print(
            f"  {name:<30} {r['n_dims']:>5}  "
            f"{r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}  "
            f"{delta_str:>10}"
        )
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
