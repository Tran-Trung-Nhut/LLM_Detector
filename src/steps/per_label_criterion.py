"""per_label_criterion.py — Per-label-criterion evaluation on the independent test set (construct validity)."""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json

# Zeroshot score threshold for "chat-style UI" proxy (tuned to ~top 25% of test distribution)
ZEROSHOT_THRESH = 0.15


def load_predictions(csv_path: Path) -> dict:
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["app_id"]] = {
                "y_true":     int(row["y_true"]),
                "y_prob":     float(row["y_prob"]),
                "text_prob":  float(row.get("text_prob", row["y_prob"])),
                "image_prob": float(row.get("image_prob", 0.0)),
            }
    return rows


def compute_metrics(y_true, y_pred) -> dict:
    if len(y_true) == 0:
        return {"n": 0, "n_pos": 0}
    n_pos = int(np.sum(y_true))
    return {
        "n":          len(y_true),
        "n_pos":      n_pos,
        "precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1":         round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "accuracy":   round(float(accuracy_score(y_true, y_pred)), 3),
    }


def assign_criterion_group(kw: np.ndarray, zeroshot: float) -> str:
    """
    Keyword feature layout (13 dims):
      [0]  total_count
      [1]  log1p(total_count)
      [2]  model_name binary   ← criterion 1
      [3]  model_name count
      [4]  core_llm binary
      [5]  core_llm count
      [6]  generation binary   ← criterion 2 component
      [7]  generation count
      [8]  interaction binary  ← criterion 2 component
      [9]  interaction count
      [10] content binary      ← criterion 2 component
      [11] content count
      [12] max_kw_len
    """
    has_model_name   = kw[2] > 0
    has_gen_behavior = (kw[6] > 0 or kw[8] > 0 or kw[10] > 0)
    has_chat_ui      = zeroshot > ZEROSHOT_THRESH

    if has_model_name:
        return "model_name_in_listing"
    if has_gen_behavior:
        return "generative_behavior"
    if has_chat_ui:
        return "chat_style_ui_only"
    return "hard_ambiguous"


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "per_label_criterion"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dir = base_dir / "independent_test"
    sv_csv   = test_dir / "predictions_soft_voting.csv"
    to_csv   = test_dir / "predictions_text_only.csv"
    if not sv_csv.exists() or not to_csv.exists():
        print(f"[error] Run independent_test_eval.py first.")
        return

    sv_preds = load_predictions(sv_csv)
    to_preds = load_predictions(to_csv)

    text_feat_path = Path(CFG.features_test_dir) / "text" / "features.npz"
    img_feat_path  = Path(CFG.features_test_dir) / "image" / "features.npz"
    if not text_feat_path.exists() or not img_feat_path.exists():
        print(f"[error] Test features not found. Extract features first.")
        return

    td  = np.load(text_feat_path,  allow_pickle=True)
    imd = np.load(img_feat_path,   allow_pickle=True)
    kw_map  = {aid: td["keywords"][i]  for i, aid in enumerate(td["app_ids"])}
    zs_map  = {aid: float(imd["zeroshot"][i][0]) for i, aid in enumerate(imd["app_ids"])}

    label_map = {}
    with open(CFG.inference_manual_csv) as f:
        for row in csv.DictReader(f):
            label_map[row["pkg_name"]] = int(row["label"])

    groups: dict[str, list] = {
        "model_name_in_listing": [],
        "generative_behavior":   [],
        "chat_style_ui_only":    [],
        "hard_ambiguous":        [],
    }

    for app_id in sv_preds:
        y_true = label_map.get(app_id, sv_preds[app_id]["y_true"])
        if y_true != 1:
            continue
        kw       = kw_map.get(app_id, np.zeros(13))
        zeroshot = zs_map.get(app_id, 0.0)
        group    = assign_criterion_group(kw, zeroshot)
        groups[group].append(app_id)

    neg_apps = [aid for aid, p in sv_preds.items()
                if label_map.get(aid, p["y_true"]) == 0]

    strategies = {
        "Text-Only":   to_preds,
        "Soft Voting": sv_preds,
    }

    results = {}
    tau = CFG.classification_threshold

    for group_name, pos_apps in groups.items():
        eval_apps = pos_apps + neg_apps
        if not eval_apps:
            continue
        results[group_name] = {}

        for strat_name, preds in strategies.items():
            y_true = np.array([label_map.get(a, preds[a]["y_true"]) for a in eval_apps if a in preds], dtype=int)
            y_pred = np.array([(preds[a]["y_prob"] >= tau) for a in eval_apps if a in preds], dtype=int)
            m = compute_metrics(y_true, y_pred)
            m["n_pos_group"] = len(pos_apps)
            results[group_name][strat_name] = m

    write_json(out_dir / "per_label_criterion.json", results)

    group_summary = {g: {"n_pos": len(apps), "app_ids": apps} for g, apps in groups.items()}
    write_json(out_dir / "group_membership.json", group_summary)

    print(f"\nPer-label-criterion evaluation (independent test set, positives only split by criterion):")
    print(f"  Zeroshot threshold for chat-style UI: {ZEROSHOT_THRESH}")
    print()
    header = f"  {'Group':<28} {'N pos':>6}  {'Text-Only F1':>13}  {'Soft Voting F1':>15}"
    print(header)
    print("  " + "-" * 70)
    for group_name in ["model_name_in_listing", "generative_behavior", "chat_style_ui_only", "hard_ambiguous"]:
        if group_name not in results:
            continue
        r = results[group_name]
        n_pos = len(groups[group_name])
        f1_to = r.get("Text-Only",   {}).get("f1", "N/A")
        f1_sv = r.get("Soft Voting", {}).get("f1", "N/A")
        print(f"  {group_name:<28} {n_pos:>6}  {str(f1_to):>13}  {str(f1_sv):>15}")

    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
