"""image_analysis_examples.py — Categorises test apps by text/image branch interaction pattern (saves, misleads, etc.)."""
import csv
import json
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


def load_sv_predictions(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "app_id":     row["app_id"],
                "y_true":     int(row["y_true"]),
                "y_prob":     float(row["y_prob"]),
                "text_prob":  float(row.get("text_prob", row["y_prob"])),
                "image_prob": float(row.get("image_prob", 0.5)),
            })
    return rows


def load_app_info(jsonl_path: str) -> dict:
    info = {}
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                info[r["app_id"]] = {
                    "title":             r.get("title", ""),
                    "category":          r.get("category", ""),
                    "short_description": r.get("short_description", ""),
                    "description":       (r.get("description", "") or "")[:300],
                }
    except FileNotFoundError:
        pass
    return info


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "image_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    sv_csv = base_dir / "independent_test" / "predictions_soft_voting.csv"
    if not sv_csv.exists():
        print(f"[error] Run independent_test_eval.py first to generate {sv_csv}")
        return

    rows     = load_sv_predictions(sv_csv)
    app_info = load_app_info(CFG.raw_inference_dataset_path)
    tau      = CFG.classification_threshold

    patterns = {
        "A_image_saves":    [],  # text miss, image catches → TP
        "B_image_misleads": [],  # text ok / miss, image wrong → FP
        "C_text_leads":     [],  # text correct, image misses → TP via text
        "D_both_agree_pos": [],  # both high → easy TP
        "E_both_agree_neg": [],  # both low → easy TN
        "F_both_wrong":     [],  # both wrong
    }

    for r in rows:
        text_hi  = r["text_prob"] >= tau
        image_hi = r["image_prob"] >= tau
        pos      = r["y_true"] == 1
        fusion_pred = int(r["y_prob"] >= tau)

        if not text_hi and image_hi and pos:
            patterns["A_image_saves"].append(r)
        elif not text_hi and image_hi and not pos:
            patterns["B_image_misleads"].append(r)
        elif text_hi and not image_hi and pos:
            patterns["C_text_leads"].append(r)
        elif text_hi and image_hi and pos:
            patterns["D_both_agree_pos"].append(r)
        elif not text_hi and not image_hi and not pos:
            patterns["E_both_agree_neg"].append(r)
        else:
            patterns["F_both_wrong"].append(r)

    summary = {}
    for pat, pat_rows in patterns.items():
        n = len(pat_rows)
        if n == 0:
            summary[pat] = {"n": 0}
            continue
        y_true = np.array([r["y_true"] for r in pat_rows])
        y_pred = np.array([int(r["y_prob"] >= tau) for r in pat_rows])
        summary[pat] = {
            "n":                n,
            "n_pos":            int(y_true.sum()),
            "fusion_accuracy":  round(float(np.mean(y_true == y_pred)), 3),
            "avg_text_prob":    round(float(np.mean([r["text_prob"]  for r in pat_rows])), 3),
            "avg_image_prob":   round(float(np.mean([r["image_prob"] for r in pat_rows])), 3),
        }

    write_json(out_dir / "image_analysis_summary.json", summary)

    def enrich(app_rows: list, top_n: int = 5) -> list:
        enriched = []
        for r in sorted(app_rows, key=lambda x: abs(x["image_prob"] - x["text_prob"]), reverse=True)[:top_n]:
            info = app_info.get(r["app_id"], {})
            enriched.append({
                **r,
                "title":             info.get("title", ""),
                "category":          info.get("category", ""),
                "short_description": info.get("short_description", ""),
                "description_snippet": info.get("description", ""),
            })
        return enriched

    write_json(out_dir / "image_helps_examples.json",   enrich(patterns["A_image_saves"]))
    write_json(out_dir / "image_hurts_examples.json",   enrich(patterns["B_image_misleads"]))
    write_json(out_dir / "text_leads_examples.json",    enrich(patterns["C_text_leads"]))

    print("\nImage Branch Analysis — Independent Test Set (N=110)")
    print(f"  tau = {tau}")
    print()
    print(f"  {'Pattern':<30} {'N':>4} {'N+':>4} {'FusionAcc':>10} {'AvgText':>9} {'AvgImg':>9}")
    print("  " + "-" * 70)
    labels_map = {
        "A_image_saves":    "A: Image saves (text miss, img hit)",
        "B_image_misleads": "B: Image misleads (text ok, img FP)",
        "C_text_leads":     "C: Text leads (text hit, img miss)",
        "D_both_agree_pos": "D: Both agree positive",
        "E_both_agree_neg": "E: Both agree negative",
        "F_both_wrong":     "F: Both wrong",
    }
    for pat, label in labels_map.items():
        s = summary.get(pat, {})
        if s.get("n", 0) == 0:
            print(f"  {label:<30} {'0':>4}")
            continue
        print(f"  {label:<30} {s['n']:>4} {s.get('n_pos',0):>4} "
              f"{s['fusion_accuracy']:>10.3f} {s['avg_text_prob']:>9.3f} {s['avg_image_prob']:>9.3f}")

    n_img_saves    = summary.get("A_image_saves", {}).get("n", 0)
    n_img_misleads = summary.get("B_image_misleads", {}).get("n", 0)
    print(f"\n  → Image branch saves {n_img_saves} TPs that text alone would miss")
    print(f"  → Image branch introduces {n_img_misleads} FPs not caused by text alone")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
