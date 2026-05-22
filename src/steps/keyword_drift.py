"""keyword_drift.py — Robustness test removing model-name keywords from the vocabulary to simulate vocabulary aging."""
import csv
import json
import math
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
from utils.io import write_json, read_jsonl
from utils.metrics import compute_binary_metrics
from utils.seed import set_seed
from steps.train_evaluate import (
    load_split, make_inner_split_indices,
    fit_select_kbest, train_lgbm, predict_lgbm,
    aggregate_metrics, find_best_threshold_from_arrays,
)


# Categories kept when model-name keywords are removed
_DRIFT_CATEGORIES = {k: v for k, v in CFG.keyword_categories.items() if k != "model_name"}


def compute_keyword_features_no_modelname(text: str) -> np.ndarray:
    """13-dim keyword feature vector without model_name category."""
    lower = text.lower()
    total_count = 0
    cat_counts = {}
    max_kw_len = 0

    for cat, kws in _DRIFT_CATEGORIES.items():
        cat_count = 0
        for kw in kws:
            n = len(re.findall(re.escape(kw), lower))
            if n > 0:
                cat_count += n
                max_kw_len = max(max_kw_len, len(kw))
        cat_counts[cat] = cat_count
        total_count += cat_count

    feats = [total_count, math.log1p(total_count)]
    feats += [0.0, 0.0]  # model_name slots zeroed to preserve 13-dim layout
    for cat in CFG.keyword_categories:
        if cat == "model_name":
            continue
        feats.append(1.0 if cat_counts[cat] > 0 else 0.0)
        feats.append(float(cat_counts[cat]))
    feats.append(float(max_kw_len))

    return np.array(feats, dtype=np.float32)


def _get_text(row: dict) -> str:
    if row.get("text"):
        return row["text"]
    parts = []
    if row.get("title"):
        parts.append(f"[TITLE] {row['title']}")
    if row.get("category"):
        parts.append(f"[CATEGORY] {row['category']}")
    if row.get("short_description"):
        parts.append(f"[SHORT_DESCRIPTION] {row['short_description']}")
    if row.get("description"):
        parts.append(f"[DESCRIPTION]\n{row['description']}")
    if row.get("recent_changes_text"):
        parts.append(f"[RECENT_CHANGES]\n{row['recent_changes_text']}")
    return "\n".join(parts)


def rebuild_keyword_features(jsonl_path: str, existing_npz: Path) -> np.ndarray:
    """Re-compute keyword features without model-name, aligned to the given npz order."""
    d = np.load(existing_npz, allow_pickle=True)
    app_ids_order = list(d["app_ids"])

    paths_to_try = [jsonl_path]
    if "inference" in jsonl_path:
        paths_to_try.append(CFG.raw_inference_dataset_path)
    else:
        paths_to_try.append(CFG.raw_dataset_path)

    text_by_id = {}
    for path in paths_to_try:
        try:
            for row in read_jsonl(path):
                text_by_id[row["app_id"]] = _get_text(row)
            break
        except FileNotFoundError:
            continue

    new_kw = []
    for aid in app_ids_order:
        text = text_by_id.get(aid, "")
        new_kw.append(compute_keyword_features_no_modelname(text))
    return np.stack(new_kw)


def run_fold(fold: int, X_all: np.ndarray, labels: np.ndarray, id2idx: dict,
             app_ids: list, k_features: int):
    split = load_split(fold)
    outer_train_idx = [id2idx[a] for a in split["train_ids"] if a in id2idx]
    test_idx        = [id2idx[a] for a in split["test_ids"]  if a in id2idx]

    X_outer_train, y_outer_train = X_all[outer_train_idx], labels[outer_train_idx]
    X_test, y_test               = X_all[test_idx], labels[test_idx]

    inner_train_rel, inner_val_rel = make_inner_split_indices(y_outer_train, fold)
    X_train  = X_outer_train[inner_train_rel]
    y_train  = y_outer_train[inner_train_rel]
    X_val    = X_outer_train[inner_val_rel]
    y_val    = y_outer_train[inner_val_rel]

    _, X_train_sel, X_val_sel, X_test_sel = fit_select_kbest(
        X_train, y_train, X_val, X_test, k_features
    )
    model    = train_lgbm(X_train_sel, y_train, X_val_sel, y_val)
    y_prob   = predict_lgbm(model, X_test_sel)
    y_prob_v = predict_lgbm(model, X_val_sel)
    metrics  = compute_binary_metrics(y_test, y_prob, threshold=CFG.classification_threshold)
    metrics["fold"] = fold

    test_preds = [{"app_id": app_ids[test_idx[i]], "y_true": int(y_test[i]),
                   "y_prob": float(y_prob[i])} for i in range(len(test_idx))]
    val_preds  = [{"app_id": app_ids[outer_train_idx[inner_val_rel[i]]],
                   "y_true": int(y_outer_train[inner_val_rel[i]]),
                   "y_prob": float(y_prob_v[i])} for i in range(len(inner_val_rel))]
    return model, metrics, test_preds, val_preds, inner_val_rel


def evaluate_on_test_set(models_with_selectors, X_test_drift: np.ndarray,
                         y_test: np.ndarray, tau: float) -> dict:
    probs = []
    for sel, mdl in models_with_selectors:
        probs.append(mdl.predict(sel.transform(X_test_drift)))
    y_prob = np.mean(np.vstack(probs), axis=0)
    return compute_binary_metrics(y_test, y_prob, threshold=tau)


def main():
    set_seed(CFG.seed)
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "keyword_drift"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"KEYWORD DRIFT — Removed: {CFG.keyword_categories['model_name']}")

    train_text_path = Path(CFG.features_dir) / "text" / "features.npz"
    td = np.load(train_text_path, allow_pickle=True)
    app_ids = list(td["app_ids"])
    labels  = td["labels"].astype(int)
    id2idx  = {a: i for i, a in enumerate(app_ids)}

    print("\n[1/4] Re-computing keyword features without model-name...")
    new_kw_train = rebuild_keyword_features(CFG.dataset_path, train_text_path)
    X_drift = np.concatenate([td["sbert"], new_kw_train, td["meta"]], axis=1)
    print(f"  Training feature shape: {X_drift.shape}")
    print("\n[2/4] Training 5-fold text-only without model-name keywords...")
    all_test_preds, all_val_preds, fold_metrics = [], [], []
    saved_models = []  # (selector, model) per fold

    for fold in range(CFG.n_folds):
        split = load_split(fold)
        outer_train_idx = [id2idx[a] for a in split["train_ids"] if a in id2idx]
        test_idx        = [id2idx[a] for a in split["test_ids"]  if a in id2idx]

        X_outer_train = X_drift[outer_train_idx]
        y_outer_train = labels[outer_train_idx]
        X_test_fold   = X_drift[test_idx]
        y_test_fold   = labels[test_idx]

        inner_tr, inner_val = make_inner_split_indices(y_outer_train, fold)
        selector, X_tr_sel, X_val_sel, X_test_sel = fit_select_kbest(
            X_outer_train[inner_tr], y_outer_train[inner_tr],
            X_outer_train[inner_val], X_test_fold, CFG.feature_selection_k
        )
        model = train_lgbm(
            X_tr_sel, y_outer_train[inner_tr],
            X_val_sel, y_outer_train[inner_val]
        )
        saved_models.append((selector, model))

        y_prob_test = predict_lgbm(model, X_test_sel)
        y_prob_val  = predict_lgbm(model, X_val_sel)
        m = compute_binary_metrics(y_test_fold, y_prob_test, CFG.classification_threshold)
        m["fold"] = fold
        fold_metrics.append(m)

        for i, idx in enumerate(test_idx):
            all_test_preds.append({"app_id": app_ids[idx], "y_true": int(y_test_fold[i]),
                                   "y_prob": float(y_prob_test[i])})
        for i, rel_idx in enumerate(inner_val):
            orig_idx = outer_train_idx[rel_idx]
            all_val_preds.append({"app_id": app_ids[orig_idx],
                                  "y_true": int(y_outer_train[rel_idx]),
                                  "y_prob": float(y_prob_val[i])})

        print(f"  Fold {fold}: f1={m['f1_pos']:.3f} auc={m['roc_auc']:.3f}")

    agg_drift_cv = aggregate_metrics(fold_metrics)

    print("\n[3/4] Evaluating on independent test set (N=110) with drift features...")
    test_text_path = Path(CFG.features_test_dir) / "text" / "features.npz"
    tdt = np.load(test_text_path, allow_pickle=True)
    new_kw_test = rebuild_keyword_features(CFG.inference_dataset_path, test_text_path)
    X_test_drift = np.concatenate([tdt["sbert"], new_kw_test, tdt["meta"]], axis=1)

    label_map = {}
    with open(CFG.inference_manual_csv) as f:
        for row in csv.DictReader(f):
            label_map[row["pkg_name"]] = int(row["label"])
    test_ids = list(tdt["app_ids"])
    y_test   = np.array([label_map.get(a, -1) for a in test_ids], dtype=int)

    probs_per_fold = []
    for sel, mdl in saved_models:
        probs_per_fold.append(mdl.predict(sel.transform(X_test_drift)))
    y_prob_test = np.mean(np.vstack(probs_per_fold), axis=0)
    metrics_test_drift = compute_binary_metrics(y_test, y_prob_test, CFG.classification_threshold)

    print("\n[4/4] Comparing with original text-only results...")
    orig_cv_path = base_dir / "text_only" / "metrics_aggregated.json"
    orig_test_path = base_dir / "independent_test" / "predictions_text_only.csv"

    orig_cv = {}
    if orig_cv_path.exists():
        with open(orig_cv_path) as f:
            orig_cv = json.load(f)

    orig_test_metrics = {}
    if orig_test_path.exists():
        yt, yp = [], []
        with open(orig_test_path, newline="") as f:
            for row in csv.DictReader(f):
                yt.append(int(row["y_true"]))
                yp.append(float(row["y_prob"]))
        orig_test_metrics = compute_binary_metrics(
            np.array(yt), np.array(yp), CFG.classification_threshold
        )

    results = {
        "cv_with_model_names": {
            "f1_pos_mean":   round(orig_cv.get("f1_pos_mean", 0), 4),
            "roc_auc_mean":  round(orig_cv.get("roc_auc_mean", 0), 4),
        },
        "cv_without_model_names": {
            "f1_pos_mean":   round(agg_drift_cv["f1_pos_mean"], 4),
            "roc_auc_mean":  round(agg_drift_cv["roc_auc_mean"], 4),
        },
        "indep_test_with_model_names": {
            "f1_pos":   round(orig_test_metrics.get("f1_pos", 0), 4),
            "roc_auc":  round(orig_test_metrics.get("roc_auc", 0), 4),
        },
        "indep_test_without_model_names": {
            "f1_pos":  round(metrics_test_drift["f1_pos"], 4),
            "roc_auc": round(metrics_test_drift["roc_auc"], 4),
        },
        "delta_cv_f1":        round(agg_drift_cv["f1_pos_mean"] - orig_cv.get("f1_pos_mean", 0), 4),
        "delta_cv_auc":       round(agg_drift_cv["roc_auc_mean"] - orig_cv.get("roc_auc_mean", 0), 4),
        "delta_test_f1":      round(metrics_test_drift["f1_pos"] - orig_test_metrics.get("f1_pos", 0), 4),
        "delta_test_auc":     round(metrics_test_drift["roc_auc"] - orig_test_metrics.get("roc_auc", 0), 4),
        "removed_keywords":   CFG.keyword_categories["model_name"],
    }

    write_json(out_dir / "keyword_drift_results.json", results)

    print("\nKEYWORD DRIFT SUMMARY")
    print(f"  {'Condition':<40} {'F1':>7} {'AUC':>7}")
    print("  " + "-" * 58)
    print(f"  {'CV — with model-name keywords':<40} "
          f"{results['cv_with_model_names']['f1_pos_mean']:>7.4f} "
          f"{results['cv_with_model_names']['roc_auc_mean']:>7.4f}")
    print(f"  {'CV — without model-name keywords':<40} "
          f"{results['cv_without_model_names']['f1_pos_mean']:>7.4f} "
          f"{results['cv_without_model_names']['roc_auc_mean']:>7.4f}")
    print(f"  {'  Δ (CV)':<40} {results['delta_cv_f1']:>+7.4f} {results['delta_cv_auc']:>+7.4f}")
    print()
    print(f"  {'Indep. test — with model-name keywords':<40} "
          f"{results['indep_test_with_model_names']['f1_pos']:>7.4f} "
          f"{results['indep_test_with_model_names']['roc_auc']:>7.4f}")
    print(f"  {'Indep. test — without model-name keywords':<40} "
          f"{results['indep_test_without_model_names']['f1_pos']:>7.4f} "
          f"{results['indep_test_without_model_names']['roc_auc']:>7.4f}")
    print(f"  {'  Δ (test)':<40} {results['delta_test_f1']:>+7.4f} {results['delta_test_auc']:>+7.4f}")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
