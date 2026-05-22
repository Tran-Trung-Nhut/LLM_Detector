"""independent_test_eval.py — Evaluates the trained 5-fold ensemble on the independent test set (Tables 12 and 20)."""
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json
from utils.metrics import compute_binary_metrics


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    lo = float(beta_dist.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return round(lo, 3), round(hi, 3)


def load_features_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    return list(d["app_ids"]), d["labels"], d


def main():
    import csv
    import json
    import joblib
    import lightgbm as lgb

    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "independent_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_text_path = Path(CFG.features_test_dir) / "text" / "features.npz"
    test_img_path  = Path(CFG.features_test_dir) / "image" / "features.npz"
    if not test_text_path.exists():
        print(f"[error] Test features not found at {test_text_path}")
        print("  Extract features for the 110-app test set first.")
        return

    test_ids, test_labels, td = load_features_npz(test_text_path)
    _, _, imd = load_features_npz(test_img_path)

    text_feats  = np.concatenate([td["sbert"], td["keywords"], td["meta"]], axis=1)
    image_feats = np.concatenate([imd["clip_mean"], imd["clip_max"], imd["zeroshot"], imd["ocr"]], axis=1)
    all_feats   = np.concatenate([text_feats, image_feats], axis=1)

    n_folds  = CFG.n_folds
    base_fus = base_dir / "fusion"

    text_probs_folds  = []
    img_probs_folds   = []
    ef_probs_folds    = []
    stack_probs_folds = []

    for fold in range(n_folds):
        sel_t    = joblib.load(base_fus / "base_models_saved" / f"text_selector_fold_{fold}.joblib")
        mdl_t    = lgb.Booster(model_file=str(base_fus / "base_models_saved" / f"text_lgbm_fold_{fold}.txt"))
        p_text_k = mdl_t.predict(sel_t.transform(text_feats))
        text_probs_folds.append(p_text_k)

        sel_i   = joblib.load(base_fus / "base_models_saved" / f"img_selector_fold_{fold}.joblib")
        mdl_i   = lgb.Booster(model_file=str(base_fus / "base_models_saved" / f"img_lgbm_fold_{fold}.txt"))
        p_img_k = mdl_i.predict(sel_i.transform(image_feats))
        img_probs_folds.append(p_img_k)

        ef_model_path = base_fus / "early_fusion" / "saved_models" / f"lgbm_fold_{fold}.txt"
        ef_sel_path   = base_fus / "early_fusion" / "saved_models" / f"selector_fold_{fold}.joblib"
        if ef_model_path.exists() and ef_sel_path.exists():
            sel_ef = joblib.load(ef_sel_path)
            mdl_ef = lgb.Booster(model_file=str(ef_model_path))
            ef_probs_folds.append(mdl_ef.predict(sel_ef.transform(all_feats)))

        meta_path   = base_fus / "late_fusion_stacking" / "saved_models" / f"meta_clf_fold_{fold}.joblib"
        scaler_path = base_fus / "late_fusion_stacking" / "saved_models" / f"scaler_fold_{fold}.joblib"
        if meta_path.exists() and scaler_path.exists():
            meta_clf  = joblib.load(meta_path)
            scaler    = joblib.load(scaler_path)
            X_meta_k  = np.column_stack([p_text_k, p_img_k])
            stack_probs_folds.append(meta_clf.predict_proba(scaler.transform(X_meta_k))[:, 1])

    text_probs = np.mean(np.vstack(text_probs_folds), axis=0)
    img_probs  = np.mean(np.vstack(img_probs_folds),  axis=0)
    ef_probs   = np.mean(np.vstack(ef_probs_folds),   axis=0) if ef_probs_folds else np.zeros(len(test_ids))

    score_max_probs = np.mean(
        np.vstack([np.maximum(t, i) for t, i in zip(text_probs_folds, img_probs_folds)]),
        axis=0,
    )

    stack_probs = np.mean(np.vstack(stack_probs_folds), axis=0) if stack_probs_folds else np.zeros(len(test_ids))

    sv_alpha_path = base_fus / "late_fusion_soft_voting" / "alpha_grid_search.json"
    alpha = 0.5
    if sv_alpha_path.exists():
        with open(sv_alpha_path) as f:
            alpha = json.load(f).get("mean_alpha", 0.5)
    soft_voting_probs = alpha * text_probs + (1.0 - alpha) * img_probs

    label_csv = Path(CFG.inference_manual_csv)
    if label_csv.exists():
        import csv as _csv
        label_map = {}
        with open(label_csv) as f:
            for row in _csv.DictReader(f):
                label_map[row["pkg_name"]] = int(row["label"])
        y_true = np.array([label_map.get(aid, int(test_labels[i])) for i, aid in enumerate(test_ids)], dtype=int)
    else:
        print(f"[warn] {label_csv} not found — using labels from NPZ (may be -1 placeholders)")
        y_true = test_labels.astype(int)

    n_pos  = int((y_true == 1).sum())
    n_neg  = int((y_true == 0).sum())
    tau    = CFG.classification_threshold

    strategies = {
        "Text-Only":    text_probs,
        "Image-Only":   img_probs,
        "Early Fusion": ef_probs,
        "Score-Max":    score_max_probs,
        "Soft Voting":  soft_voting_probs,
        "Stacking":     stack_probs,
    }

    results = {}
    all_pred_rows = []

    print(f"\nIndependent Test Set: N={len(y_true)}, N_+={n_pos}, N_-={n_neg}")
    print(f"\n{'Strategy':<16} {'Acc':>6} {'Prec':>6} {'Prec CI':>16} {'Recall':>7} {'Recall CI':>16} {'F1':>6} {'AUC':>7} {'AP':>6}")
    print("-" * 95)

    for name, y_prob in strategies.items():
        m = compute_binary_metrics(y_true, y_prob, threshold=tau)
        tp   = int(np.sum((y_prob >= tau) & (y_true == 1)))
        fp   = int(np.sum((y_prob >= tau) & (y_true == 0)))
        r_lo, r_hi = clopper_pearson_ci(tp, n_pos)
        p_lo, p_hi = clopper_pearson_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 1.0)

        results[name] = {
            **m,
            "recall_ci95":    [r_lo, r_hi],
            "precision_ci95": [p_lo, p_hi],
        }
        print(f"  {name:<14} {m['accuracy']:>6.3f} {m['precision_pos']:>6.3f} "
              f"[{p_lo:.3f},{p_hi:.3f}] {m['recall_pos']:>7.3f} [{r_lo:.3f},{r_hi:.3f}] "
              f"{m['f1_pos']:>6.3f} {m['roc_auc']:>7.3f} {m['pr_auc']:>6.3f}")

        for i, aid in enumerate(test_ids):
            all_pred_rows.append({
                "app_id":     aid,
                "y_true":     int(y_true[i]),
                "y_prob":     round(float(y_prob[i]), 6),
                "strategy":   name,
                "text_prob":  round(float(text_probs[i]), 6),
                "image_prob": round(float(img_probs[i]), 6),
            })

    write_json(out_dir / "table20_independent_test.json", results)

    for name in strategies:
        rows = [r for r in all_pred_rows if r["strategy"] == name]
        slug = name.lower().replace(" ", "_").replace("-", "_")
        csv_path = out_dir / f"predictions_{slug}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["app_id", "y_true", "y_prob", "text_prob", "image_prob"])
            writer.writeheader()
            writer.writerows([{k: r[k] for k in ["app_id", "y_true", "y_prob", "text_prob", "image_prob"]} for r in rows])

    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
