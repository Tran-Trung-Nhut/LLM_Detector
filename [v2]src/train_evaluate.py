"""
train_evaluate.py — Train and evaluate classifiers with 5-fold CV.

Three experiment modes run automatically:
  A) Text-only  classifier   (SBERT + keywords + meta)
  B) Image-only classifier   (CLIP + zero-shot + OCR keywords)
  C) Fusion classifier        (all features, late-fusion stacking)

For each, trains LightGBM (or XGBoost) per fold, reports per-fold and
aggregated metrics, and saves predictions + feature importances.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils.io import read_jsonl, write_json, write_predictions_csv
from utils.metrics import compute_binary_metrics
from utils.seed import set_seed


# ── Feature loading ──────────────────────────────────────────────────────────

def load_features():
    """Load and merge text + image features into one dict keyed by app_id."""
    text_path = Path(CFG.features_dir) / "text" / "features.npz"
    image_path = Path(CFG.features_dir) / "image" / "features.npz"

    td = np.load(text_path, allow_pickle=True)
    imd = np.load(image_path, allow_pickle=True)

    app_ids = list(td["app_ids"])
    labels = td["labels"]

    # text features
    sbert = td["sbert"]        # (N, 1024)
    keywords = td["keywords"]  # (N, 13)
    meta = td["meta"]          # (N, 21)

    # image features
    clip_mean = imd["clip_mean"]  # (N, 768)
    clip_max = imd["clip_max"]    # (N, 768)
    zeroshot = imd["zeroshot"]    # (N, ~12)
    ocr = imd["ocr"]             # (N, 15)

    assert list(td["app_ids"]) == list(imd["app_ids"]), "Feature files must have same app order"

    # Build feature groups
    text_feats = np.concatenate([sbert, keywords, meta], axis=1)
    image_feats = np.concatenate([clip_mean, clip_max, zeroshot, ocr], axis=1)
    all_feats = np.concatenate([text_feats, image_feats], axis=1)

    id2idx = {aid: i for i, aid in enumerate(app_ids)}

    return {
        "app_ids": app_ids,
        "labels": labels,
        "id2idx": id2idx,
        "text_feats": text_feats,
        "image_feats": image_feats,
        "all_feats": all_feats,
        "sbert": sbert,
        "keywords": keywords,
        "meta": meta,
        "clip_mean": clip_mean,
        "clip_max": clip_max,
        "zeroshot": zeroshot,
        "ocr": ocr,
    }


def load_split(fold: int):
    split_path = Path(CFG.splits_dir) / f"fold_{fold}.json"
    with open(split_path) as f:
        return json.load(f)


# ── LightGBM classifier ─────────────────────────────────────────────────────

# LightGBM params are now provided centrally via `CFG.lgbm_params` in config.py


def train_lgbm(X_train, y_train, X_val, y_val, num_rounds=500):
    """Train a LightGBM model with early stopping."""
    params = dict(CFG.lgbm_params)
    if params.get("seed") is None:
        params["seed"] = CFG.seed
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model = lgb.train(
        params, dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


def predict_lgbm(model, X):
    return model.predict(X)


# ── Stacking fusion ─────────────────────────────────────────────────────────

def train_stacking_fusion(text_probs_train, image_probs_train, y_train,
                          text_probs_val, image_probs_val):
    """
    Meta-learner: Logistic Regression stacking text & image probabilities.
    Input: probability predictions from text-only and image-only models.
    """
    X_meta_train = np.column_stack([text_probs_train, image_probs_train])
    X_meta_val = np.column_stack([text_probs_val, image_probs_val])

    scaler = StandardScaler()
    X_meta_train = scaler.fit_transform(X_meta_train)
    X_meta_val = scaler.transform(X_meta_val)

    meta_clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=CFG.seed)
    meta_clf.fit(X_meta_train, y_train)
    return meta_clf.predict_proba(X_meta_val)[:, 1], meta_clf, scaler


# ── Per-fold training & evaluation ───────────────────────────────────────────

def run_single_experiment(name: str, X_all: np.ndarray, data: dict, run_dir: Path, k_features: int = 100):
    """
    Train and evaluate a single feature set across all folds.
    Returns aggregated metrics and per-fold predictions.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    id2idx = data["id2idx"]
    labels = data["labels"]

    all_preds = []
    fold_metrics = []

    for fold in range(CFG.n_folds):
        split = load_split(fold)
        train_idx = [id2idx[aid] for aid in split["train_ids"] if aid in id2idx]
        test_idx = [id2idx[aid] for aid in split["test_ids"] if aid in id2idx]

        X_train, y_train = X_all[train_idx], labels[train_idx]
        X_test, y_test = X_all[test_idx], labels[test_idx]

        # ─── BẮT ĐẦU: FEATURE SELECTION ───
        # Chọn k đặc trưng tốt nhất. Nếu số feature ban đầu < k thì lấy toàn bộ.
        actual_k = min(k_features, X_train.shape[1])
        if actual_k < X_train.shape[1]:
            # Dùng f_classif (ANOVA F-value) để chấm điểm mức độ quan trọng
            selector = SelectKBest(score_func=f_classif, k=actual_k)
            # CHÚ Ý: Chỉ fit trên tập Train để tránh Data Leakage
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
        # ─── KẾT THÚC: FEATURE SELECTION ───

        model = train_lgbm(X_train, y_train, X_test, y_test)
        y_prob = predict_lgbm(model, X_test)

        metrics = compute_binary_metrics(y_test, y_prob, threshold=0.5)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

        for i, idx in enumerate(test_idx):
            all_preds.append({
                "app_id": data["app_ids"][idx],
                "fold": fold,
                "y_true": int(y_test[i]),
                "y_prob": float(y_prob[i]),
            })

        # Cập nhật phần lưu Feature Importance để biết chính xác cột nào được giữ lại
        if fold == 0 and hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            
            # Lấy index gốc của các feature đã được giữ lại
            if actual_k < X_all.shape[1]:
                selected_indices = selector.get_support(indices=True)
            else:
                selected_indices = np.arange(X_all.shape[1])

            write_json(run_dir / "feature_importance_fold0.json", {
                "importance_gain": imp.tolist(),
                "selected_original_indices": selected_indices.tolist() # Rất quan trọng cho bài báo
            })

        print(f"  Fold {fold}: acc={metrics['accuracy']:.3f} "
              f"f1={metrics['f1_pos']:.3f} auc={metrics['roc_auc']:.3f}")

    # Aggregate
    agg = aggregate_metrics(fold_metrics)
    write_json(run_dir / "metrics_per_fold.json", fold_metrics)
    write_json(run_dir / "metrics_aggregated.json", agg)
    write_predictions_csv(run_dir / "predictions.csv", all_preds)

    print(f"  ── {name} AGGREGATE: acc={agg['accuracy_mean']:.3f}±{agg['accuracy_std']:.3f} "
          f"f1={agg['f1_pos_mean']:.3f}±{agg['f1_pos_std']:.3f} "
          f"auc={agg['roc_auc_mean']:.3f}±{agg['roc_auc_std']:.3f}")
    return all_preds, fold_metrics

def aggregate_metrics(fold_metrics: list[dict]) -> dict:
    """Average metrics across folds."""
    keys = ["accuracy", "precision_pos", "recall_pos", "f1_pos", "macro_f1", "pr_auc", "roc_auc"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


# ── Fusion experiment ────────────────────────────────────────────────────────

def run_fusion_experiment(data: dict, run_dir: Path):
    """
    Three sub-experiments + stacking/voting fusion.
    Fusion A: concatenate all features → single LightGBM  (early fusion)
    Fusion B: stack text-only & image-only probabilities (late fusion)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    id2idx = data["id2idx"]
    labels = data["labels"]

    # ── Early fusion (all features → LightGBM) ──
    print("\n[C1] Early Fusion (all features → LightGBM)")
    # Vẫn gọi hàm run_single_experiment đã có SelectKBest ở trên
    run_single_experiment("EarlyFusion", data["all_feats"], data, run_dir / "early_fusion")

    # ── Late fusion (Stacking hoặc Soft Voting) ──
    print(f"\n[C2] Late Fusion (Strategy: {CFG.fusion_strategy})")
    stacking_dir = run_dir / "late_fusion"
    stacking_dir.mkdir(parents=True, exist_ok=True)

    all_preds_stack = []
    fold_metrics_stack = []

    from sklearn.feature_selection import SelectKBest, f_classif

    for fold in range(CFG.n_folds):
        split = load_split(fold)
        train_idx = [id2idx[aid] for aid in split["train_ids"] if aid in id2idx]
        test_idx = [id2idx[aid] for aid in split["test_ids"] if aid in id2idx]

        X_text_tr, y_tr = data["text_feats"][train_idx], labels[train_idx]
        X_text_te, y_te = data["text_feats"][test_idx], labels[test_idx]
        
        X_img_tr = data["image_feats"][train_idx]
        X_img_te = data["image_feats"][test_idx]


        text_selector = SelectKBest(score_func=f_classif, k=min(100, X_text_tr.shape[1]))
        X_text_tr = text_selector.fit_transform(X_text_tr, y_tr)
        X_text_te = text_selector.transform(X_text_te)

        img_selector = SelectKBest(score_func=f_classif, k=min(100, X_img_tr.shape[1]))
        X_img_tr = img_selector.fit_transform(X_img_tr, y_tr)
        X_img_te = img_selector.transform(X_img_te)


        text_model = train_lgbm(X_text_tr, y_tr, X_text_te, y_te)
        text_prob_te = text_model.predict(X_text_te)

        img_model = train_lgbm(X_img_tr, y_tr, X_img_te, y_te)
        img_prob_te = img_model.predict(X_img_te)

        if CFG.fusion_strategy == "stacking":
            text_prob_tr = text_model.predict(X_text_tr)
            img_prob_tr = img_model.predict(X_img_tr)

            y_prob, _, _ = train_stacking_fusion(
                text_prob_tr, img_prob_tr, y_tr,
                text_prob_te, img_prob_te,
            )
        elif CFG.fusion_strategy == "soft_voting":
            y_prob = (text_prob_te + img_prob_te) / 2.0
        elif CFG.fusion_strategy == "max_voting":
            y_prob = np.maximum(text_prob_te, img_prob_te)
        else:
            raise ValueError(f"Unknown fusion strategy: {CFG.fusion_strategy}")

        metrics = compute_binary_metrics(y_te, y_prob, threshold=0.5)
        metrics["fold"] = fold
        fold_metrics_stack.append(metrics)

        for i, idx in enumerate(test_idx):
            all_preds_stack.append({
                "app_id": data["app_ids"][idx],
                "fold": fold,
                "y_true": int(y_te[i]),
                "y_prob": float(y_prob[i]),
                "text_prob": float(text_prob_te[i]),
                "image_prob": float(img_prob_te[i]),
            })

        print(f"  Fold {fold}: acc={metrics['accuracy']:.3f} "
              f"f1={metrics['f1_pos']:.3f} auc={metrics['roc_auc']:.3f}")

    agg = aggregate_metrics(fold_metrics_stack)
    write_json(stacking_dir / "metrics_per_fold.json", fold_metrics_stack)
    write_json(stacking_dir / "metrics_aggregated.json", agg)
    write_predictions_csv(stacking_dir / "predictions.csv", all_preds_stack)

    print(f"  ── LateFusion AGGREGATE: acc={agg['accuracy_mean']:.3f}±{agg['accuracy_std']:.3f} "
          f"f1={agg['f1_pos_mean']:.3f}±{agg['f1_pos_std']:.3f} "
          f"auc={agg['roc_auc_mean']:.3f}±{agg['roc_auc_std']:.3f}")


# ── Threshold search ─────────────────────────────────────────────────────────

def find_best_threshold(preds_csv: Path) -> dict:
    """Grid-search best threshold on aggregated predictions."""
    df = pd.read_csv(preds_csv)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.30, 0.71, 0.01):
        m = compute_binary_metrics(df["y_true"].values, df["y_prob"].values, threshold=t)
        if m["f1_pos"] > best_f1:
            best_f1 = m["f1_pos"]
            best_t = t
    best_metrics = compute_binary_metrics(df["y_true"].values, df["y_prob"].values, threshold=best_t)
    best_metrics["best_threshold"] = float(best_t)
    return best_metrics


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    set_seed(CFG.seed)
    base_dir = Path(CFG.runs_dir) / CFG.run_name

    print("Loading features ...")
    data = load_features()
    n = len(data["app_ids"])
    print(f"  {n} apps, text_feats={data['text_feats'].shape[1]}d, "
          f"image_feats={data['image_feats'].shape[1]}d, "
          f"all_feats={data['all_feats'].shape[1]}d")

    # Experiment A: Text-only
    print("\n" + "=" * 60)
    print("[A] Text-Only Classifier")
    print("=" * 60)
    run_single_experiment("TextOnly", data["text_feats"], data, base_dir / "text_only")

    # Experiment B: Image-only
    print("\n" + "=" * 60)
    print("[B] Image-Only Classifier")
    print("=" * 60)
    run_single_experiment("ImageOnly", data["image_feats"], data, base_dir / "image_only")

    # Experiment C: Fusion
    print("\n" + "=" * 60)
    print("[C] Fusion Classifiers")
    print("=" * 60)
    run_fusion_experiment(data, base_dir / "fusion")

    # Threshold search on best models
    print("\n" + "=" * 60)
    print("Threshold search on predictions ...")
    for sub in ["text_only", "image_only", "fusion/early_fusion", "fusion/late_fusion"]:
        csv_path = base_dir / sub / "predictions.csv"
        if csv_path.exists():
            best = find_best_threshold(csv_path)
            write_json(csv_path.parent / "best_threshold_metrics.json", best)
            print(f"  {sub}: best_t={best['best_threshold']:.2f} "
                  f"acc={best['accuracy']:.3f} f1={best['f1_pos']:.3f}")

    print("\nDone! Results saved to:", base_dir)


if __name__ == "__main__":
    main()
