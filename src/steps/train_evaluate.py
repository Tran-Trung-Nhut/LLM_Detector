"""train_evaluate.py — 5-fold CV training and evaluation for all fusion strategies."""
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json, write_predictions_csv
from utils.metrics import compute_binary_metrics
from utils.seed import set_seed


def load_features():
    text_path = Path(CFG.features_dir) / "text" / "features.npz"
    image_path = Path(CFG.features_dir) / "image" / "features.npz"

    td = np.load(text_path, allow_pickle=True)
    imd = np.load(image_path, allow_pickle=True)

    app_ids = list(td["app_ids"])
    labels = td["labels"]

    sbert    = td["sbert"]
    keywords = td["keywords"]
    meta     = td["meta"]

    clip_mean = imd["clip_mean"]
    clip_max  = imd["clip_max"]
    zeroshot  = imd["zeroshot"]
    ocr       = imd["ocr"]

    assert list(td["app_ids"]) == list(imd["app_ids"]), "Feature files must have same app order"
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


def load_slm_score(app_ids: list) -> Optional[np.ndarray]:
    """Load SLM features for ablation comparison only. Returns None if not cached."""
    slm_path = Path(CFG.features_dir) / "slm" / "features.npz"
    if not slm_path.exists():
        return None
    slmd = np.load(slm_path, allow_pickle=True)
    assert list(slmd["app_ids"]) == app_ids, "SLM feature order mismatch"
    return slmd["slm_score"]  # (N, 1)


def load_split(fold: int):
    split_path = Path(CFG.splits_dir) / f"fold_{fold}.json"
    with open(split_path) as f:
        return json.load(f)


def make_inner_split_indices(y: np.ndarray, fold: int, val_size: Optional[float] = None):
    y = np.asarray(y).astype(int)
    if val_size is None:
        val_size = CFG.inner_val_ratio
    if not (0.0 < val_size < 1.0):
        raise ValueError("CFG.inner_val_ratio must be in (0, 1).")

    idx = np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    stratify = y if (len(classes) > 1 and counts.min() >= 2) else None

    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        shuffle=True,
        random_state=CFG.seed + fold,
        stratify=stratify,
    )
    return np.sort(train_idx), np.sort(val_idx)


def resolve_inner_cv_splits(y: np.ndarray, requested_splits: int) -> int:
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Inner CV requires at least 2 classes.")
    max_splits = int(min(len(y), counts.min()))
    n_splits = min(requested_splits, max_splits)
    if n_splits < 2:
        raise ValueError("Not enough samples to run inner CV safely.")
    return n_splits


def find_best_threshold_from_arrays(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0:
        raise ValueError("Threshold search received empty validation predictions.")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Threshold search requires both classes in validation predictions.")

    best_f1, best_t = -1.0, CFG.classification_threshold
    for t in np.arange(CFG.threshold_search_min, CFG.threshold_search_max + 1e-9, CFG.threshold_search_step):
        m = compute_binary_metrics(y_true, y_prob, threshold=float(t))
        if m["f1_pos"] > best_f1:
            best_f1 = m["f1_pos"]
            best_t = float(t)

    best_metrics = compute_binary_metrics(y_true, y_prob, threshold=best_t)
    best_metrics["best_threshold"] = best_t
    return best_metrics


def train_lgbm(X_train, y_train, X_val, y_val, num_rounds=None):
    if num_rounds is None:
        num_rounds = CFG.lgbm_num_rounds
    params = dict(CFG.lgbm_params)
    if params.get("seed") is None:
        params["seed"] = CFG.seed

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    callbacks = [
        lgb.early_stopping(stopping_rounds=CFG.lgbm_early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


def predict_lgbm(model, X):
    return model.predict(X)


def fit_select_kbest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    X_test: np.ndarray,
    k_features: int,
):
    actual_k = min(k_features, X_train.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=actual_k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_eval_sel = selector.transform(X_eval)
    X_test_sel = selector.transform(X_test)
    return selector, X_train_sel, X_eval_sel, X_test_sel


def run_single_experiment(
    name: str, X_all: np.ndarray, data: dict, run_dir: Path,
    k_features: Optional[int] = None,
):
    if k_features is None:
        k_features = CFG.feature_selection_k
    run_dir.mkdir(parents=True, exist_ok=True)

    models_dir = run_dir / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    id2idx = data["id2idx"]
    labels = data["labels"]

    all_preds = []
    fold_metrics = []
    val_preds = []

    for fold in range(CFG.n_folds):
        split = load_split(fold)
        outer_train_idx = [id2idx[aid] for aid in split["train_ids"] if aid in id2idx]
        test_idx = [id2idx[aid] for aid in split["test_ids"] if aid in id2idx]

        X_outer_train, y_outer_train = X_all[outer_train_idx], labels[outer_train_idx]
        X_test, y_test = X_all[test_idx], labels[test_idx]

        inner_train_rel, inner_val_rel = make_inner_split_indices(y_outer_train, fold)
        X_train = X_outer_train[inner_train_rel]
        y_train = y_outer_train[inner_train_rel]
        X_val = X_outer_train[inner_val_rel]
        y_val = y_outer_train[inner_val_rel]

        selector, X_train_sel, X_val_sel, X_test_sel = fit_select_kbest(
            X_train, y_train, X_val, X_test, k_features
        )
        joblib.dump(selector, models_dir / f"selector_fold_{fold}.joblib")

        model = train_lgbm(X_train_sel, y_train, X_val_sel, y_val)
        model.save_model(str(models_dir / f"lgbm_fold_{fold}.txt"))

        y_prob_test = predict_lgbm(model, X_test_sel)
        y_prob_val = predict_lgbm(model, X_val_sel)

        metrics = compute_binary_metrics(y_test, y_prob_test, threshold=CFG.classification_threshold)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

        for i, idx in enumerate(test_idx):
            all_preds.append({
                "app_id": data["app_ids"][idx],
                "fold": fold,
                "y_true": int(y_test[i]),
                "y_prob": float(y_prob_test[i]),
            })

        for i, rel_idx in enumerate(inner_val_rel):
            orig_idx = outer_train_idx[rel_idx]
            val_preds.append({
                "app_id": data["app_ids"][orig_idx],
                "fold": fold,
                "y_true": int(y_val[i]),
                "y_prob": float(y_prob_val[i]),
            })

        if fold == 0 and hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            selected_indices = selector.get_support(indices=True)
            write_json(run_dir / "feature_importance_fold0.json", {
                "importance_gain": imp.tolist(),
                "selected_original_indices": selected_indices.tolist(),
            })

        print(
            f"  Fold {fold}: acc={metrics['accuracy']:.3f} "
            f"f1={metrics['f1_pos']:.3f} auc={metrics['roc_auc']:.3f}"
        )

    agg = aggregate_metrics(fold_metrics)
    write_json(run_dir / "metrics_per_fold.json", fold_metrics)
    write_json(run_dir / "metrics_aggregated.json", agg)
    write_predictions_csv(run_dir / "predictions.csv", all_preds)
    write_predictions_csv(run_dir / "validation_predictions.csv", val_preds)

    best = find_best_threshold_from_arrays(
        np.array([r["y_true"] for r in val_preds], dtype=int),
        np.array([r["y_prob"] for r in val_preds], dtype=float),
    )
    write_json(run_dir / "best_threshold_metrics.json", best)

    print(
        f"  ── {name} AGGREGATE: acc={agg['accuracy_mean']:.3f}±{agg['accuracy_std']:.3f} "
        f"f1={agg['f1_pos_mean']:.3f}±{agg['f1_pos_std']:.3f} "
        f"auc={agg['roc_auc_mean']:.3f}±{agg['roc_auc_std']:.3f}"
    )
    print(f"  ── {name} VALIDATION BEST THRESHOLD: {best['best_threshold']:.2f} (f1={best['f1_pos']:.3f})")
    return all_preds, fold_metrics


def aggregate_metrics(fold_metrics: list[dict]) -> dict:
    keys = ["accuracy", "precision_pos", "recall_pos", "f1_pos", "macro_f1", "pr_auc", "roc_auc"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


def build_oof_and_test_probs(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    X_test: np.ndarray,
    fold: int,
    k_features: int,
):
    n_splits = resolve_inner_cv_splits(y_outer_train, CFG.stacking_inner_cv_folds)
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.seed + fold)

    oof_probs = np.zeros(len(y_outer_train), dtype=np.float64)
    test_fold_probs = []

    for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
        X_train = X_outer_train[inner_train_idx]
        y_train = y_outer_train[inner_train_idx]
        X_val = X_outer_train[inner_val_idx]
        y_val = y_outer_train[inner_val_idx]

        selector, X_train_sel, X_val_sel, X_test_sel = fit_select_kbest(
            X_train, y_train, X_val, X_test, k_features
        )
        model = train_lgbm(X_train_sel, y_train, X_val_sel, y_val)

        oof_probs[inner_val_idx] = predict_lgbm(model, X_val_sel)
        test_fold_probs.append(predict_lgbm(model, X_test_sel))

    test_probs = np.mean(np.vstack(test_fold_probs), axis=0)
    return oof_probs, test_probs


def train_and_save_base_model(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    fold: int,
    models_dir: Path,
    selector_name: str,
    model_name: str,
    k_features: int,
):
    inner_train_rel, inner_val_rel = make_inner_split_indices(y_outer_train, fold)
    X_train = X_outer_train[inner_train_rel]
    y_train = y_outer_train[inner_train_rel]
    X_val = X_outer_train[inner_val_rel]
    y_val = y_outer_train[inner_val_rel]

    selector, X_train_sel, X_val_sel, _ = fit_select_kbest(
        X_train, y_train, X_val, X_val, k_features
    )
    model = train_lgbm(X_train_sel, y_train, X_val_sel, y_val)

    joblib.dump(selector, models_dir / selector_name)
    model.save_model(str(models_dir / model_name))


def run_fusion_experiment(data: dict, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)

    base_models_dir = run_dir / "base_models_saved"
    base_models_dir.mkdir(parents=True, exist_ok=True)

    id2idx = data["id2idx"]
    labels = data["labels"]

    print("\n[C1] Early Fusion")
    run_single_experiment("EarlyFusion", data["all_feats"], data, run_dir / "early_fusion")

    print(f"\n[C2] Late Fusion — building base models...")
    fold_predictions = []

    for fold in range(CFG.n_folds):
        split = load_split(fold)
        train_idx = [id2idx[aid] for aid in split["train_ids"] if aid in id2idx]
        test_idx = [id2idx[aid] for aid in split["test_ids"] if aid in id2idx]

        X_text_outer_train = data["text_feats"][train_idx]
        X_img_outer_train = data["image_feats"][train_idx]
        y_outer_train = labels[train_idx]

        X_text_test = data["text_feats"][test_idx]
        X_img_test = data["image_feats"][test_idx]
        y_test = labels[test_idx]

        text_prob_meta, text_prob_test = build_oof_and_test_probs(
            X_text_outer_train, y_outer_train, X_text_test, fold,
            k_features=CFG.feature_selection_k
        )
        img_prob_meta, img_prob_test = build_oof_and_test_probs(
            X_img_outer_train, y_outer_train, X_img_test, fold,
            k_features=CFG.feature_selection_k
        )

        train_and_save_base_model(
            X_text_outer_train, y_outer_train, fold, base_models_dir,
            selector_name=f"text_selector_fold_{fold}.joblib",
            model_name=f"text_lgbm_fold_{fold}.txt",
            k_features=CFG.feature_selection_k,
        )
        train_and_save_base_model(
            X_img_outer_train, y_outer_train, fold, base_models_dir,
            selector_name=f"img_selector_fold_{fold}.joblib",
            model_name=f"img_lgbm_fold_{fold}.txt",
            k_features=CFG.feature_selection_k,
        )

        meta_fit_idx, meta_val_idx = make_inner_split_indices(y_outer_train, fold)
        fold_predictions.append({
            "fold": fold,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "y_meta": y_outer_train,
            "y_test": y_test,
            "meta_fit_idx": meta_fit_idx,
            "meta_val_idx": meta_val_idx,
            "text_prob_meta": text_prob_meta,
            "img_prob_meta": img_prob_meta,
            "text_prob_test": text_prob_test,
            "img_prob_test": img_prob_test,
        })
        print(f"  Fold {fold}: Base OOF predictions built & models saved")

    print(f"\n[C2] Late Fusion - Testing {len(CFG.fusion_strategy)} strategies...")
    for strategy in CFG.fusion_strategy:
        print(f"\n  → Strategy: {strategy.upper()}")
        strategy_dir = run_dir / f"late_fusion_{strategy}"
        strategy_dir.mkdir(parents=True, exist_ok=True)

        meta_models_dir = strategy_dir / "saved_models"
        if strategy == "stacking":
            meta_models_dir.mkdir(parents=True, exist_ok=True)

        all_preds = []
        val_preds = []
        fold_metrics = []

        for fold_data in fold_predictions:
            fold = fold_data["fold"]
            train_idx = fold_data["train_idx"]
            test_idx = fold_data["test_idx"]

            y_meta = fold_data["y_meta"]
            y_test = fold_data["y_test"]
            meta_fit_idx = fold_data["meta_fit_idx"]
            meta_val_idx = fold_data["meta_val_idx"]

            text_prob_meta = fold_data["text_prob_meta"]
            img_prob_meta = fold_data["img_prob_meta"]
            text_prob_test = fold_data["text_prob_test"]
            img_prob_test = fold_data["img_prob_test"]

            if strategy == "stacking":
                X_meta = np.column_stack([text_prob_meta, img_prob_meta])
                X_meta_fit = X_meta[meta_fit_idx]
                y_meta_fit = y_meta[meta_fit_idx]
                X_meta_val = X_meta[meta_val_idx]
                y_meta_val = y_meta[meta_val_idx]
                X_meta_test = np.column_stack([text_prob_test, img_prob_test])

                scaler = StandardScaler()
                X_meta_fit_scaled = scaler.fit_transform(X_meta_fit)
                X_meta_val_scaled = scaler.transform(X_meta_val)
                X_meta_test_scaled = scaler.transform(X_meta_test)

                meta_clf = LogisticRegression(
                    C=CFG.meta_learner_C,
                    solver="lbfgs",
                    max_iter=CFG.meta_learner_max_iter,
                    random_state=CFG.seed,
                )
                meta_clf.fit(X_meta_fit_scaled, y_meta_fit)

                y_prob = meta_clf.predict_proba(X_meta_test_scaled)[:, 1]
                y_prob_val = meta_clf.predict_proba(X_meta_val_scaled)[:, 1]

                joblib.dump(meta_clf, meta_models_dir / f"meta_clf_fold_{fold}.joblib")
                joblib.dump(scaler, meta_models_dir / f"scaler_fold_{fold}.joblib")

                if fold == 0:
                    weights = meta_clf.coef_[0]
                    intercept = meta_clf.intercept_[0]
                    write_json(strategy_dir / "meta_learner_weights.json", {
                        "text_weight": float(weights[0]),
                        "image_weight": float(weights[1]),
                        "intercept": float(intercept),
                        "interpretation": f"Text: {weights[0]:.3f}, Image: {weights[1]:.3f}",
                    })
            elif strategy == "soft_voting":
                _best_alpha, _best_val_f1 = 0.5, -1.0
                for _a in CFG.soft_voting_alpha_candidates:
                    _vp = _a * text_prob_meta[meta_val_idx] + (1.0 - _a) * img_prob_meta[meta_val_idx]
                    _m = compute_binary_metrics(
                        y_meta[meta_val_idx], _vp, threshold=CFG.classification_threshold
                    )
                    if _m["f1_pos"] > _best_val_f1:
                        _best_val_f1, _best_alpha = _m["f1_pos"], _a
                y_prob = _best_alpha * text_prob_test + (1.0 - _best_alpha) * img_prob_test
                y_prob_val = _best_alpha * text_prob_meta[meta_val_idx] + (1.0 - _best_alpha) * img_prob_meta[meta_val_idx]
                y_meta_val = y_meta[meta_val_idx]
            elif strategy == "score_max":
                y_prob = np.maximum(text_prob_test, img_prob_test)
                y_prob_val = np.maximum(text_prob_meta[meta_val_idx], img_prob_meta[meta_val_idx])
                y_meta_val = y_meta[meta_val_idx]
            else:
                raise ValueError(f"Unknown fusion strategy: {strategy}")

            metrics = compute_binary_metrics(y_test, y_prob, threshold=CFG.classification_threshold)
            metrics["fold"] = fold
            if strategy == "soft_voting":
                metrics["best_alpha"] = _best_alpha
            fold_metrics.append(metrics)

            for i, idx in enumerate(test_idx):
                all_preds.append({
                    "app_id": data["app_ids"][idx],
                    "fold": fold,
                    "y_true": int(y_test[i]),
                    "y_prob": float(y_prob[i]),
                    "text_prob": float(text_prob_test[i]),
                    "image_prob": float(img_prob_test[i]),
                })

            for i, rel_idx in enumerate(meta_val_idx):
                orig_idx = train_idx[rel_idx]
                val_preds.append({
                    "app_id": data["app_ids"][orig_idx],
                    "fold": fold,
                    "y_true": int(y_meta_val[i]),
                    "y_prob": float(y_prob_val[i]),
                })

            _alpha_info = f"  α={metrics['best_alpha']:.1f}" if "best_alpha" in metrics else ""
            print(
                f"    Fold {fold}: acc={metrics['accuracy']:.3f} "
                f"f1={metrics['f1_pos']:.3f} auc={metrics['roc_auc']:.3f}{_alpha_info}"
            )

        agg = aggregate_metrics(fold_metrics)
        if strategy == "soft_voting":
            _alphas = [m.get("best_alpha", 0.5) for m in fold_metrics]
            write_json(strategy_dir / "alpha_grid_search.json", {
                "alpha_per_fold": _alphas,
                "mean_alpha": round(float(np.mean(_alphas)), 2),
                "alpha_candidates": list(CFG.soft_voting_alpha_candidates),
            })
            print(f"    ── SOFT_VOTING alpha per fold: {_alphas}  mean={np.mean(_alphas):.2f}")
        write_json(strategy_dir / "metrics_per_fold.json", fold_metrics)
        write_json(strategy_dir / "metrics_aggregated.json", agg)
        write_predictions_csv(strategy_dir / "predictions.csv", all_preds)
        write_predictions_csv(strategy_dir / "validation_predictions.csv", val_preds)

        best = find_best_threshold_from_arrays(
            np.array([r["y_true"] for r in val_preds], dtype=int),
            np.array([r["y_prob"] for r in val_preds], dtype=float),
        )
        write_json(strategy_dir / "best_threshold_metrics.json", best)

        print(
            f"    ── {strategy.upper()} AGGREGATE: "
            f"acc={agg['accuracy_mean']:.3f}±{agg['accuracy_std']:.3f} "
            f"f1={agg['f1_pos_mean']:.3f}±{agg['f1_pos_std']:.3f} "
            f"auc={agg['roc_auc_mean']:.3f}±{agg['roc_auc_std']:.3f}"
        )
        print(f"    ── {strategy.upper()} VALIDATION BEST THRESHOLD: {best['best_threshold']:.2f}")


def run_ablation_experiment(data: dict, run_dir: Path):
    """Ablation study on text-branch feature subsets (SBERT, Handcrafted, SLM)."""
    run_dir.mkdir(parents=True, exist_ok=True)

    handcrafted = np.concatenate([data["keywords"], data["meta"]], axis=1)

    slm_score = load_slm_score(data["app_ids"])
    has_slm = slm_score is not None

    ablation_configs: dict[str, np.ndarray] = {
        "sbert_only":        data["sbert"],
        "handcrafted_only":  handcrafted,
        "sbert_handcrafted": data["text_feats"],
    }
    if has_slm:
        ablation_configs["slm_only"]          = slm_score
        ablation_configs["sbert_slm"]         = np.concatenate([data["sbert"], slm_score], axis=1)
        ablation_configs["handcrafted_slm"]   = np.concatenate([handcrafted, slm_score], axis=1)
        ablation_configs["sbert_handcrafted_slm"] = np.concatenate([data["text_feats"], slm_score], axis=1)
    else:
        print("  [Ablation] SLM features not found — skipping SLM configs")

    summary = {}
    for name, X in ablation_configs.items():
        marker = " ← main" if name == "sbert_handcrafted" else ""
        print(f"\n  [Ablation] {name}{marker}  ({X.shape[1]} dims)")
        _, fold_metrics = run_single_experiment(name, X, data, run_dir / name)
        agg = aggregate_metrics(fold_metrics)
        summary[name] = {
            "n_dims": int(X.shape[1]),
            "roc_auc_mean": round(agg["roc_auc_mean"], 4),
            "roc_auc_std": round(agg["roc_auc_std"], 4),
            "f1_pos_mean": round(agg["f1_pos_mean"], 4),
            "f1_pos_std": round(agg["f1_pos_std"], 4),
        }

    write_json(run_dir / "ablation_summary.json", summary)

    print("\nABLATION STUDY — Text Branch Components")
    print(f"  {'Config':<28} {'Dims':>5}  {'ROC-AUC':>14}  {'F1':>14}")
    print("  " + "-" * 68)
    for name, r in summary.items():
        marker = " ← main" if name == "sbert_handcrafted" else "       "
        print(
            f"  {name:<28} {r['n_dims']:>5}{marker} "
            f"{r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}  "
            f"{r['f1_pos_mean']:.4f}±{r['f1_pos_std']:.4f}"
        )


def main():
    set_seed(CFG.seed)
    base_dir = Path(CFG.runs_dir) / CFG.run_name

    data = load_features()
    n = len(data["app_ids"])
    print(
        f"  {n} apps, text_feats={data['text_feats'].shape[1]}d, "
        f"image_feats={data['image_feats'].shape[1]}d, "
        f"all_feats={data['all_feats'].shape[1]}d"
    )

    print("\n[A] Text-Only Classifier")
    run_single_experiment("TextOnly", data["text_feats"], data, base_dir / "text_only")

    print("\n[B] Image-Only Classifier")
    run_single_experiment("ImageOnly", data["image_feats"], data, base_dir / "image_only")

    print("\n[C] Fusion Classifiers")
    run_fusion_experiment(data, base_dir / "fusion")

    ablation_dir = base_dir / "ablation"
    if (ablation_dir / "ablation_summary.json").exists():
        print(f"\n[skip] Ablation already done at {ablation_dir}")
    else:
        print("\n[D] Ablation Study — Text Branch")
        run_ablation_experiment(data, ablation_dir)

    print("\nValidation-based threshold summary ...")
    search_paths = ["text_only", "image_only", "fusion/early_fusion"]
    for strategy in CFG.fusion_strategy:
        search_paths.append(f"fusion/late_fusion_{strategy}")

    for sub in search_paths:
        json_path = base_dir / sub / "best_threshold_metrics.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                best = json.load(f)
            print(
                f"  {sub}: best_t={best['best_threshold']:.2f} "
                f"acc={best['accuracy']:.3f} f1={best['f1_pos']:.3f}"
            )

    print("\nSUMMARY: Late Fusion Strategy Comparison")
    for strategy in CFG.fusion_strategy:
        strategy_dir = base_dir / "fusion" / f"late_fusion_{strategy}"
        agg_path = strategy_dir / "metrics_aggregated.json"
        if agg_path.exists():
            with open(agg_path) as f:
                agg = json.load(f)
            print(
                f"  {strategy.upper():15s}: "
                f"F1={agg['f1_pos_mean']:.4f}±{agg['f1_pos_std']:.4f}  "
                f"ROC-AUC={agg['roc_auc_mean']:.4f}±{agg['roc_auc_std']:.4f}"
            )

    print("\nDone! Results & Models saved to:", base_dir)


if __name__ == "__main__":
    main()
