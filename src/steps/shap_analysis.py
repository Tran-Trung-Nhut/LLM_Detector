"""shap_analysis.py — TreeSHAP feature importance for Text-Only and Early Fusion models, aggregated across 5 folds."""
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

try:
    import shap
except ImportError:
    print("[error] shap not installed. Run:  pip install shap")
    sys.exit(1)

import lightgbm as lgb
import joblib


def make_feature_names(mode: str) -> list[str]:
    kw_names = [
        "kw_total_count", "kw_log_count",
        "kw_model_name_hit", "kw_model_name_count",
        "kw_core_llm_hit",   "kw_core_llm_count",
        "kw_generation_hit", "kw_generation_count",
        "kw_interaction_hit","kw_interaction_count",
        "kw_content_hit",    "kw_content_count",
        "kw_max_len",
    ]
    meta_names = (
        ["meta_desc_len", "meta_short_len", "meta_title_len",
         "meta_has_changes", "meta_n_images"]
        + [f"meta_cat_{c.lower().replace(' & ','_').replace(' ','_')}"
           for c in CFG.top_categories]
        + ["meta_cat_other"]
    )
    sbert_names = [f"sbert_{i}" for i in range(CFG.text_embed_dim)]

    text_names = sbert_names + kw_names + meta_names  # 1024+13+21 = 1058

    if mode == "text_only":
        return text_names

    # Early Fusion appends image features
    clip_mean_names = [f"clip_mean_{i}" for i in range(CFG.clip_embed_dim)]
    clip_max_names  = [f"clip_max_{i}"  for i in range(CFG.clip_embed_dim)]
    zs_names        = ["zeroshot_chatbot_ui"]
    ocr_names       = [f"ocr_{i}" for i in range(15)]
    return text_names + clip_mean_names + clip_max_names + zs_names + ocr_names


def top_features(shap_values: np.ndarray, feature_names: list[str],
                 selected_indices: np.ndarray, top_n: int = 30) -> list[dict]:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    results = []
    for rank, local_idx in enumerate(np.argsort(mean_abs)[::-1][:top_n]):
        orig_idx = int(selected_indices[local_idx])
        results.append({
            "rank":          rank + 1,
            "feature_name":  feature_names[orig_idx] if orig_idx < len(feature_names) else f"feat_{orig_idx}",
            "original_idx":  orig_idx,
            "mean_abs_shap": round(float(mean_abs[local_idx]), 6),
        })
    return results


def run_shap_for_model(model_dir: Path, mode: str,
                       X_all: np.ndarray, labels: np.ndarray,
                       feature_names: list[str]) -> dict:
    from steps.train_evaluate import load_split

    id2idx_path = Path(CFG.features_dir) / "text" / "features.npz"
    td = np.load(id2idx_path, allow_pickle=True)
    app_ids = list(td["app_ids"])
    id2idx  = {a: i for i, a in enumerate(app_ids)}

    all_shap_values  = []
    all_selected_idx = []

    for fold in range(CFG.n_folds):
        sel_path = model_dir / f"selector_fold_{fold}.joblib"
        mdl_path = model_dir / f"lgbm_fold_{fold}.txt"
        if not sel_path.exists() or not mdl_path.exists():
            print(f"  [skip] fold {fold}: model files not found")
            continue

        selector = joblib.load(sel_path)
        model    = lgb.Booster(model_file=str(mdl_path))
        selected = selector.get_support(indices=True)

        split = load_split(fold)
        test_idx = [id2idx[a] for a in split["test_ids"] if a in id2idx]
        X_explain = selector.transform(X_all[test_idx])

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # binary classification: take positive class

        all_shap_values.append(shap_values)
        all_selected_idx.append(selected)
        print(f"  Fold {fold}: SHAP computed on {len(test_idx)} samples")

    if not all_shap_values:
        return {}

    combined_shap = np.vstack(all_shap_values)
    selected_idx  = all_selected_idx[0]  # fold 0 indices are representative when k is fixed

    top = top_features(combined_shap, feature_names, selected_idx, top_n=30)

    mean_abs_all = np.mean(np.abs(combined_shap), axis=0)
    group_imp = {"sbert": 0.0, "keyword": 0.0, "meta": 0.0,
                 "clip_mean": 0.0, "clip_max": 0.0, "zeroshot": 0.0, "ocr": 0.0}
    for local_i, orig_i in enumerate(selected_idx):
        v = float(mean_abs_all[local_i])
        if orig_i < 1024:
            group_imp["sbert"] += v
        elif orig_i < 1037:
            group_imp["keyword"] += v
        elif orig_i < 1058:
            group_imp["meta"] += v
        elif orig_i < 1826:
            group_imp["clip_mean"] += v
        elif orig_i < 2594:
            group_imp["clip_max"] += v
        elif orig_i < 2595:
            group_imp["zeroshot"] += v
        else:
            group_imp["ocr"] += v

    return {
        "top_30_features": top,
        "group_importance": {k: round(v, 6) for k, v in group_imp.items()},
        "n_folds_aggregated": len(all_shap_values),
        "n_samples_total": int(combined_shap.shape[0]),
    }


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "shap_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    td  = np.load(Path(CFG.features_dir) / "text"  / "features.npz", allow_pickle=True)
    imd = np.load(Path(CFG.features_dir) / "image" / "features.npz", allow_pickle=True)
    labels = td["labels"].astype(int)

    text_feats  = np.concatenate([td["sbert"], td["keywords"], td["meta"]], axis=1)
    image_feats = np.concatenate([imd["clip_mean"], imd["clip_max"], imd["zeroshot"], imd["ocr"]], axis=1)
    all_feats   = np.concatenate([text_feats, image_feats], axis=1)

    results = {}

    text_only_dir = base_dir / "text_only" / "saved_models"
    if text_only_dir.exists():
        print("\n[1/2] Text-Only SHAP ...")
        feat_names_text = make_feature_names("text_only")
        r = run_shap_for_model(text_only_dir, "text_only", text_feats, labels, feat_names_text)
        results["text_only"] = r
        if r:
            write_json(out_dir / "shap_text_only.json", r)
            print(f"  Top 5 features: {[x['feature_name'] for x in r['top_30_features'][:5]]}")
    else:
        print(f"\n[skip] Text-Only: saved_models not found at {text_only_dir}")

    ef_dir = base_dir / "fusion" / "early_fusion" / "saved_models"
    if ef_dir.exists():
        print("\n[2/2] Early Fusion SHAP ...")
        feat_names_ef = make_feature_names("early_fusion")
        r = run_shap_for_model(ef_dir, "early_fusion", all_feats, labels, feat_names_ef)
        results["early_fusion"] = r
        if r:
            write_json(out_dir / "shap_early_fusion.json", r)
            print(f"  Top 5 features: {[x['feature_name'] for x in r['top_30_features'][:5]]}")
    else:
        print(f"\n[skip] Early Fusion: saved_models not found at {ef_dir}")

    write_json(out_dir / "shap_combined.json", results)

    print("\nGROUP-LEVEL IMPORTANCE (mean |SHAP| summed per group)")
    groups = ["sbert", "keyword", "meta", "clip_mean", "clip_max", "zeroshot", "ocr"]
    print(f"  {'Group':<12} {'Text-Only':>11} {'Early Fusion':>13}")
    print("  " + "-" * 40)
    for g in groups:
        to_v = results.get("text_only",    {}).get("group_importance", {}).get(g, 0)
        ef_v = results.get("early_fusion", {}).get("group_importance", {}).get(g, 0)
        if to_v == 0 and ef_v == 0:
            continue
        print(f"  {g:<12} {to_v:>11.6f} {ef_v:>13.6f}")

    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
