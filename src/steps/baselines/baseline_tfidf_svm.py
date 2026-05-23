"""
baseline_tfidf_svm.py — TF-IDF + linear SVM description-only baseline.
Paper Section 3.5, Table 12.

TF-IDF (1-2 grams, max 20k features) + L2-regularized LinearSVC trained on the
298-app training set, evaluated on the 110-app independent test set.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from config import CFG
from utils.io import read_jsonl, write_json, load_label_map
from utils.metrics import compute_binary_metrics


def _texts(rows: list) -> list:
    return [f"{r.get('title', '')} {r.get('description', '') or ''}".strip() for r in rows]


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    out_dir = Path(CFG.runs_dir) / CFG.run_name / "independent_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(CFG.dataset_path)
    X_train    = _texts(train_rows)
    y_train    = np.array([r["label"] for r in train_rows], dtype=int)

    test_rows = read_jsonl(CFG.raw_inference_dataset_path)
    test_ids  = [r["app_id"] for r in test_rows]
    X_test    = _texts(test_rows)
    label_map = load_label_map(CFG.inference_manual_csv)
    y_test    = np.array([label_map.get(aid, -1) for aid in test_ids], dtype=int)

    valid    = y_test >= 0
    X_test   = [x for x, v in zip(X_test, valid) if v]
    y_test   = y_test[valid]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=CFG.tfidf_svm_ngram_range,
                                   max_features=CFG.tfidf_svm_max_features)),
        ("svm",   LinearSVC(C=CFG.tfidf_svm_C, max_iter=5000)),
    ])
    model.fit(X_train, y_train)

    t0    = time.perf_counter()
    y_prob = expit(model.decision_function(X_test))
    latency_s_per_app = round((time.perf_counter() - t0) / len(y_test), 6)

    m = compute_binary_metrics(y_test, y_prob, threshold=0.5)
    write_json(out_dir / "baseline_tfidf_svm.json", {
        "metrics": m,
        "latency_s_per_app": latency_s_per_app,
        "protocol": f"train on {len(y_train)} apps, test on {len(y_test)} apps",
    })


if __name__ == "__main__":
    main()
