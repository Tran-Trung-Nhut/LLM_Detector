import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    f1_score,
)

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": float(threshold),
        "precision_pos": float(precision),
        "recall_pos": float(recall),
        "f1_pos": float(f1),
        "macro_f1": float(macro_f1),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
    }