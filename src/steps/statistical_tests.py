"""statistical_tests.py — Paired significance tests (bootstrap, DeLong, McNemar, Holm/BH) on the independent test set."""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm
from sklearn.metrics import roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json


def load_predictions(csv_path: Path):
    app_ids, y_true, y_prob = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            app_ids.append(row["app_id"])
            y_true.append(int(row["y_true"]))
            y_prob.append(float(row["y_prob"]))
    order = sorted(range(len(app_ids)), key=lambda i: app_ids[i])
    app_ids = [app_ids[i] for i in order]
    y_true = np.array([y_true[i] for i in order], dtype=int)
    y_prob = np.array([y_prob[i] for i in order], dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    return app_ids, y_true, y_prob, y_pred


def mcnemar_test(y_true: np.ndarray, pred_base: np.ndarray, pred_cmp: np.ndarray) -> dict:
    n01 = int(np.sum((pred_base != y_true) & (pred_cmp == y_true)))
    n10 = int(np.sum((pred_base == y_true) & (pred_cmp != y_true)))
    denom = n01 + n10
    if denom == 0:
        return {"n01": 0, "n10": 0, "chi2_stat": 0.0, "p_value": 1.0, "note": "no disagreements"}
    chi2_stat = float((abs(n01 - n10) - 1) ** 2 / denom)
    p_value = float(1.0 - chi2.cdf(chi2_stat, df=1))
    return {"n01": n01, "n10": n10, "chi2_stat": round(chi2_stat, 4), "p_value": round(p_value, 4)}


def bootstrap_auc(
    y_true: np.ndarray,
    prob_base: np.ndarray,
    prob_cmp: np.ndarray,
    n_bootstrap: int = None,
    seed: int = None,
) -> dict:
    if n_bootstrap is None:
        n_bootstrap = CFG.n_bootstrap
    if seed is None:
        seed = CFG.seed
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(
            roc_auc_score(y_true[idx], prob_cmp[idx])
            - roc_auc_score(y_true[idx], prob_base[idx])
        )
    diffs = np.array(diffs)
    auc_base = float(roc_auc_score(y_true, prob_base))
    auc_cmp = float(roc_auc_score(y_true, prob_cmp))
    observed_diff = auc_cmp - auc_base
    p_one_sided = float(np.mean(diffs <= 0))
    return {
        "auc_base": round(auc_base, 4),
        "auc_cmp": round(auc_cmp, 4),
        "delta_auc": round(observed_diff, 4),
        "ci95_lower": round(float(np.percentile(diffs, 2.5)), 4),
        "ci95_upper": round(float(np.percentile(diffs, 97.5)), 4),
        "p_value_bootstrap": round(p_one_sided, 4),
    }


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def bootstrap_f1_accuracy(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_cmp: np.ndarray,
    n_bootstrap: int = None,
    seed: int = None,
) -> dict:
    """Paired bootstrap test for F1 and accuracy difference (cmp − base)."""
    if n_bootstrap is None:
        n_bootstrap = CFG.n_bootstrap
    if seed is None:
        seed = CFG.seed
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1_diffs, acc_diffs = [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt, pb, pc = y_true[idx], pred_base[idx], pred_cmp[idx]
        if len(np.unique(yt)) < 2:
            continue
        f1_diffs.append(_f1_score(yt, pc) - _f1_score(yt, pb))
        acc_diffs.append(float(np.mean(yt == pc)) - float(np.mean(yt == pb)))
    f1_diffs = np.array(f1_diffs)
    acc_diffs = np.array(acc_diffs)

    obs_f1_base = _f1_score(y_true, pred_base)
    obs_f1_cmp  = _f1_score(y_true, pred_cmp)
    obs_acc_base = float(np.mean(y_true == pred_base))
    obs_acc_cmp  = float(np.mean(y_true == pred_cmp))

    return {
        "f1_base":           round(obs_f1_base, 4),
        "f1_cmp":            round(obs_f1_cmp, 4),
        "delta_f1":          round(obs_f1_cmp - obs_f1_base, 4),
        "f1_ci95_lower":     round(float(np.percentile(f1_diffs, 2.5)), 4),
        "f1_ci95_upper":     round(float(np.percentile(f1_diffs, 97.5)), 4),
        "p_value_f1":        round(float(np.mean(f1_diffs <= 0)), 4),
        "acc_base":          round(obs_acc_base, 4),
        "acc_cmp":           round(obs_acc_cmp, 4),
        "delta_acc":         round(obs_acc_cmp - obs_acc_base, 4),
        "acc_ci95_lower":    round(float(np.percentile(acc_diffs, 2.5)), 4),
        "acc_ci95_upper":    round(float(np.percentile(acc_diffs, 97.5)), 4),
        "p_value_acc":       round(float(np.mean(acc_diffs <= 0)), 4),
    }


def delong_test(y_true: np.ndarray, prob_base: np.ndarray, prob_cmp: np.ndarray) -> dict:
    def auc_and_components(y, pred):
        pos = pred[y == 1]
        neg = pred[y == 0]
        n_pos, n_neg = len(pos), len(neg)
        if n_pos == 0 or n_neg == 0:
            return 0.5, np.zeros(n_pos), np.zeros(n_neg)
        v_pos = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
        v_neg = np.array([np.mean(n < pos) + 0.5 * np.mean(n == pos) for n in neg])
        return float(np.mean(v_pos)), v_pos, v_neg

    auc1, vp1, vn1 = auc_and_components(y_true, prob_base)
    auc2, vp2, vn2 = auc_and_components(y_true, prob_cmp)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    var1 = np.var(vp1, ddof=1) / n_pos + np.var(vn1, ddof=1) / n_neg if n_pos > 1 and n_neg > 1 else 0.0
    var2 = np.var(vp2, ddof=1) / n_pos + np.var(vn2, ddof=1) / n_neg if n_pos > 1 and n_neg > 1 else 0.0
    cov = (
        (np.cov(vp1, vp2, ddof=1)[0, 1] / n_pos if n_pos > 1 else 0.0) +
        (np.cov(vn1, vn2, ddof=1)[0, 1] / n_neg if n_neg > 1 else 0.0)
    )

    delta = auc2 - auc1
    se = float(np.sqrt(max(var1 + var2 - 2 * cov, 1e-12)))
    z = delta / se
    p_two_sided = float(2 * (1 - norm.cdf(abs(z))))
    return {
        "auc_base": round(float(auc1), 4),
        "auc_cmp": round(float(auc2), 4),
        "delta_auc": round(float(delta), 4),
        "z_stat": round(z, 4),
        "p_delong": round(p_two_sided, 4),
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[float]:
    from statsmodels.stats.multitest import multipletests
    _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return [round(float(p), 4) for p in p_corrected]


def benjamini_hochberg(p_values: list[float], q: float = 0.10) -> list[float]:
    from statsmodels.stats.multitest import multipletests
    _, p_corrected, _, _ = multipletests(p_values, alpha=q, method="fdr_bh")
    return [round(float(p), 4) for p in p_corrected]


def cliffs_delta(fold_auc_base: list[float], fold_auc_cmp: list[float]) -> float:
    a = np.array(fold_auc_base)
    b = np.array(fold_auc_cmp)
    greater = sum(1 for x in b for y in a if x > y)
    less = sum(1 for x in b for y in a if x < y)
    return round(float((greater - less) / (len(a) * len(b))), 4)


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir = base_dir / "statistical_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dir = base_dir / "independent_test"
    baseline_csv = test_dir / "predictions_text_only.csv"
    if not baseline_csv.exists():
        print(f"[error] Independent test predictions not found at {baseline_csv}")
        print("  Run independent_test_eval.py first.")
        return

    base_ids, y_true_base, y_prob_base, y_pred_base = load_predictions(baseline_csv)

    comparisons = {
        "early_fusion":      test_dir / "predictions_early_fusion.csv",
        "late_score_max":    test_dir / "predictions_score_max.csv",
        "late_soft_voting":  test_dir / "predictions_soft_voting.csv",
        "late_stacking":     test_dir / "predictions_stacking.csv",
        "image_only":        test_dir / "predictions_image_only.csv",
    }

    def get_fold_aucs(json_path: Path) -> list[float]:
        if not json_path.exists():
            return []
        with open(json_path) as f:
            folds = json.load(f)
        return [m["roc_auc"] for m in folds]

    base_fold_aucs = get_fold_aucs(base_dir / "text_only" / "metrics_per_fold.json")

    raw_results = {}
    comparison_names = []
    delong_pvals = []
    f1_pvals = []
    mcnemar_pvals = []

    for name, csv_path in comparisons.items():
        if not csv_path.exists():
            print(f"  [skip] {name}: {csv_path.name} not found")
            continue
        cmp_ids, y_true_cmp, y_prob_cmp, y_pred_cmp = load_predictions(csv_path)
        assert base_ids == cmp_ids, f"ID mismatch for {name}"

        mc   = mcnemar_test(y_true_base, y_pred_base, y_pred_cmp)
        boot = bootstrap_auc(y_true_base, y_prob_base, y_prob_cmp)
        dl   = delong_test(y_true_base, y_prob_base, y_prob_cmp)
        f1b  = bootstrap_f1_accuracy(y_true_base, y_pred_base, y_pred_cmp)

        if name == "early_fusion":
            fold_auc_path = base_dir / "fusion" / "early_fusion" / "metrics_per_fold.json"
        elif name == "image_only":
            fold_auc_path = base_dir / "image_only" / "metrics_per_fold.json"
        else:
            strat_key = name.replace("late_", "")
            fold_auc_path = base_dir / "fusion" / f"late_fusion_{strat_key}" / "metrics_per_fold.json"
        cmp_fold_aucs = get_fold_aucs(fold_auc_path)
        cliff = cliffs_delta(base_fold_aucs, cmp_fold_aucs) if (base_fold_aucs and cmp_fold_aucs) else None

        raw_results[name] = {
            "mcnemar": mc,
            "bootstrap_auc": boot,
            "bootstrap_f1_acc": f1b,
            "delong": dl,
            "cliffs_delta": cliff,
        }
        comparison_names.append(name)
        delong_pvals.append(dl["p_delong"])
        f1_pvals.append(f1b["p_value_f1"])
        mcnemar_pvals.append(mc.get("p_value", 1.0))

    if not comparison_names:
        print("[error] No comparison files found.")
        return

    p_holm_delong = holm_bonferroni(delong_pvals)
    p_bh_delong   = benjamini_hochberg(delong_pvals)
    p_holm_f1     = holm_bonferroni(f1_pvals)

    final_results = {}
    for i, name in enumerate(comparison_names):
        final_results[name] = {
            **raw_results[name],
            "p_holm_delong": p_holm_delong[i],
            "p_bh_delong":   p_bh_delong[i],
            "p_holm_f1":     p_holm_f1[i],
        }

    write_json(out_dir / "results.json", final_results)

    summary_rows = []
    for name, r in final_results.items():
        boot = r["bootstrap_auc"]
        f1b  = r["bootstrap_f1_acc"]
        dl   = r["delong"]
        mc   = r["mcnemar"]
        summary_rows.append({
            "comparison":        name,
            "f1_base":           f1b["f1_base"],
            "f1_cmp":            f1b["f1_cmp"],
            "delta_f1":          f1b["delta_f1"],
            "f1_ci95":           f"[{f1b['f1_ci95_lower']},{f1b['f1_ci95_upper']}]",
            "p_f1_bootstrap":    f1b["p_value_f1"],
            "p_holm_f1":         r["p_holm_f1"],
            "delta_auc":         boot["delta_auc"],
            "auc_ci95":          f"[{boot['ci95_lower']},{boot['ci95_upper']}]",
            "p_auc_bootstrap":   boot["p_value_bootstrap"],
            "p_delong":          dl["p_delong"],
            "p_holm_delong":     r["p_holm_delong"],
            "p_bh_delong":       r["p_bh_delong"],
            "cliffs_delta":      r["cliffs_delta"],
            "mcnemar_p":         mc.get("p_value", ""),
        })

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'Comparison':<22} {'ΔF1':>7} {'p_F1':>8} {'ΔAUC':>7} {'p_DeLong':>9} {'Sig?':>5}")
    print("-" * 65)
    for r in summary_rows:
        sig = "*" if (r["p_f1_bootstrap"] < 0.05 or r["p_delong"] < 0.05) else ""
        print(f"  {r['comparison']:<20} {r['delta_f1']:>+7.4f} {r['p_f1_bootstrap']:>8.4f} "
              f"{r['delta_auc']:>+7.4f} {r['p_delong']:>9.4f} {sig:>5}")
    print(f"\nAll tests on independent test set (N={len(y_true_base)}).")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
