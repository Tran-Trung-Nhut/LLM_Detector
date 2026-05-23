import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG

ANDROZOO_DOWNLOAD_URL = "https://androzoo.uni.lu/api/download"
ANDROZOO_CSV          = "data/androzoo_latest.csv"
CODE_VAL_APPS_CSV     = "data/code_validation_apps.csv"
APK_DIR               = Path("data/apks")
DECOMPILE_DIR         = Path("data/decompiled")
CODE_VAL_CHECKPOINT   = Path("data/code_validation_checkpoint.json")
IND_CHECKPOINT        = Path("data/ai_disc_independent_checkpoint.json")

AI_DISC_CLI = _PROJECT_ROOT / "AIApp-custom" / "identification" / "ai_discriminator_cli.py"
AI_DISC_BIN = f"python {AI_DISC_CLI}"
APKTOOL_JAR = _PROJECT_ROOT / "AIApp-custom" / "identification" / "apktool_2.5.0.jar"


def load_pkg2sha(target_pkgs: set) -> dict:
    pkg2sha, pkg2date = {}, {}
    with open(ANDROZOO_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            pkg = row.get("pkg_name", "").strip()
            if pkg not in target_pkgs:
                continue
            date = row.get("dex_date", "")
            if pkg not in pkg2date or date > pkg2date[pkg]:
                pkg2date[pkg] = date
                pkg2sha[pkg]  = row["sha256"].strip()
    return pkg2sha


def download_apk(api_key: str, sha256: str, out_path: Path) -> bool:
    for attempt in range(4):
        try:
            r = requests.get(ANDROZOO_DOWNLOAD_URL,
                             params={"apikey": api_key, "sha256": sha256},
                             stream=True, timeout=180)
            if r.status_code == 200:
                out_path.write_bytes(r.content)
                return True
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return False


def decompile(apk_path: Path, out_dir: Path) -> bool:
    if out_dir.exists() and any(out_dir.iterdir()):
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["java", "-jar", str(APKTOOL_JAR), "d", "-f", str(apk_path), "-o", str(out_dir)],
        capture_output=True, timeout=300,
    )
    return result.returncode == 0


def run_ai_discriminator(decompiled_dir: Path) -> int:
    result = subprocess.run(
        AI_DISC_BIN.split() + ["--dir", str(decompiled_dir)],
        capture_output=True, text=True, timeout=600,
    )
    for line in reversed(result.stdout.strip().splitlines()):
        if line.strip() in ("0", "1"):
            return int(line.strip())
    return -1


def _run_ai_disc_pipeline(api_key: str, apps: list, out_csv: str, checkpoint_path: Path):
    pkg2sha = load_pkg2sha({a["pkg"] for a in apps})
    APK_DIR.mkdir(parents=True, exist_ok=True)
    DECOMPILE_DIR.mkdir(parents=True, exist_ok=True)

    done = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    results = []
    n = len(apps)

    for i, app in enumerate(apps, 1):
        pkg   = app["pkg"]
        label = app["listing_label"]

        if pkg in done:
            results.append(done[pkg])
            print(f"[{i}/{n}] {pkg}  skip")
            continue

        row = {"pkg_name": pkg, "listing_label": label, "ai_discriminator_label": -1}

        sha256 = pkg2sha.get(pkg)
        if not sha256:
            row["note"] = "not_in_androzoo"
            done[pkg] = row; checkpoint_path.write_text(json.dumps(done, indent=2)); results.append(row)
            print(f"[{i}/{n}] {pkg}  not_in_androzoo")
            continue

        apk_path = APK_DIR / f"{pkg}.apk"
        if not apk_path.exists() and not download_apk(api_key, sha256, apk_path):
            row["note"] = "download_failed"
            done[pkg] = row; checkpoint_path.write_text(json.dumps(done, indent=2)); results.append(row)
            print(f"[{i}/{n}] {pkg}  download_failed")
            continue

        dec_dir = DECOMPILE_DIR / pkg
        if not decompile(apk_path, dec_dir):
            row["note"] = "decompile_failed"
            done[pkg] = row; checkpoint_path.write_text(json.dumps(done, indent=2)); results.append(row)
            print(f"[{i}/{n}] {pkg}  decompile_failed")
            continue

        row["ai_discriminator_label"] = run_ai_discriminator(dec_dir)
        done[pkg] = row; checkpoint_path.write_text(json.dumps(done, indent=2)); results.append(row)
        print(f"[{i}/{n}] {pkg}  -> {row['ai_discriminator_label']}")

    valid = [r for r in results if r.get("ai_discriminator_label", -1) != -1]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pkg_name", "listing_label", "ai_discriminator_label"])
        writer.writeheader()
        writer.writerows([{k: r.get(k, -1) for k in writer.fieldnames} for r in valid])


def _assert_androzoo():
    if not Path(ANDROZOO_CSV).exists():
        sys.exit(f"[error] {ANDROZOO_CSV} not found. Download from androzoo.uni.lu/static/lists/latest.csv.gz")


def _compute_row(df: pd.DataFrame) -> dict:
    ref  = df["listing_label"].values.astype(int)
    pred = df["ai_discriminator_label"].values.astype(int)
    tn, fp, fn, tp = confusion_matrix(ref, pred, labels=[0, 1]).ravel()
    return dict(
        n=len(df), kappa=cohen_kappa_score(ref, pred),
        pct=float((ref == pred).mean()) * 100,
        f1=f1_score(ref, pred, pos_label=1, zero_division=0),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def _print_row(label: str, r: dict):
    print(f"\n{label}")
    print(f"  {'':12} {'TP':>4} {'FN':>4} {'TN':>4} {'FP':>4} {'F1':>6} {'κ':>6} {'Agree':>7}")
    print(f"  {'Computed':<12} {r['tp']:>4} {r['fn']:>4} {r['tn']:>4} {r['fp']:>4} {r['f1']:>6.3f} {r['kappa']:>6.3f} {r['pct']:>6.1f}%")


def _load_ef_predictions(pkg_names: list) -> dict:
    ef_path = Path(CFG.runs_dir) / CFG.run_name / "independent_test" / "predictions_early_fusion.csv"
    if not ef_path.exists():
        return {}
    pred_map = {}
    with open(ef_path) as f:
        for row in csv.DictReader(f):
            pred_map[row["app_id"]] = {"y_true": int(row["y_true"]), "y_prob": float(row["y_prob"])}
    return {pkg: pred_map[pkg] for pkg in pkg_names if pkg in pred_map}


def _save_metrics(label: str, r: dict, extra: dict | None = None):
    out_path = Path(CFG.runs_dir) / "cohen_kappa" / "validation.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode) as f:
        slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        f.write(f"[{label}]\n")
        f.write(f"tp={r['tp']} fn={r['fn']} tn={r['tn']} fp={r['fp']}\n")
        f.write(f"f1={r['f1']:.3f}\n")
        f.write(f"kappa={r['kappa']:.3f}\n")
        f.write(f"agreement={r['pct']:.1f}%\n")
        if extra:
            for k, v in extra.items():
                f.write(f"{k}={v}\n")
        f.write("\n")


def phase1(api_key: str):
    _assert_androzoo()
    apps = []
    with open(CODE_VAL_APPS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            apps.append({"pkg": row["pkg_name"], "listing_label": int(row["listing_label"])})
    _run_ai_disc_pipeline(api_key, apps, CFG.code_validation_csv, CODE_VAL_CHECKPOINT)

    path = Path(CFG.code_validation_csv)
    if path.exists():
        r = _compute_row(pd.read_csv(path))
        _print_row("Code-validation (N=80)", r)
        _save_metrics("Code-validation (N=80)", r)


def phase2(api_key: str):
    _assert_androzoo()
    apps = []
    with open(CFG.inference_manual_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            apps.append({"pkg": row["pkg_name"], "listing_label": int(row["label"])})
    _run_ai_disc_pipeline(api_key, apps, CFG.ai_disc_independent_csv, IND_CHECKPOINT)

    path = Path(CFG.ai_disc_independent_csv)
    if not path.exists():
        return
    df  = pd.read_csv(path)
    r   = _compute_row(df)
    _print_row("Independent test (N=110)", r)

    ef_preds = _load_ef_predictions(df["pkg_name"].tolist())
    extra = {}
    if ef_preds:
        common    = [p for p in df["pkg_name"] if p in ef_preds]
        y_true_ef = np.array([ef_preds[p]["y_true"] for p in common])
        y_prob_ef = np.array([ef_preds[p]["y_prob"] for p in common])
        ef_f1 = f1_score(y_true_ef, (y_prob_ef >= CFG.classification_threshold).astype(int),
                         pos_label=1, zero_division=0)
        print(f"\n  On {len(common)} identical apps:")
        print(f"  {'':20} {'AI Disc':>8} {'EF':>8}")
        print(f"  {'Computed F1':<20} {r['f1']:>8.3f} {ef_f1:>8.3f}")
        extra = {"ef_f1": f"{ef_f1:.3f}", "n_identical": len(common)}
    _save_metrics("Independent test (N=110)", r, extra)


def _require_api_key() -> str:
    key = os.environ.get("ANDROZOO_API_KEY", "")
    if not key:
        sys.exit("[error] Set ANDROZOO_API_KEY environment variable first.")
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    args = parser.parse_args()

    if 1 in args.phases:
        phase1(_require_api_key())
    if 2 in args.phases:
        phase2(_require_api_key())


if __name__ == "__main__":
    main()
