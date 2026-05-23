"""latency_benchmark.py — Table 16: Per-app inference latency for all pipeline components."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import read_jsonl, write_json


def time_bge_encoding(records: list, n_runs: int = None) -> float:
    if n_runs is None:
        n_runs = CFG.latency_n_runs
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model)
    model     = AutoModel.from_pretrained(CFG.text_model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    times = []
    for r in records[:min(len(records), 20)]:
        texts = [f"{r.get('title', '')} {(r.get('description') or '')[:512]}"]
        for _ in range(n_runs):
            t0     = time.perf_counter()
            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=CFG.text_max_length).to(device)
            with torch.no_grad():
                out = model(**inputs)
            _ = out.last_hidden_state[:, 0].cpu().numpy()
            times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def time_clip_encoding(records: list, n_screenshots: int = None, n_runs: int = None) -> float:
    if n_screenshots is None:
        n_screenshots = CFG.latency_n_screenshots
    if n_runs is None:
        n_runs = CFG.latency_n_runs
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image

    processor = CLIPProcessor.from_pretrained(CFG.clip_model)
    model     = CLIPModel.from_pretrained(CFG.clip_model, dtype=torch.float16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    times = []
    for r in records[:min(len(records), 20)]:
        img_paths = [p for p in r.get("image_paths", []) if Path(p).exists()][:n_screenshots]
        if not img_paths:
            continue
        images = [Image.open(p).convert("RGB") for p in img_paths]
        for _ in range(n_runs):
            t0     = time.perf_counter()
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                model.vision_model(pixel_values=inputs["pixel_values"])
            times.append(time.perf_counter() - t0)
    return float(np.mean(times)) if times else 0.0


def time_ocr(records: list, n_screenshots: int = None) -> float:
    if n_screenshots is None:
        n_screenshots = CFG.latency_n_screenshots
    import pytesseract
    from PIL import Image

    times = []
    for r in records[:min(len(records), 20)]:
        img_paths = [p for p in r.get("image_paths", []) if Path(p).exists()][:n_screenshots]
        if not img_paths:
            continue
        t0 = time.perf_counter()
        for p in img_paths:
            try:
                pytesseract.image_to_string(Image.open(p), lang=CFG.ocr_lang)
            except Exception:
                pass
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)) if times else 0.0


def time_lgbm_inference(n_apps: int = 110) -> dict:
    import lightgbm as lgb
    import joblib

    base_dir   = Path(CFG.runs_dir) / CFG.run_name
    models_dir = base_dir / "fusion" / "base_models_saved"

    sel_t = joblib.load(models_dir / "text_selector_fold_0.joblib")
    mdl_t = lgb.Booster(model_file=str(models_dir / "text_lgbm_fold_0.txt"))
    sel_i = joblib.load(models_dir / "img_selector_fold_0.joblib")
    mdl_i = lgb.Booster(model_file=str(models_dir / "img_lgbm_fold_0.txt"))

    X_text = np.random.rand(n_apps, 1058).astype(np.float32)
    X_img  = np.random.rand(n_apps, 1552).astype(np.float32)

    runs = 5
    text_times, img_times, fuse_times = [], [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        p_t = mdl_t.predict(sel_t.transform(X_text))
        text_times.append((time.perf_counter() - t0) / n_apps)

        t0 = time.perf_counter()
        p_i = mdl_i.predict(sel_i.transform(X_img))
        img_times.append((time.perf_counter() - t0) / n_apps)

        t0 = time.perf_counter()
        _ = np.maximum(p_t, p_i)
        fuse_times.append((time.perf_counter() - t0) / n_apps)

    return {
        "lgbm_text_s_per_app":  round(float(np.mean(text_times)), 4),
        "lgbm_image_s_per_app": round(float(np.mean(img_times)),  4),
        "fusion_s_per_app":     round(float(np.mean(fuse_times)), 4),
    }


def main():
    out_dir = Path(CFG.runs_dir) / CFG.run_name / "latency"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(CFG.raw_inference_dataset_path))
    print(f"Timing on {len(records)} apps (sample 20 for neural encoders)...")

    results = {}

    results["bge_encode_s_per_app"] = round(time_bge_encoding(records), 4)
    results["clip_encode_s_per_app"] = round(time_clip_encoding(records), 4)
    results["ocr_s_per_app"] = round(time_ocr(records), 4)
    results.update(time_lgbm_inference(n_apps=110))

    results["total_s_per_app"] = round(
        results["bge_encode_s_per_app"]
        + results["clip_encode_s_per_app"]
        + results["ocr_s_per_app"]
        + results["lgbm_text_s_per_app"]
        + results["lgbm_image_s_per_app"]
        + results["fusion_s_per_app"],
        4,
    )

    test_dir = Path(CFG.runs_dir) / CFG.run_name / "independent_test"
    baseline_latency_files = {
        "TF-IDF + linear SVM":              test_dir / "baseline_tfidf_svm.json",
        "Qwen2.5-7B (desc. only)":         test_dir / "baseline_qwen.json",
        "GPT-4o-mini (zero-shot)":          test_dir / "baseline_mllm_zeroshot_gpt_4o_mini.json",
        "GPT-4o (6-shot)":                  test_dir / "baseline_mllm_fewshot_gpt4o.json",
        "E2E transformer":                  test_dir / "baseline_e2e_transformer.json",
    }
    baseline_latencies = {}
    for name, path in baseline_latency_files.items():
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            if "latency_s_per_app" in d:
                baseline_latencies[name] = d["latency_s_per_app"]

    results["baseline_latencies"] = baseline_latencies
    write_json(out_dir / "table16_latency.json", results)

    print("\nTable 16 — LLMDroid components:")
    print(f"  BGE encode:       {results['bge_encode_s_per_app']:.3f} s/app")
    print(f"  CLIP encode (4):  {results['clip_encode_s_per_app']:.3f} s/app")
    print(f"  OCR (4 screens):  {results['ocr_s_per_app']:.3f} s/app")
    print(f"  LightGBM (text):  {results['lgbm_text_s_per_app']:.4f} s/app")
    print(f"  LightGBM (image): {results['lgbm_image_s_per_app']:.4f} s/app")
    print(f"  Fusion:           {results['fusion_s_per_app']:.4f} s/app")
    print(f"  {'─' * 33}")
    print(f"  TOTAL:            {results['total_s_per_app']:.3f} s/app")

    if baseline_latencies:
        print("\nTable 16 — Baselines:")
        for name, lat in baseline_latencies.items():
            print(f"  {name:<35} {lat:.3f} s/app")
    else:
        print("\n[info] No baseline latencies found. Run baseline scripts first to populate.")

    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
