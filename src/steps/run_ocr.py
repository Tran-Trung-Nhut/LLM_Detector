"""
Run Tesseract OCR on all app screenshots. Reused from V1.
"""
import json
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pytesseract

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG


def run_ocr_on_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return pytesseract.image_to_string(img, lang=CFG.ocr_lang).strip()
    except Exception:
        return ""


def main():
    rows = []
    with open(CFG.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    total_images = sum(len(r.get("image_paths", [])) for r in rows)
    print(f"Found {len(rows)} apps with {total_images} images")

    processed = 0
    pbar = tqdm(total=total_images, desc="Running OCR")
    for row in rows:
        if "ocr_by_image" not in row or not isinstance(row["ocr_by_image"], dict):
            row["ocr_by_image"] = {}
        for img_path in row.get("image_paths", []):
            if img_path not in row["ocr_by_image"]:
                row["ocr_by_image"][img_path] = run_ocr_on_image(img_path)
                processed += 1
            pbar.update(1)
    pbar.close()

    if processed > 0:
        with open(CFG.dataset_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        total_ocr = sum(len(r.get("ocr_by_image", {})) for r in rows)
        non_empty = sum(1 for r in rows for t in r.get("ocr_by_image", {}).values() if t.strip())
        print(f"OCR done — {non_empty}/{total_ocr} images have text")
    else:
        print("All images already have OCR data")


if __name__ == "__main__":
    main()
