import json
from pathlib import Path
import random
from PIL import Image
from src.keywords import KEYWORDS

class AppsSingleImageDataset:
    def __init__(self, rows, image_strategy="best", seed=42):
        self.rows = rows
        self.image_strategy = image_strategy
        self.rng = random.Random(seed)

    def _score_with_ocr(self, r, img_path):
        ocr = ""
        ocr_map = r.get("ocr_by_image", None)
        if isinstance(ocr_map, dict):
            ocr = (ocr_map.get(img_path) or "").lower()

        has_kw = any(k in ocr for k in KEYWORDS)
        length = len(ocr)
        return (5.0 if has_kw else 0.0) + (length ** 0.5) * 0.1

    def pick_image(self, r):
        paths = r.get("image_paths") or []
        if not paths:
            return None

        if self.image_strategy == "first":
            return paths[0]
        if self.image_strategy == "random":
            return self.rng.choice(paths)
        if self.image_strategy == "best":
            scored = [(self._score_with_ocr(r, p), p) for p in paths]
            scored.sort(reverse=True, key=lambda x: x[0])
            best_score, best_path = scored[0]
            if best_score == 0.0 and not isinstance(r.get("ocr_by_image", None), dict):
                print(f"Warning: no OCR info for app_id {r['app_id']}, falling back to middle image")
                return paths[len(paths)//2] #fallback to middle image if no OCR info available
            return best_path

        raise ValueError(self.image_strategy)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_path = self.pick_image(r)
        image = Image.open(img_path).convert("RGB")
        return {
            "app_id": r["app_id"],
            "text": r["text"],
            "label_binary": int(r["label_binary"]),
            "image": image,
        }