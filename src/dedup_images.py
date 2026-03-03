import json
from pathlib import Path
from PIL import Image
import imagehash

def dedup_image_paths(image_paths, max_dist=4):
    hashes = []
    kept = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        h = imagehash.phash(img)
        if any((h - h2) <= max_dist for h2 in hashes):
            continue
        hashes.append(h)
        kept.append(p)
    return kept

def main(in_path="data/apps.raw.jsonl", out_path="data/apps.jsonl", max_dist=4):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            r = json.loads(line)
            raw = r["image_paths"]
            unique = dedup_image_paths(raw, max_dist=max_dist)
            r["image_paths"] = unique 
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote dedup dataset: {out_path}")

if __name__ == "__main__":
    main()