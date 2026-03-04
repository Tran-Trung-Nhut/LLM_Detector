"""
Preprocessing module for cleaning and preparing app data
- Clean text fields (strip HTML, normalize whitespace)
- Deduplicate images (perceptual hash)
- Build a unified `text` field for training/inference reproducibility
"""
import json
import re
from pathlib import Path
from PIL import Image
import imagehash
from html import unescape
from src.config import CFG


def dedup_image_paths(image_paths, max_dist=4):
    """
    Remove duplicate images based on perceptual hash
    """
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


def clean_html(text):
    """
    Remove HTML tags and unescape HTML entities
    """
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    return text


def normalize_whitespace(text):
    """
    Normalize whitespace in text
    """
    if not text:
        return ""
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = text.strip()
    return text


def clean_text(text):
    """
    Clean text - remove HTML and normalize whitespace
    """
    if not text:
        return ""
    return normalize_whitespace(clean_html(text))


def build_app_text(record: dict, include_recent_changes: bool = True) -> str:
    """
    Build a single text field used by all models.
    """
    title = clean_text(record.get("title", ""))
    cat = clean_text(record.get("category", ""))
    short_desc = clean_text(record.get("short_description", ""))
    desc = clean_text(record.get("description", ""))
    recent = clean_text(record.get("recent_changes_text", "")) if include_recent_changes else ""

    parts = []
    if title:
        parts.append(f"[TITLE] {title}")
    if cat:
        parts.append(f"[CATEGORY] {cat}")
    if short_desc:
        parts.append(f"[SHORT_DESCRIPTION] {short_desc}")
    if desc:
        parts.append(f"[DESCRIPTION]\n{desc}")
    if recent:
        parts.append(f"[RECENT_CHANGES]\n{recent}")

    return "\n".join(parts).strip()


def main():
    input_file = CFG.raw_dataset_path
    output_file = CFG.dataset_path

    print(f"Đọc file {input_file}...")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "images_before": 0,
        "images_after": 0,
        "cleaned_fields": 0,
        "text_built": 0,
    }

    text_fields = ["description", "short_description", "recent_changes_text", "title", "category"]

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            record = json.loads(line)
            stats["total"] += 1

            for field in text_fields:
                if field in record and record[field]:
                    original = record[field]
                    cleaned = clean_text(original)
                    if original != cleaned:
                        stats["cleaned_fields"] += 1
                    record[field] = cleaned

            raw_paths = record.get("image_paths", [])
            stats["images_before"] += len(raw_paths)
            unique_paths = dedup_image_paths(raw_paths, max_dist=4)
            stats["images_after"] += len(unique_paths)
            record["image_paths"] = unique_paths

            record["text"] = build_app_text(record, include_recent_changes=True)
            stats["text_built"] += 1

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("KẾT QUẢ:")
    print(f"  Tổng số apps: {stats['total']}")
    print(f"  Text fields cleaned: {stats['cleaned_fields']}")
    print(f"  Text built: {stats['text_built']}")
    print(f"  Images before: {stats['images_before']}")
    print(f"  Images after: {stats['images_after']}")
    print(f"  Images removed: {stats['images_before'] - stats['images_after']}")
    print(f"\nĐầu ra: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()