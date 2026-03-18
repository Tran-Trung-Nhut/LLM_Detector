"""
Preprocessing: clean text, deduplicate images, build unified text field.
Reused from V1 with import path updated to [v2]src.config.
"""
import json
import os
import re
import sys
from pathlib import Path
from PIL import Image
import imagehash
from html import unescape

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG


def dedup_image_paths(image_paths, max_dist=None):
    if max_dist is None:
        max_dist = CFG.image_dedup_max_dist
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
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    return text


# Footer markers are now centralized in config.py
FOOTER_MARKERS = list(CFG.footer_markers)


def remove_low_signal(text):
    if not text:
        return ""
    t = text
    t = re.sub(r"https?://[^\s]+", "", t)
    t = re.sub(r"www\.[^\s]+", "", t)
    t = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", t)
    t = re.sub(r"@[a-zA-Z0-9_]+", "", t)

    lower = t.lower()
    cut_pos = None
    for m in FOOTER_MARKERS:
        mm = re.search(m, lower)
        if mm and mm.start() >= 500:
            cut_pos = mm.start() if cut_pos is None else min(cut_pos, mm.start())
    if cut_pos is not None:
        t = t[:cut_pos]

    t = re.sub(r"(?im)^[\s\-•*]*rate\s*us.*$", "", t)
    t = re.sub(r"(?im)^[\s\-•*]*give\s*us\s*\d+\s*stars?.*$", "", t)
    t = re.sub(r"(?im)^[\s\-•*]*don'?t\s*forget\s*to\s*(rate|review).*$", "", t)
    t = re.sub(r"(?im)^[\s\-•*]*⭐+.*$", "", t)
    return t


def normalize_whitespace(text):
    if not text:
        return ""
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def clean_text(text):
    if not text:
        return ""
    return normalize_whitespace(remove_low_signal(clean_html(text)))


def build_app_text(record: dict, include_recent_changes: bool = True) -> str:
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
    print(f"Reading {input_file} ...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    stats = dict(total=0, images_before=0, images_after=0,
                 cleaned_fields=0, text_built=0, chars_before=0, chars_after=0)
    text_fields = ["description", "short_description", "recent_changes_text", "title", "category"]

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            stats["total"] += 1
            for fld in text_fields:
                if fld in record and record[fld]:
                    orig = record[fld]
                    stats["chars_before"] += len(orig)
                    cleaned = clean_text(orig)
                    stats["chars_after"] += len(cleaned)
                    if orig != cleaned:
                        stats["cleaned_fields"] += 1
                    record[fld] = cleaned

            raw_paths = record.get("image_paths", [])
            stats["images_before"] += len(raw_paths)
            unique_paths = dedup_image_paths(raw_paths, max_dist=4)
            stats["images_after"] += len(unique_paths)
            record["image_paths"] = unique_paths

            record["text"] = build_app_text(record, include_recent_changes=True)
            stats["text_built"] += 1
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Apps: {stats['total']}")
    print(f"  Chars: {stats['chars_before']:,} → {stats['chars_after']:,} "
          f"(removed {stats['chars_before'] - stats['chars_after']:,})")
    print(f"  Images: {stats['images_before']} → {stats['images_after']} "
          f"(deduped {stats['images_before'] - stats['images_after']})")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
