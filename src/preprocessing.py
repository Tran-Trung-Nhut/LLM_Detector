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


# Patterns for low-signal content removal
LOW_SIGNAL_PATTERNS = [
    # URLs and emails
    r"https?://[^\s]+",
    r"www\.[^\s]+",
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    # Social media handles
    r"@[a-zA-Z0-9_]+",
    # Boilerplate section headers (case insensitive, match whole line/paragraph)
    r"(?i)^[\s\-•*]*privacy\s*policy[:\s]*.*$",
    r"(?i)^[\s\-•*]*terms\s*(of\s*(use|service))?[:\s]*.*$",
    r"(?i)^[\s\-•*]*contact\s*us[:\s]*.*$",
    r"(?i)^[\s\-•*]*follow\s*us[:\s]*.*$",
    r"(?i)^[\s\-•*]*connect\s*with\s*us[:\s]*.*$",
    r"(?i)^[\s\-•*]*subscriptions?[:\s]*.*$",
    r"(?i)^[\s\-•*]*feedback[:\s]*.*$",
    r"(?i)^[\s\-•*]*more\s*about\s*.*$",
    r"(?i)^[\s\-•*]*visit\s*(us|our\s*website)[:\s]*.*$",
    # Common footer phrases
    r"(?i)rate\s*us\s*(and\s*)?(write\s*a\s*)?review.*",
    r"(?i)give\s*us\s*\d+\s*stars?.*",
    r"(?i)⭐+.*",
    r"(?i)don'?t\s*forget\s*to\s*(rate|review).*",
]


def remove_low_signal(text):
    """
    Remove URLs/emails and (more importantly) DROP entire boilerplate footer sections
    like Privacy Policy / Terms / Contact / Subscriptions, etc.
    """
    if not text:
        return ""

    t = text

    # 1) remove URLs/emails/handles globally (keep it simple)
    t = re.sub(r"https?://[^\s]+", "", t)
    t = re.sub(r"www\.[^\s]+", "", t)
    t = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", t)
    t = re.sub(r"@[a-zA-Z0-9_]+", "", t)

    # 2) truncate at earliest boilerplate "footer section" marker
    #    (these usually appear near the end but can span multiple paragraphs)
    footer_markers = [
        r"privacy\s*policy",
        r"terms\s*(of\s*(use|service))?",
        r"contact\s*us",
        r"follow\s*us",
        r"connect\s*with\s*us",
        r"subscriptions?",
        r"in[-\s]*app\s*purchases?",
        r"need\s*help",
        r"feedback",
        r"refund",
    ]

    lower = t.lower()
    cut_pos = None
    for m in footer_markers:
        mm = re.search(m, lower)
        if mm:
            pos = mm.start()
            # (e.g., "support" inside feature description). Require some minimum length.
            if pos >= 500:
                cut_pos = pos if cut_pos is None else min(cut_pos, pos)

    if cut_pos is not None:
        t = t[:cut_pos]

    # 3) remove remaining common low-signal lines (still line-based)
    #    Keep this small: only phrases that are almost always non-informative.
    t = re.sub(r"(?im)^[\s\-•*]*rate\s*us.*$", "", t)
    t = re.sub(r"(?im)^[\s\-•*]*give\s*us\s*\d+\s*stars?.*$", "", t)
    t = re.sub(r"(?im)^[\s\-•*]*don'?t\s*forget\s*to\s*(rate|review).*$", "", t)
    t = re.sub(r"(?im)^[\s\-•*]*⭐+.*$", "", t)

    return t


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
    Clean text - remove HTML, low-signal content, and normalize whitespace
    """
    if not text:
        return ""
    text = clean_html(text)
    text = remove_low_signal(text)
    text = normalize_whitespace(text)
    return text


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
        "chars_before": 0,
        "chars_after": 0,
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
                    stats["chars_before"] += len(original)
                    cleaned = clean_text(original)
                    stats["chars_after"] += len(cleaned)
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
    print(f"  Chars before: {stats['chars_before']:,}")
    print(f"  Chars after: {stats['chars_after']:,}")
    print(f"  Chars removed: {stats['chars_before'] - stats['chars_after']:,} ({100*(stats['chars_before'] - stats['chars_after'])/max(stats['chars_before'],1):.1f}%)")
    print(f"  Images before: {stats['images_before']}")
    print(f"  Images after: {stats['images_after']}")
    print(f"  Images removed: {stats['images_before'] - stats['images_after']}")
    print(f"\nĐầu ra: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()