"""
Preprocessing module for cleaning and preparing app data
"""
import json
import re
from pathlib import Path
from PIL import Image
import imagehash
from html import unescape
from datetime import datetime


def dedup_image_paths(image_paths, max_dist=4):
    """
    Remove duplicate images based on perceptual hash
    
    Args:
        image_paths: List of image file paths
        max_dist: Maximum hash distance to consider images as duplicates
        
    Returns:
        List of unique image paths
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
    
    Args:
        text: Text containing HTML
        
    Returns:
        Clean text without HTML
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Unescape HTML entities (&nbsp;, &lt;, etc.)
    text = unescape(text)
    
    return text


def normalize_whitespace(text):
    """
    Normalize whitespace in text
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_text(text):
    """
    Clean text - remove HTML and normalize whitespace
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    text = clean_html(text)
    text = normalize_whitespace(text)
    
    return text


def main():
    input_file = 'data/apps.raw.jsonl'
    output_file = 'data/apps.jsonl'
    
    print(f"Đọc file {input_file}...")
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create backup
    backup_file = f"{output_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Tạo backup: {backup_file}")
    
    stats = {
        'total': 0,
        'images_before': 0,
        'images_after': 0,
        'cleaned_fields': 0
    }
    
    # Fields to clean
    text_fields = ['description', 'short_description', 'recent_changes_text', 'title']
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out, \
         open(backup_file, 'w', encoding='utf-8') as f_backup:
        
        for line in f_in:
            record = json.loads(line)
            stats['total'] += 1
            
            # Write to backup
            f_backup.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Clean text fields
            for field in text_fields:
                if field in record and record[field]:
                    original = record[field]
                    cleaned = clean_text(original)
                    if original != cleaned:
                        stats['cleaned_fields'] += 1
                    record[field] = cleaned
            
            # Deduplicate images
            raw_paths = record.get("image_paths", [])
            stats['images_before'] += len(raw_paths)
            
            unique_paths = dedup_image_paths(raw_paths, max_dist=4)
            stats['images_after'] += len(unique_paths)
            
            record["image_paths"] = unique_paths
            
            # Write to output
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n" + "="*50)
    print("KẾT QUẢ:")
    print(f"  Tổng số apps: {stats['total']}")
    print(f"  Text fields cleaned: {stats['cleaned_fields']}")
    print(f"  Images before: {stats['images_before']}")
    print(f"  Images after: {stats['images_after']}")
    print(f"  Images removed: {stats['images_before'] - stats['images_after']}")
    print(f"\nĐầu ra: {output_file}")
    print("="*50)


if __name__ == "__main__":
    main()
