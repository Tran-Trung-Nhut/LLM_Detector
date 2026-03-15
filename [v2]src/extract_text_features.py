"""
extract_text_features.py — Text branch feature extraction.

Extracts three types of text features per app:
  1. Sentence-BERT embedding of the cleaned description  (1024-d for BGE-large)
  2. Keyword match features  (count, weighted score, per-category binary)
  3. Handcrafted metadata features  (description length, category one-hot, etc.)

Output: one .npz file per app_id stored under features_dir/text/
"""
import os
import re
import math
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from keywords import KEYWORD_CATEGORIES, TOP_CATEGORIES_KEYWORDS

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils.io import read_jsonl


# ── Keyword feature helpers ──────────────────────────────────────────────────
def compute_keyword_features(text: str) -> np.ndarray:
    """
    Return a feature vector from keyword matching on *text*.
      - total keyword hit count  (1)
      - log(1 + count)           (1)
      - per-category binary      (5 categories)
      - per-category count       (5 categories)
      - max keyword match length (1, proxy for specificity)
    Total: 13 features
    """
    lower = text.lower()

    total_count = 0
    cat_counts = {}
    max_kw_len = 0

    for cat, kws in KEYWORD_CATEGORIES.items():
        cat_count = 0
        for kw in kws:
            n = len(re.findall(re.escape(kw), lower))
            if n > 0:
                cat_count += n
                max_kw_len = max(max_kw_len, len(kw))
        cat_counts[cat] = cat_count
        total_count += cat_count

    feats = [
        total_count,
        math.log1p(total_count),
    ]
    for cat in KEYWORD_CATEGORIES:
        feats.append(1.0 if cat_counts[cat] > 0 else 0.0)
        feats.append(float(cat_counts[cat]))
    feats.append(float(max_kw_len))

    return np.array(feats, dtype=np.float32)


# ── Handcrafted metadata features ────────────────────────────────────────────


def compute_meta_features(record: dict) -> np.ndarray:
    """
    Handcrafted features from metadata fields.
      - description length (chars)       (1)
      - short_description length          (1)
      - title length                      (1)
      - has recent_changes                (1)
      - num images                        (1)
      - category one-hot                  (len(TOP_CATEGORIES) + 1 for 'other')
    Total: 5 + 16 = 21 features
    """
    desc = record.get("description", "") or ""
    short = record.get("short_description", "") or ""
    title = record.get("title", "") or ""
    recent = record.get("recent_changes_text", "") or ""
    n_images = len(record.get("image_paths", []))
    cat = (record.get("category", "") or "").strip()

    feats = [
        float(len(desc)),
        float(len(short)),
        float(len(title)),
        1.0 if len(recent.strip()) > 0 else 0.0,
        float(n_images),
    ]

    cat_onehot = [0.0] * (len(TOP_CATEGORIES_KEYWORDS) + 1)
    if cat in TOP_CATEGORIES_KEYWORDS:
        cat_onehot[TOP_CATEGORIES_KEYWORDS.index(cat)] = 1.0
    else:
        cat_onehot[-1] = 1.0
    feats.extend(cat_onehot)

    return np.array(feats, dtype=np.float32)


# ── Sentence-BERT embedding ──────────────────────────────────────────────────

def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model)
    model = AutoModel.from_pretrained(CFG.text_model, torch_dtype=torch.float16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def encode_texts(texts: list[str], tokenizer, model, device,
                 batch_size: int = 32) -> np.ndarray:
    """Encode a list of texts into embeddings using mean pooling."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        # Mean pooling over non-padding tokens
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embeds = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embeds.append(embeds.cpu().float().numpy())
    return np.concatenate(all_embeds, axis=0)


# ── Main extraction ──────────────────────────────────────────────────────────

def main():
    """Extract and cache all text features."""
    rows = read_jsonl(CFG.dataset_path)
    out_dir = Path(CFG.features_dir) / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Keyword + metadata features (fast, CPU)
    print("Extracting keyword & metadata features ...")
    kw_feats, meta_feats, app_ids, labels = [], [], [], []
    for r in tqdm(rows, desc="keyword+meta"):
        app_ids.append(r["app_id"])
        labels.append(r["label_binary"])
        text = r.get("text", "")
        kw_feats.append(compute_keyword_features(text))
        meta_feats.append(compute_meta_features(r))

    kw_feats = np.stack(kw_feats)      # (N, 13)
    meta_feats = np.stack(meta_feats)  # (N, 21)

    # 2) Sentence-BERT embeddings (GPU)
    print(f"Loading text model: {CFG.text_model} ...")
    tokenizer, model, device = load_text_model()
    texts = [r.get("text", "") for r in rows]
    print(f"Encoding {len(texts)} texts ...")
    sbert_feats = encode_texts(texts, tokenizer, model, device, CFG.text_batch_size)  # (N, 1024)

    # 3) Save
    np.savez_compressed(
        out_dir / "features.npz",
        app_ids=np.array(app_ids),
        labels=np.array(labels, dtype=np.int32),
        sbert=sbert_feats,
        keywords=kw_feats,
        meta=meta_feats,
    )
    print(f"Saved text features → {out_dir / 'features.npz'}")
    print(f"  sbert:    {sbert_feats.shape}")
    print(f"  keywords: {kw_feats.shape}")
    print(f"  meta:     {meta_feats.shape}")

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
