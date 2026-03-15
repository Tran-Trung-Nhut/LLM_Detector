"""
extract_image_features.py — Image branch feature extraction using CLIP.

Extracts per app:
  1. CLIP image embeddings  (mean + max pooled over all screenshots → 768-d each)
  2. CLIP zero-shot similarity scores  (cosine sim vs positive/negative prompts → ~9 scores)
  3. OCR-based keyword features from screenshot text  (13 features, same schema as text branch)

Output: one .npz file under features_dir/image/
"""
import json
import os
import re
import math
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from extract_text_features import compute_keyword_features
from utils.io import read_jsonl


# ── CLIP model loading ───────────────────────────────────────────────────────

def load_clip_model():
    processor = CLIPProcessor.from_pretrained(CFG.clip_model)
    model = CLIPModel.from_pretrained(CFG.clip_model, dtype=torch.float16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device


# ── Encode images in batches ─────────────────────────────────────────────────

def encode_images_clip(image_paths: list[str], processor, model, device,
                       batch_size: int = 16) -> np.ndarray:
    """Return (N_images, embed_dim) float32 array of CLIP image embeddings."""
    all_embeds = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                continue
        if not images:
            continue
            
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            # Bước 1: Trích xuất đầu ra thô từ Vision Model
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            # Bước 2: Lấy vector Pooling
            pooled_output = vision_outputs.pooler_output if hasattr(vision_outputs, "pooler_output") else vision_outputs[1]
            # Bước 3: Chiếu (Project) về đúng 768 chiều
            embeds = model.visual_projection(pooled_output)
            
        # Chuẩn hóa (Normalize)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu().float().numpy())

    if all_embeds:
        return np.concatenate(all_embeds, axis=0)
    return np.zeros((0, CFG.clip_embed_dim), dtype=np.float32)


def encode_texts_clip(texts: list[str], processor, model, device) -> np.ndarray:
    """Return (N_texts, embed_dim) float32 array of CLIP text embeddings."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        # Bước 1: Trích xuất đầu ra thô từ Text Model
        text_outputs = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        # Bước 2: Lấy vector Pooling
        pooled_output = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs[1]
        # Bước 3: Chiếu (Project) về đúng 768 chiều
        embeds = model.text_projection(pooled_output)
        
    # Chuẩn hóa (Normalize)
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds.cpu().float().numpy()

# ── Zero-shot similarity ─────────────────────────────────────────────────────

def compute_zeroshot_scores(image_embeds: np.ndarray,
                            pos_text_embeds: np.ndarray,
                            neg_text_embeds: np.ndarray) -> np.ndarray:
    """
    Given per-image CLIP embeds and prompt embeds, compute per-app scores.
    Returns feature vector:
      - max_pos_sim: max similarity to any positive prompt across all images  (1)
      - mean_pos_sim: mean similarity to positive prompts                     (1)
      - per-positive-prompt max sim                                           (n_pos)
      - max_neg_sim                                                           (1)
      - mean_neg_sim                                                          (1)
      - pos_minus_neg: max_pos - max_neg (discriminative gap)                 (1)
    """
    if image_embeds.shape[0] == 0:
        n_feats = 4 + pos_text_embeds.shape[0] + 1
        return np.zeros(n_feats, dtype=np.float32)

    # (N_images, N_pos)
    pos_sims = image_embeds @ pos_text_embeds.T
    # (N_images, N_neg)
    neg_sims = image_embeds @ neg_text_embeds.T

    max_pos = float(pos_sims.max())
    mean_pos = float(pos_sims.mean())
    per_prompt_max = pos_sims.max(axis=0).tolist()
    max_neg = float(neg_sims.max())
    mean_neg = float(neg_sims.mean())
    gap = max_pos - max_neg

    return np.array(
        [max_pos, mean_pos] + per_prompt_max + [max_neg, mean_neg, gap],
        dtype=np.float32,
    )


# ── OCR keyword features ─────────────────────────────────────────────────────

def compute_ocr_features(record: dict) -> np.ndarray:
    """
    Aggregate OCR text from all screenshots, then extract keyword features.
    Additional features:
      - total OCR text length  (1)
      - fraction of images with non-empty OCR  (1)
    Plus keyword features (13) → total 15.
    """
    ocr_map = record.get("ocr_by_image", {})
    all_texts = list(ocr_map.values()) if ocr_map else []
    combined = " ".join(t for t in all_texts if t)
    n_images = max(len(record.get("image_paths", [])), 1)
    n_with_text = sum(1 for t in all_texts if t.strip())

    kw_feats = compute_keyword_features(combined)  # (13,)
    extra = np.array([
        float(len(combined)),
        float(n_with_text) / float(n_images),
    ], dtype=np.float32)
    return np.concatenate([extra, kw_feats])


# ── Main extraction ──────────────────────────────────────────────────────────

def main():
    rows = read_jsonl(CFG.dataset_path)
    out_dir = Path(CFG.features_dir) / "image"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP model: {CFG.clip_model} ...")
    processor, model, device = load_clip_model()

    # Pre-encode text prompts for zero-shot
    pos_embeds = encode_texts_clip(list(CFG.clip_positive_prompts), processor, model, device)
    neg_embeds = encode_texts_clip(list(CFG.clip_negative_prompts), processor, model, device)

    app_ids, labels = [], []
    clip_mean_list, clip_max_list = [], []
    zs_list, ocr_list = [], []

    for r in tqdm(rows, desc="Image features"):
        app_ids.append(r["app_id"])
        labels.append(r["label_binary"])
        img_paths = r.get("image_paths", [])

        # CLIP image embeddings
        if img_paths:
            img_embeds = encode_images_clip(img_paths, processor, model, device, CFG.clip_batch_size)
        else:
            img_embeds = np.zeros((0, CFG.clip_embed_dim), dtype=np.float32)

        # Pooling
        if img_embeds.shape[0] > 0:
            clip_mean_list.append(img_embeds.mean(axis=0))
            clip_max_list.append(img_embeds.max(axis=0))
        else:
            clip_mean_list.append(np.zeros(CFG.clip_embed_dim, dtype=np.float32))
            clip_max_list.append(np.zeros(CFG.clip_embed_dim, dtype=np.float32))

        # Zero-shot scores
        zs_feats = compute_zeroshot_scores(img_embeds, pos_embeds, neg_embeds)
        zs_list.append(zs_feats)

        # OCR keyword features
        ocr_feats = compute_ocr_features(r)
        ocr_list.append(ocr_feats)

    clip_mean = np.stack(clip_mean_list)    # (N, 768)
    clip_max = np.stack(clip_max_list)      # (N, 768)
    zeroshot = np.stack(zs_list)            # (N, ~12)
    ocr = np.stack(ocr_list)               # (N, 15)

    np.savez_compressed(
        out_dir / "features.npz",
        app_ids=np.array(app_ids),
        labels=np.array(labels, dtype=np.int32),
        clip_mean=clip_mean,
        clip_max=clip_max,
        zeroshot=zeroshot,
        ocr=ocr,
    )
    print(f"Saved image features → {out_dir / 'features.npz'}")
    print(f"  clip_mean: {clip_mean.shape}")
    print(f"  clip_max:  {clip_max.shape}")
    print(f"  zeroshot:  {zeroshot.shape}")
    print(f"  ocr:       {ocr.shape}")

    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
