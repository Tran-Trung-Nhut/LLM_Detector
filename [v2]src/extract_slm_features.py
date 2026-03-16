"""
extract_slm_features.py — SLM Text Reasoning Module.

Sử dụng Small Language Model (SLM) để đọc hiểu mô tả ứng dụng và suy luận
xem ứng dụng đó có khả năng tích hợp LLM hay không. 
Output: 1 đặc trưng (Reasoning Score từ 0.0 đến 1.0)
"""
import os
import sys
import numpy as np
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils.io import read_jsonl

SYSTEM_PROMPT = CFG.slm_system_prompt
USER_PROMPT_TEMPLATE = CFG.slm_user_prompt_template

def extract_score(text: str) -> float:
    """Dùng Regex để tìm con số đầu tiên trong câu trả lời của SLM"""
    match = re.search(r'\d+', text)
    if match:
        score = float(match.group())
        return min(max(score / 100.0, 0.0), 1.0) # Chuẩn hóa về 0.0 - 1.0
    return 0.0

def main():
    rows = read_jsonl(CFG.dataset_path)
    out_dir = Path(CFG.features_dir) / "slm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SLM Reasoning Model: {CFG.slm_model} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CFG.slm_model)
    model = AutoModelForCausalLM.from_pretrained(
        CFG.slm_model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    app_ids = []
    labels = []
    slm_scores = []

    # Xử lý từng batch để tránh OOM
    for i in tqdm(range(0, len(rows), CFG.slm_batch_size), desc="SLM Reasoning"):
        batch = rows[i: i + CFG.slm_batch_size]
        prompts = []
        for r in batch:
            app_ids.append(r["app_id"])
            labels.append(r["label_binary"])
            
            # Tận dụng luôn text đã được clean rất sạch từ preprocessing
            text = r.get("text", "")[:CFG.slm_text_max_length] # Cắt bớt nếu quá dài để tránh vỡ ngữ cảnh
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{USER_PROMPT_TEMPLATE.format(text=text)}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=CFG.slm_max_new_tokens,
                do_sample=False, # Tham lam (Greedy) để kết quả ổn định
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Bóc tách kết quả
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for res in responses:
            slm_scores.append([extract_score(res)])

    slm_feats = np.array(slm_scores, dtype=np.float32)

    np.savez_compressed(
        out_dir / "features.npz",
        app_ids=np.array(app_ids),
        labels=np.array(labels, dtype=np.int32),
        slm_score=slm_feats
    )
    print(f"Saved SLM reasoning features → {out_dir / 'features.npz'}")
    print(f"  slm_score: {slm_feats.shape}")

    del model, tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()