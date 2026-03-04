import json
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

from src.config import CFG
from src.prompts import BINARY_PROMPT
from src.utils.io import read_jsonl, write_predictions_csv, write_json
from src.utils.metrics import compute_binary_metrics


@torch.no_grad()
def prob_yes_single(model, processor, text, image, device, max_text_len=512):
    prompt = BINARY_PROMPT.format(text=text)
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_len,
    ).to(device)

    out = model(**inputs)
    yes_id = processor.tokenizer("YES", add_special_tokens=False)["input_ids"][0]
    last_logits = out.logits[:, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    return float(probs[0, yes_id].detach().cpu())


def infer_one_fold(fold: int):
    dataset_path = CFG.dataset_path
    split_path = f"{CFG.splits_dir}/fold_{fold}.json"
    lora_path = f"{CFG.runs_dir}/{CFG.train_run_name}/fold_{fold}/lora_adapter"
    out_path = f"{CFG.runs_dir}/{CFG.infer_run_name}/fold_{fold}/predictions.csv"
    pooling = CFG.multi_image_pooling

    rows = read_jsonl(dataset_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    test_ids = set(split["test_ids"])
    test_rows = [r for r in rows if r["app_id"] in test_ids]

    processor = AutoProcessor.from_pretrained(CFG.base_model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        CFG.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    device = next(model.parameters()).device

    pred_rows = []
    y_true, y_prob = [], []

    for r in tqdm(test_rows, desc=f"fold {fold} multi-image infer"):
        probs = []
        for img_path in r["image_paths"]:
            image = Image.open(img_path).convert("RGB")
            p = prob_yes_single(model, processor, r["text"], image, device, max_text_len=CFG.max_text_len)
            probs.append(p)

        if pooling == "max":
            p_app = max(probs) if probs else 0.0
        elif pooling == "mean":
            p_app = sum(probs) / max(len(probs), 1)
        else:
            raise ValueError(pooling)

        y_true.append(int(r["label_binary"]))
        y_prob.append(float(p_app))
        pred_rows.append({
            "app_id": r["app_id"],
            "y_true": int(r["label_binary"]),
            "y_prob_yes": float(p_app),
            "pooling": pooling,
            "max_image_prob": max(probs) if probs else None,
            "mean_image_prob": (sum(probs)/len(probs)) if probs else None,
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(out_path, pred_rows)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    write_json(str(Path(out_path).with_suffix(".metrics.json")), metrics)
    print(f"[fold {fold}] {metrics}")


def main():
    for fold in range(CFG.n_folds):
        infer_one_fold(fold)


if __name__ == "__main__":
    main()