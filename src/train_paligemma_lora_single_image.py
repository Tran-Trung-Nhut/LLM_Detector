import os
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model

from src.utils_seed import set_seed
from src.utils_io import read_jsonl, write_json, write_predictions_csv
from src.utils_metrics import compute_binary_metrics
from src.dataset_apps import AppsSingleImageDataset
from src.prompts import BINARY_PROMPT

@dataclass
class TrainConfig:
    model_name: str = "google/paligemma-3b-pt-224"
    seed: int = 42
    lr: float = 2e-4
    weight_decay: float = 0.0
    num_epochs: int = 5
    batch_size: int = 1
    grad_accum: int = 16
    max_text_len: int = 512
    image_strategy: str = "first"
    output_dir: str = "runs/paligemma_single_image"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

def collate_fn(processor, batch, max_text_len):
    images = [b["image"] for b in batch]
    prompts = [BINARY_PROMPT.format(text=b["text"]) for b in batch]
    # target: YES/NO
    targets = ["YES" if b["label_binary"] == 1 else "NO" for b in batch]

    # PaliGemma processor handles image + text
    model_inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_len,
    )

    with processor.as_target_processor():
        labels = processor(
            text=targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8,
        )["input_ids"]

    # Replace padding token id with -100 for loss
    labels[labels == processor.tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels

    meta = [{"app_id": b["app_id"], "y": b["label_binary"]} for b in batch]
    return model_inputs, meta

@torch.no_grad()
def infer_yes_prob(model, processor, dataloader, device, max_text_len):
    model.eval()
    y_true, y_prob, rows = [], [], []

    yes_ids = processor.tokenizer("YES", add_special_tokens=False)["input_ids"]
    no_ids = processor.tokenizer("NO", add_special_tokens=False)["input_ids"]
    # assume single token; if not, take first token
    yes_id = yes_ids[0]
    no_id = no_ids[0]

    for (inputs, meta) in tqdm(dataloader, desc="infer"):
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(device)

        out = model(**inputs)
        # logits: [B, T, V]. We only have target length ~1 token.
        # Take last token position where labels != -100 (simple: last position)
        logits = out.logits  # [B, seq, vocab]
        last_logits = logits[:, -1, :]  # [B, vocab]
        probs = torch.softmax(last_logits, dim=-1)
        p_yes = probs[:, yes_id].detach().cpu().tolist()

        for i, m in enumerate(meta):
            y_true.append(m["y"])
            y_prob.append(p_yes[i])
            rows.append({
                "app_id": m["app_id"],
                "y_true": m["y"],
                "y_prob_yes": p_yes[i],
            })

    return y_true, y_prob, rows

def main(
    dataset_path="data/apps.jsonl",
    split_path="data/splits/fold_0.json",
    run_dir="runs/paligemma_single_image/fold_0",
):
    cfg = TrainConfig(output_dir=run_dir)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(run_dir) / "config.json", cfg.__dict__)

    set_seed(cfg.seed)

    rows = read_jsonl(dataset_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_rows = [r for r in rows if r["app_id"] in set(split["train_ids"])]
    test_rows = [r for r in rows if r["app_id"] in set(split["test_ids"])]

    processor = AutoProcessor.from_pretrained(cfg.model_name)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    device = next(model.parameters()).device

    train_ds = AppsSingleImageDataset(train_rows, image_strategy=cfg.image_strategy, seed=cfg.seed)
    test_ds = AppsSingleImageDataset(test_rows, image_strategy="first", seed=cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: collate_fn(processor, b, cfg.max_text_len),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: collate_fn(processor, b, cfg.max_text_len),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        optim.zero_grad(set_to_none=True)
        for (inputs, _meta) in pbar:
            for k in list(inputs.keys()):
                inputs[k] = inputs[k].to(device)
            out = model(**inputs)
            loss = out.loss / cfg.grad_accum
            loss.backward()

            global_step += 1
            if global_step % cfg.grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": float(loss.detach().cpu())})

        # quick eval each epoch (optional)
        y_true, y_prob, pred_rows = infer_yes_prob(model, processor, test_loader, device, cfg.max_text_len)
        metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
        write_json(Path(run_dir) / f"metrics_epoch_{epoch}.json", metrics)
        write_predictions_csv(Path(run_dir) / f"predictions_epoch_{epoch}.csv", pred_rows)

    # final save adapter
    model.save_pretrained(Path(run_dir) / "lora_adapter")
    print("Done. Saved LoRA adapter.")

if __name__ == "__main__":
    main()