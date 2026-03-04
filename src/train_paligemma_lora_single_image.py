import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model

from src.config import CFG
from src.dataset_apps import AppsSingleImageDataset
from src.prompts import BINARY_PROMPT
from src.tils.seed import set_seed
from src.utils.io import read_jsonl, write_json, write_predictions_csv
from src.utils.metrics import compute_binary_metrics


@dataclass
class TrainConfig:
    model_name: str = CFG.base_model
    seed: int = CFG.seed
    lr: float = CFG.lr
    weight_decay: float = CFG.weight_decay
    num_epochs: int = CFG.num_epochs
    batch_size: int = CFG.batch_size
    grad_accum: int = CFG.grad_accum
    max_text_len: int = CFG.max_text_len
    image_strategy: str = CFG.image_strategy

    lora_r: int = CFG.lora_r
    lora_alpha: int = CFG.lora_alpha
    lora_dropout: float = CFG.lora_dropout


def collate_fn(processor, batch, max_text_len):
    images = [b["image"] for b in batch]
    prompts = [BINARY_PROMPT.format(text=b["text"]) for b in batch]
    targets = ["YES" if b["label_binary"] == 1 else "NO" for b in batch]

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

    labels[labels == processor.tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    meta = [{"app_id": b["app_id"], "y": b["label_binary"]} for b in batch]
    return model_inputs, meta


@torch.no_grad()
def infer_yes_prob(model, processor, dataloader, device):
    model.eval()
    y_true, y_prob, rows = [], [], []
    yes_id = processor.tokenizer("YES", add_special_tokens=False)["input_ids"][0]

    for (inputs, meta) in tqdm(dataloader, desc="infer"):
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(device)
        out = model(**inputs)
        last_logits = out.logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        p_yes = probs[:, yes_id].detach().cpu().tolist()

        for i, m in enumerate(meta):
            y_true.append(m["y"])
            y_prob.append(p_yes[i])
            rows.append({"app_id": m["app_id"], "y_true": m["y"], "y_prob_yes": p_yes[i]})
    return y_true, y_prob, rows


def train_one_fold(fold: int):
    cfg = TrainConfig()

    # Paths derived from config
    dataset_path = CFG.dataset_path
    split_path = f"{CFG.splits_dir}/fold_{fold}.json"
    run_dir = f"{CFG.runs_dir}/{CFG.train_run_name}/fold_{fold}"

    Path(run_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(run_dir) / "config.json", {**cfg.__dict__, **CFG.__dict__, "fold": fold})

    set_seed(cfg.seed)

    rows = read_jsonl(dataset_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_ids = set(split["train_ids"])
    test_ids = set(split["test_ids"])
    train_rows = [r for r in rows if r["app_id"] in train_ids]
    test_rows = [r for r in rows if r["app_id"] in test_ids]

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
    test_ds = AppsSingleImageDataset(test_rows, image_strategy=cfg.image_strategy, seed=cfg.seed)

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
        pbar = tqdm(train_loader, desc=f"fold {fold} train epoch {epoch}")
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

        y_true, y_prob, pred_rows = infer_yes_prob(model, processor, test_loader, device)
        metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
        write_json(Path(run_dir) / f"metrics_epoch_{epoch}.json", metrics)
        write_predictions_csv(Path(run_dir) / f"predictions_epoch_{epoch}.csv", pred_rows)

    model.save_pretrained(Path(run_dir) / "lora_adapter")
    print(f"[fold {fold}] saved adapter: {Path(run_dir) / 'lora_adapter'}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Run only fold 0 for quick test")
    args = ap.parse_args()

    folds = [0] if args.demo else range(CFG.n_folds)
    for fold in folds:
        train_one_fold(fold)


if __name__ == "__main__":
    main()