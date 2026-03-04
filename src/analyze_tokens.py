"""
Analyze token counts for dataset texts.
Usage: python -m src.analyze_tokens
"""
import json
from transformers import AutoProcessor
from src.config import CFG
from src.prompts import BINARY_PROMPT


def main():
    processor = AutoProcessor.from_pretrained(CFG.base_model)
    tokenizer = processor.tokenizer

    rows = []
    with open(CFG.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    token_counts = []
    long_apps = []

    for r in rows:
        # Build text như trong training
        text = r.get("text", "")
        if not text:
            # Fallback nếu chưa có field text
            text = f"[TITLE] {r.get('title', '')}\n[CATEGORY] {r.get('category', '')}\n[DESCRIPTION]\n{r.get('description', '')}\n[RECENT_CHANGES]\n{r.get('recent_changes_text', '')}"

        # Build full prompt
        prompt = BINARY_PROMPT.format(text=text)

        # Tokenize
        tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        n_tokens = len(tokens)
        token_counts.append(n_tokens)

        if n_tokens > CFG.max_text_len:
            long_apps.append({
                "app_id": r["app_id"],
                "title": r.get("title", ""),
                "tokens": n_tokens,
                "truncated": n_tokens - CFG.max_text_len,
            })

    # Statistics
    token_counts.sort()
    n = len(token_counts)

    print("=" * 60)
    print("TOKEN COUNT STATISTICS")
    print("=" * 60)
    print(f"Total apps: {n}")
    print(f"Min tokens: {min(token_counts)}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"Mean tokens: {sum(token_counts) / n:.1f}")
    print(f"Median tokens: {token_counts[n // 2]}")
    print(f"P90 tokens: {token_counts[int(n * 0.9)]}")
    print(f"P95 tokens: {token_counts[int(n * 0.95)]}")
    print(f"P99 tokens: {token_counts[int(n * 0.99)]}")
    print()
    print(f"Current max_text_len: {CFG.max_text_len}")
    print(f"Apps exceeding limit: {len(long_apps)} ({100 * len(long_apps) / n:.1f}%)")

    if long_apps:
        print()
        print("TOP 10 LONGEST APPS (will be truncated):")
        print("-" * 60)
        long_apps.sort(key=lambda x: -x["tokens"])
        for app in long_apps[:10]:
            print(f"  {app['app_id'][:40]:<40} {app['tokens']:>5} tokens (+{app['truncated']})")

    # Distribution
    print()
    print("DISTRIBUTION:")
    print("-" * 60)
    buckets = [128, 256, 384, 512, 768, 1024, 2048]
    for i, b in enumerate(buckets):
        count = sum(1 for t in token_counts if t <= b)
        pct = 100 * count / n
        bar = "█" * int(pct / 2)
        print(f"  <= {b:4}: {count:4} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
