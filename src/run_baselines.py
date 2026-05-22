#!/usr/bin/env python3
"""run_baselines.py — Runs Qwen, E2E, and optional API baselines (GPT-4o-mini, GPT-4o)."""
import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils.runner import run_step, print_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--skip-qwen",  action="store_true")
    parser.add_argument("--skip-e2e",   action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args()

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    results = {}

    if not args.skip_local:
        if not args.skip_qwen:
            results["Qwen2.5-7B"] = run_step(
                "Qwen2.5-7B — description-only zero-shot",
                "steps.baselines.baseline_qwen",
            )
        if not args.skip_e2e:
            results["E2E transformer"] = run_step(
                "E2E fine-tuned multimodal transformer",
                "steps.baselines.baseline_e2e_transformer",
            )

    if args.all:
        if has_openai:
            results["MLLM zero-shot"] = run_step(
                "MLLM zero-shot — GPT-4o-mini",
                "steps.baselines.baseline_mllm_zeroshot",
            )
        else:
            print("\n[SKIP] MLLM zero-shot — no OPENAI_API_KEY")

        if has_openai:
            results["GPT-4o 6-shot"] = run_step(
                "GPT-4o 6-shot few-shot",
                "steps.baselines.baseline_mllm_fewshot",
            )
        else:
            print("\n[SKIP] GPT-4o 6-shot — no OPENAI_API_KEY")
    else:
        print("\n[SKIP] API baselines — pass --all to enable")

    if not args.no_latency:
        results["Latency benchmark"] = run_step(
            "Latency benchmark (Table 16)",
            "steps.latency_benchmark",
        )

    print_summary(results)


if __name__ == "__main__":
    main()
