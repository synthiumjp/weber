"""
M3 100-Boundary Counterbalanced Identification
===============================================
Runs counterbalanced A/B identification for decade_100 and control_150.

Uses the same run_counterbalanced_trial() and load_model_and_tokenizer()
from m3_rerun_identification, but with 100-boundary framings.

Usage:
  python m3_run_identification_100.py
  python m3_run_identification_100.py --model llama3-8b-instruct

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from m3_rerun_identification import (
    ExtractionConfig,
    load_model_and_tokenizer,
    run_counterbalanced_trial,
)

EXTRACTION_DIR = Path("extractions")
CONDITIONS = ["decade_100", "control_150"]

MODELS = [
    {"short": "llama3-8b-instruct", "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
     "precision": "float16", "has_chat_template": True},
    {"short": "mistral-7b-instruct", "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
     "precision": "float16", "has_chat_template": True},
    {"short": "gemma2-9b-it", "hf_id": "google/gemma-2-9b-it",
     "precision": "bfloat16", "has_chat_template": True},
    {"short": "qwen25-7b-instruct", "hf_id": "Qwen/Qwen2.5-7B-Instruct",
     "precision": "float16", "has_chat_template": True},
    {"short": "phi35-mini-instruct", "hf_id": "Lexius/Phi-3.5-mini-instruct",
     "precision": "float16", "has_chat_template": True},
    # Base model skipped — no chat template
]

# 100-boundary identification framings
FRAMINGS_100 = {
    "two_three_digit": {
        "question": "Does the number {N} have two digits or three digits?",
        "category_a_label": "two digits",
        "category_b_label": "three digits",
    },
    "tens_hundreds": {
        "question": "Is {N} in the tens or in the hundreds?",
        "category_a_label": "tens",
        "category_b_label": "hundreds",
    },
    "less_more_100": {
        "question": "Is {N} less than one hundred or one hundred or more?",
        "category_a_label": "less than one hundred",
        "category_b_label": "one hundred or more",
    },
}

# control_150 framing (just small/large — no meaningful boundary)
FRAMINGS_150 = {
    "small_large": {
        "question": "Is {N} a small number or a large number?",
        "category_a_label": "small",
        "category_b_label": "large",
    },
}


def run_model(model_info):
    short = model_info["short"]
    hf_id = model_info["hf_id"]
    precision = model_info["precision"]

    config = ExtractionConfig()
    config.model_name = hf_id
    config.model_short = short
    config.precision = precision

    print(f"\n  Loading: {hf_id}")
    model, tokenizer = load_model_and_tokenizer(config)

    # Verify chat template
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception as e:
        # Try without system role (Gemma)
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            print(f"  [SKIP] Chat template failed: {e}")
            del model
            torch.cuda.empty_cache()
            return

    for condition in CONDITIONS:
        meta_path = EXTRACTION_DIR / f"m3_meta_{condition}_{short}.json"
        if not meta_path.exists():
            print(f"  [SKIP] No metadata: {meta_path}")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        values = meta["values"]
        framings = FRAMINGS_100 if condition == "decade_100" else FRAMINGS_150

        print(f"\n  {'='*55}")
        print(f"  {condition} — {len(values)} values × {len(framings)} framings")
        print(f"  {'='*55}")

        results = []
        t0 = time.time()

        for framing_key, spec in framings.items():
            print(f"\n  Framing: {framing_key}")
            print(f"  {'Value':>6s} | {'O1 P(B)':>8s} | {'O2 P(B)':>8s} | "
                  f"{'Avg P(B)':>8s} | {'Bias':>7s} | Category")
            print(f"  {'-'*68}")

            for v in values:
                question = spec["question"].format(N=str(v))
                trial = run_counterbalanced_trial(
                    model, tokenizer, question,
                    spec["category_a_label"], spec["category_b_label"],
                    config.device,
                )

                choice = (spec["category_b_label"]
                          if trial["prob_category_b"] > 0.5
                          else spec["category_a_label"])

                print(f"  {v:6d} | {trial['order1_p_catb']:.4f}   | "
                      f"{trial['order2_p_catb']:.4f}   | "
                      f"{trial['prob_category_b']:.4f}   | "
                      f"{trial['position_bias']:+.4f}  | {choice}")

                results.append({
                    "value": v,
                    "framing": framing_key,
                    "prob_category_a": trial["prob_category_a"],
                    "prob_category_b": trial["prob_category_b"],
                    "order1_p_catb": trial["order1_p_catb"],
                    "order2_p_catb": trial["order2_p_catb"],
                    "position_bias": trial["position_bias"],
                    "raw_logit_a": trial["order1_logit_a"],
                    "raw_logit_b": trial["order1_logit_b"],
                    "option_a_label": spec["category_a_label"],
                    "option_b_label": spec["category_b_label"],
                })

        elapsed = time.time() - t0

        # Update metadata
        meta["identification_results_100"] = results
        meta["identification_method_100"] = "chat_template_ab_counterbalanced"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  Updated: {meta_path} ({elapsed:.1f}s)")

    del model
    torch.cuda.empty_cache()
    print(f"  GPU cleared.")


def main():
    parser = argparse.ArgumentParser(description="M3 100-Boundary Identification")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    if args.model:
        targets = [m for m in MODELS if m["short"] == args.model]
        if not targets:
            print(f"Model '{args.model}' not found")
            sys.exit(1)
    else:
        targets = MODELS

    print(f"M3 100-Boundary Identification")
    print(f"Models: {len(targets)}")
    print(f"Conditions: {CONDITIONS}")
    print(f"{'='*60}")

    for i, m in enumerate(targets):
        print(f"\n{'#'*60}")
        print(f"  MODEL {i+1}/{len(targets)}: {m['short']}")
        print(f"{'#'*60}")
        t0 = time.time()
        run_model(m)
        print(f"\n  Total: {time.time() - t0:.1f}s")

    print(f"\n{'='*60}")
    print("Done. Identification results saved to metadata JSONs.")


if __name__ == "__main__":
    main()
