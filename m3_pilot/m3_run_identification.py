"""
M3 Multi-Model Counterbalanced Identification
===============================================
Runs m3_rerun_identification.py logic for all extracted models.
Each model gets chat-templated A/B forced-choice with counterbalancing.

Note: Base model (no chat template) will be skipped with a warning.

Usage:
  python m3_run_identification.py              # All models
  python m3_run_identification.py --model mistral-7b-instruct
  python m3_run_identification.py --list

Author: JP Cacioli
Programme: Classical Minds, Modern Machines
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from m3_extract import ExtractionConfig, load_model_and_tokenizer
from m3_rerun_identification import (
    IDENTIFICATION_FRAMINGS,
    run_counterbalanced_trial,
)

# ─────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────

MODELS = [
    {
        "short": "llama3-8b-instruct",
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "precision": "float16",
        "has_chat_template": True,
        "role": "primary (pilot)",
    },
    {
        "short": "mistral-7b-instruct",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "precision": "float16",
        "has_chat_template": True,
        "role": "primary",
    },
    {
        "short": "gemma2-9b-it",
        "hf_id": "google/gemma-2-9b-it",
        "precision": "bfloat16",
        "has_chat_template": True,
        "role": "primary",
    },
    {
        "short": "qwen25-7b-instruct",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "precision": "float16",
        "has_chat_template": True,
        "role": "primary",
    },
    {
        "short": "phi35-mini-instruct",
        "hf_id": "Lexius/Phi-3.5-mini-instruct",
        "precision": "float16",
        "has_chat_template": True,
        "role": "primary",
    },
    {
        "short": "llama3-8b-base",
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "precision": "float16",
        "has_chat_template": False,  # No chat template — skip identification
        "role": "exploratory",
    },
]

CONDITIONS = ["decade_10", "control_15"]
EXTRACTION_DIR = Path("extractions")


def find_available_models():
    """Check which models have extraction metadata."""
    available = []
    for m in MODELS:
        meta = EXTRACTION_DIR / f"m3_meta_decade_10_{m['short']}.json"
        if meta.exists():
            available.append(m)
    return available


def run_identification_for_model(model_info):
    """Run counterbalanced identification for one model across both conditions."""
    short = model_info["short"]
    hf_id = model_info["hf_id"]
    precision = model_info["precision"]

    if not model_info["has_chat_template"]:
        print(f"  [SKIP] {short} — no chat template (base model)")
        print(f"  Raw-logit identification already in extraction metadata.")
        return {"skipped": True, "reason": "no_chat_template"}

    # Build config
    config = ExtractionConfig()
    config.model_name = hf_id
    config.model_short = short
    config.precision = precision

    print(f"\n  Loading model: {hf_id}")
    model, tokenizer = load_model_and_tokenizer(config)

    # Verify chat template exists
    try:
        test = tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception as e:
        print(f"  [SKIP] Chat template failed: {e}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"skipped": True, "reason": str(e)}

    all_results = {}

    for condition in CONDITIONS:
        print(f"\n  {'='*55}")
        print(f"  Condition: {condition}")
        print(f"  {'='*55}")

        meta_path = EXTRACTION_DIR / f"m3_meta_{condition}_{short}.json"
        if not meta_path.exists():
            print(f"  [SKIP] No metadata: {meta_path}")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        values = meta["values"]
        n_framings = len(IDENTIFICATION_FRAMINGS)
        n_total = len(values) * n_framings
        print(f"  Values: {values}")
        print(f"  Trials: {n_total} × 2 orders = {n_total * 2} forward passes")

        results = []
        t0 = time.time()

        for framing_key, spec in IDENTIFICATION_FRAMINGS.items():
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

        # Update metadata in place
        meta["identification_results"] = results
        meta["identification_method"] = "chat_template_ab_counterbalanced"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  Updated: {meta_path} ({elapsed:.1f}s)")

        all_results[condition] = {
            "n_trials": len(results),
            "elapsed": elapsed,
        }

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  GPU memory cleared.")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="M3 Multi-Model Counterbalanced Identification"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Single model by output tag")
    parser.add_argument("--list", action="store_true",
                        help="List available models")
    args = parser.parse_args()

    # Set ROCm env
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    available = find_available_models()

    if args.list:
        print("Available models:")
        for m in available:
            chat = "chat" if m["has_chat_template"] else "NO CHAT"
            print(f"  {m['short']:<25} {m['role']:<20} [{chat}]")
        return

    if args.model:
        targets = [m for m in available if m["short"] == args.model]
        if not targets:
            print(f"ERROR: '{args.model}' not found")
            print(f"Available: {', '.join(m['short'] for m in available)}")
            sys.exit(1)
    else:
        targets = available

    print("=" * 70)
    print("M3 COUNTERBALANCED IDENTIFICATION — All Models")
    print(f"Models: {len(targets)}")
    print(f"Framings: {list(IDENTIFICATION_FRAMINGS.keys())}")
    print("=" * 70)

    summary = {}
    for i, model_info in enumerate(targets):
        print(f"\n\n{'#'*70}")
        print(f"MODEL {i+1}/{len(targets)}: {model_info['short']}")
        print(f"{'#'*70}")

        t0 = time.time()
        results = run_identification_for_model(model_info)
        elapsed = time.time() - t0

        summary[model_info["short"]] = {
            "elapsed": elapsed,
            "skipped": results.get("skipped", False),
        }
        print(f"\n  Total: {elapsed:.1f}s")

    # Summary
    print(f"\n\n{'='*70}")
    print("IDENTIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Status':<12} {'Time':<10}")
    print(f"{'-'*25} {'-'*12} {'-'*10}")
    for short, info in summary.items():
        status = "SKIPPED" if info["skipped"] else "OK"
        print(f"{short:<25} {status:<12} {info['elapsed']:.1f}s")
    print(f"{'='*70}")
    print("\nRerun m3_run_analysis.py to regenerate figures with updated identification data.")


if __name__ == "__main__":
    main()
