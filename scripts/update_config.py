#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — Config Update (Task 4)
=================================================
Run AFTER prereg_finalise.py to update config.json with:
  - v2.7 infrastructure (HuggingFace, FP16, ROCm)
  - Cross-precision verification results
  - Frequency-matched nouns
  - Model commit hashes

Usage:
    python scripts/update_config.py [--results-dir C:\\weber\\results\\prereg_finalisation]
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse


def find_latest_results(results_dir):
    """Find the most recent prereg_finalisation_*.json file."""
    candidates = sorted(results_dir.glob("prereg_finalisation_*.json"), reverse=True)
    if not candidates:
        print(f"ERROR: No prereg_finalisation_*.json found in {results_dir}")
        print("Run prereg_finalise.py first.")
        sys.exit(1)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path(r"C:\weber\results\prereg_finalisation"))
    parser.add_argument("--config", type=Path,
                        default=Path(r"C:\weber\config.json"))
    args = parser.parse_args()

    # Load existing config or start fresh
    if args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
        print(f"Loaded existing config from {args.config}")
    else:
        config = {}
        print("Creating new config.json")

    # Load finalisation results
    results_path = find_latest_results(args.results_dir)
    with open(results_path) as f:
        results = json.load(f)
    print(f"Loaded results from {results_path}")

    # Update config
    config.update({
        "project": "Classical Minds, Modern Machines — Project 4.2",
        "study": "Weber's Law in Transformer Magnitude Representations",
        "plan_version": "v2.7",
        "plan_file": "Weber_Project_Plan_v2_7.pdf",
        "updated": datetime.now().isoformat(),

        # Infrastructure (v2.7)
        "infrastructure": {
            "inference_engine": "HuggingFace transformers",
            "precision": "FP16 (torch.float16)",
            "gpu": results.get("gpu", "AMD Radeon RX 7900 GRE"),
            "pytorch_version": results.get("pytorch_version"),
            "compute_backend": "ROCm 6.0 (HSA_OVERRIDE_GFX_VERSION=11.0.0)",
            "python": "3.12.0",
            "note": "llama-cpp-python embed() returns static embeddings — confirmed in Phase 0. "
                    "HuggingFace with output_hidden_states=True is required for contextualised representations.",
        },

        # Models with commit hashes
        "models": {},

        # Cross-precision verification
        "cross_precision_verification": {
            "passed": results.get("cross_precision", {}).get("overall_pass"),
            "worst_pearson": results.get("cross_precision", {}).get("worst_pearson"),
            "worst_spearman": results.get("cross_precision", {}).get("worst_spearman"),
            "criterion": "> 0.99 Pearson and Spearman at layers 16–31",
            "magnitudes_tested": [1, 5, 10, 20, 50, 100, 200, 500, 700, 1000],
        },

        # Frequency-matched nouns
        "frequency_matched_nouns": {
            "passed": results.get("frequency_matched_nouns", {}).get("overall_pass"),
            "selected_nouns": results.get("frequency_matched_nouns", {}).get("selected_nouns"),
            "spearman_rho": results.get("frequency_matched_nouns", {}).get("spearman_rho"),
            "max_deviation": results.get("frequency_matched_nouns", {}).get("max_deviation"),
            "criterion": "Spearman > 0.85, max deviation < 1.0 log-units",
            "prompt": "The next word is",
        },

        # Stimuli
        "stimuli": {
            "directory": "stimuli/",
            "n_files": 18,
            "checksum_file": "stimuli/stimulus_checksums.json",
        },

        # Power analysis
        "power_analysis": {
            "results_file": "results/power_analysis/power_analysis_v3_results.json",
            "H1_power": "≥99%",
            "H2_power": "≥99%",
            "H3_power": "≥80% at |ρ| ≥ 0.56",
            "H7_power": "≥80% at d ≥ 3.0 (conditional)",
        },
    })

    # Populate model hashes
    model_hashes = results.get("model_commit_hashes", {})
    model_roles = {
        "meta-llama/Meta-Llama-3-8B-Instruct": {"role": "primary_1", "name": "llama3_instruct"},
        "mistralai/Mistral-7B-Instruct-v0.3": {"role": "primary_2", "name": "mistral_instruct"},
        "meta-llama/Meta-Llama-3-8B": {"role": "exploratory", "name": "llama3_base"},
    }
    for model_id, info in model_hashes.items():
        role_info = model_roles.get(model_id, {"role": "unknown", "name": model_id})
        config["models"][model_id] = {
            "role": role_info["role"],
            "short_name": role_info["name"],
            "commit_hash": info.get("commit_hash") if isinstance(info, dict) else None,
            "precision": "FP16",
        }

    # Save
    with open(args.config, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {args.config}")

    # Print summary
    print(f"\n  Plan version:     {config['plan_version']}")
    print(f"  Infrastructure:   {config['infrastructure']['inference_engine']}, "
          f"{config['infrastructure']['precision']}")
    print(f"  Cross-precision:  {'PASS' if config['cross_precision_verification']['passed'] else 'FAIL'}")
    print(f"  Freq nouns:       {'PASS' if config['frequency_matched_nouns']['passed'] else 'FAIL'}")
    print(f"  Models: {len(config['models'])}")
    for mid, info in config["models"].items():
        h = info.get("commit_hash", "?")
        print(f"    {info['role']:12s} {mid}: {h[:12] if h else '?'}...")


if __name__ == "__main__":
    main()
