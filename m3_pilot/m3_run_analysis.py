"""
M3 Multi-Model Analysis Runner
===============================
Runs m3_pilot_analysis.py for all extracted models.
Handles per-model output directories and layer range scaling.

Usage:
  python m3_run_analysis.py              # All models
  python m3_run_analysis.py --model mistral-7b-instruct  # Single model
  python m3_run_analysis.py --list       # Show available models

Author: JP Cacioli
Programme: Classical Minds, Modern Machines
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

# Add current directory to path so we can import the analysis module
sys.path.insert(0, str(Path(__file__).parent))

import m3_pilot_analysis as analysis


# ─────────────────────────────────────────────────────────
# Model registry — must match extraction output tags
# ─────────────────────────────────────────────────────────

MODELS = [
    {
        "short": "llama3-8b-instruct",
        "n_layers": 33,  # 32 transformer + 1 embedding
        "primary_range": (8, 25),
        "role": "primary (pilot)",
    },
    {
        "short": "mistral-7b-instruct",
        "n_layers": 33,
        "primary_range": (8, 25),
        "role": "primary",
    },
    {
        "short": "gemma2-9b-it",
        "n_layers": 43,  # 42 transformer + 1 embedding
        "primary_range": (11, 34),  # Scale: ~25-80% of transformer layers
        "role": "primary",
    },
    {
        "short": "qwen25-7b-instruct",
        "n_layers": 29,  # 28 transformer + 1 embedding
        "primary_range": (7, 22),
        "role": "primary",
    },
    {
        "short": "phi35-mini-instruct",
        "n_layers": 33,
        "primary_range": (8, 25),
        "role": "primary",
    },
    {
        "short": "llama3-8b-base",
        "n_layers": 33,
        "primary_range": (8, 25),
        "role": "exploratory",
    },
]

EXTRACTION_DIR = Path("extractions")


def find_available_models():
    """Check which models have extraction outputs."""
    available = []
    for m in MODELS:
        npz = EXTRACTION_DIR / f"m3_centroids_decade_10_{m['short']}.npz"
        if npz.exists():
            available.append(m)
    return available


def run_analysis_for_model(model_info, n_permutations=10_000):
    """Run the full analysis pipeline for a single model."""
    short = model_info["short"]

    # Per-model output directory
    output_dir = Path("results") / short
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure
    config = analysis.AnalysisConfig()
    config.model_short = short
    config.primary_layer_range = model_info["primary_range"]
    config.output_dir = output_dir
    config.n_permutations = n_permutations

    print("=" * 70)
    print(f"M3 Analysis — {short}")
    print(f"  Role: {model_info['role']}")
    print(f"  Layers: {model_info['n_layers']} (primary: {model_info['primary_range']})")
    print(f"  Permutations: {n_permutations}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    all_results = {}

    for condition in config.conditions:
        print(f"\n{'='*60}")
        print(f"Analysing condition: {condition}")
        print(f"{'='*60}")

        try:
            centroid_data = analysis.load_centroids(config, condition)
            stim_data = analysis.load_stimuli(config, condition)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping {condition} for {short}")
            continue

        rsa_centroids = centroid_data["rsa_centroids"]
        values = centroid_data["values"]
        boundary = stim_data["metadata"]["boundary"]
        meta = centroid_data["meta"]

        print(f"  Values: {values}")
        print(f"  Boundary: {boundary}")
        print(f"  Centroid shape: {rsa_centroids.shape}")

        # Step 1: Empirical RDMs
        print("\n  Step 1: Computing empirical RDMs (cosine)")
        empirical_rdms = analysis.compute_rdms_all_layers(rsa_centroids, metric="cosine")
        print(f"  RDM shape: {empirical_rdms.shape}")

        # Plot RDM at representative primary layer (middle of primary range)
        rep_layer = (model_info["primary_range"][0] + model_info["primary_range"][1]) // 2
        if rep_layer < empirical_rdms.shape[0]:
            analysis.plot_rdm_heatmap(
                empirical_rdms[rep_layer], values,
                f"Empirical RDM — {condition} (Layer {rep_layer}, Cosine) [{short}]",
                output_dir / f"rdm_heatmap_{condition}_layer{rep_layer}.png",
            )
            print(f"  RDM heatmap saved (layer {rep_layer})")

        # Step 2: Theoretical RDMs
        print("\n  Step 2: Building theoretical RDMs")
        theoretical_rdms = analysis.build_theoretical_rdms(values, boundary)
        for name, rdm in theoretical_rdms.items():
            print(f"    {name}: shape={rdm.shape}, "
                  f"range=[{rdm.min():.3f}, {rdm.max():.3f}]")

        # Step 3: RSA
        print(f"\n  Step 3: RSA with Mantel tests ({config.n_permutations} permutations)")
        rsa_results = analysis.rsa_all_layers(
            empirical_rdms, theoretical_rdms,
            n_permutations=config.n_permutations,
            seed=config.seed,
        )

        l_start, l_end = config.primary_layer_range
        print(f"\n  RSA Summary (primary layers {l_start}-{l_end-1}):")
        for name in ["continuous", "cp_additive", "categorical", "linear"]:
            if name in rsa_results:
                primary_rhos = rsa_results[name]["rho"][l_start:l_end]
                primary_ps = rsa_results[name]["p"][l_start:l_end]
                print(f"    {name:15s}: mean ρ = {np.mean(primary_rhos):.4f}, "
                      f"max ρ = {np.max(primary_rhos):.4f}, "
                      f"min p = {np.min(primary_ps):.4f}")

        analysis.plot_rsa_comparison(
            rsa_results, condition,
            output_dir / f"rsa_comparison_{condition}.png",
            config.primary_layer_range,
        )
        print("  RSA comparison plot saved")

        # Step 4: Precision gradient
        print("\n  Step 4: Precision gradient")
        precision_data = analysis.compute_precision_gradient(
            rsa_centroids, values
        )
        analysis.plot_precision_gradient(
            precision_data, values, boundary, condition,
            output_dir / f"precision_gradient_{condition}.png",
            primary_layers=config.primary_layer_range,
        )
        print("  Precision gradient plot saved")

        # Step 5: Identification
        print("\n  Step 5: Identification analysis")
        id_results = meta.get("identification_results", [])
        if id_results:
            id_analysis = analysis.analyse_identification(meta)
            analysis.plot_identification_function(
                id_analysis, condition, boundary,
                output_dir / f"identification_{condition}.png",
            )
            print("  Identification plot saved")
        else:
            id_analysis = {"note": "No identification data"}
            print("  No identification data available")

        all_results[condition] = {
            "rsa_results": rsa_results,
            "precision_data": precision_data,
            "identification": id_analysis,
            "values": values.tolist(),
            "boundary": boundary,
        }

    # Go/No-Go (if both conditions present)
    if "decade_10" in all_results and "control_15" in all_results:
        print(f"\n{'='*60}")
        print("GO/NO-GO DECISION")
        print(f"{'='*60}")

        decision = analysis.go_nogo_analysis(
            rsa_results_decade=all_results["decade_10"]["rsa_results"],
            rsa_results_control=all_results["control_15"]["rsa_results"],
            precision_decade={
                "distances": np.array(all_results["decade_10"]["precision_data"]["distances"]),
                "midpoints": np.array(all_results["decade_10"]["precision_data"]["midpoints"]),
            },
            precision_control={
                "distances": np.array(all_results["control_15"]["precision_data"]["distances"]),
                "midpoints": np.array(all_results["control_15"]["precision_data"]["midpoints"]),
            },
            values_decade=np.array(all_results["decade_10"]["values"]),
            values_control=np.array(all_results["control_15"]["values"]),
            primary_layers=config.primary_layer_range,
        )

        print(f"\n{decision['summary']}")

        analysis.plot_go_nogo_summary(
            all_results["decade_10"]["rsa_results"],
            all_results["control_15"]["rsa_results"],
            {
                "distances": np.array(all_results["decade_10"]["precision_data"]["distances"]),
                "midpoints": np.array(all_results["decade_10"]["precision_data"]["midpoints"]),
            },
            {
                "distances": np.array(all_results["control_15"]["precision_data"]["distances"]),
                "midpoints": np.array(all_results["control_15"]["precision_data"]["midpoints"]),
            },
            np.array(all_results["decade_10"]["values"]),
            np.array(all_results["control_15"]["values"]),
            decision,
            output_dir / "go_nogo_summary.png",
            config.primary_layer_range,
        )
        print("  Go/no-go summary figure saved")
        all_results["go_nogo_decision"] = decision

    # Save results JSON
    def make_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serialisable(v) for v in obj]
        return obj

    results_path = output_dir / f"m3_results_{short}.json"
    with open(results_path, "w") as f:
        json.dump(make_serialisable(all_results), f, indent=2)
    print(f"\n  Full results saved: {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="M3 Multi-Model Analysis Runner")
    parser.add_argument("--model", type=str, default=None,
                        help="Run analysis for a single model (by output tag)")
    parser.add_argument("--list", action="store_true",
                        help="List available models with extraction data")
    parser.add_argument("--permutations", type=int, default=10_000,
                        help="Number of Mantel permutations (default: 10000)")
    args = parser.parse_args()

    available = find_available_models()

    if args.list:
        print("Available models with extraction data:")
        for m in available:
            print(f"  {m['short']:<25} {m['role']}")
        missing = [m for m in MODELS if m not in available]
        if missing:
            print("\nMissing extraction data:")
            for m in missing:
                print(f"  {m['short']:<25} {m['role']}")
        return

    if args.model:
        targets = [m for m in available if m["short"] == args.model]
        if not targets:
            print(f"ERROR: No extraction data for '{args.model}'")
            print(f"Available: {', '.join(m['short'] for m in available)}")
            sys.exit(1)
    else:
        targets = available

    print("=" * 70)
    print("M3 MULTI-MODEL ANALYSIS")
    print(f"Models: {len(targets)}")
    print(f"Permutations: {args.permutations}")
    print("=" * 70)

    all_model_results = {}
    for i, model_info in enumerate(targets):
        print(f"\n\n{'#'*70}")
        print(f"MODEL {i+1}/{len(targets)}: {model_info['short']}")
        print(f"{'#'*70}")

        t0 = time.time()
        results = run_analysis_for_model(model_info, n_permutations=args.permutations)
        elapsed = time.time() - t0

        go_decision = results.get("go_nogo_decision", {})
        all_model_results[model_info["short"]] = {
            "go_nogo": go_decision.get("decision", "N/A") if isinstance(go_decision, dict) else "N/A",
            "elapsed": elapsed,
        }

        print(f"\n  Completed in {elapsed:.1f}s")

    # Cross-model summary
    print(f"\n\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Go/No-Go':<12} {'Time':<10}")
    print(f"{'-'*25} {'-'*12} {'-'*10}")
    for short, info in all_model_results.items():
        print(f"{short:<25} {info['go_nogo']:<12} {info['elapsed']:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
