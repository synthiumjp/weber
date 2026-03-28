"""
M3 E10 Analysis — Nonce Remapping RSA
======================================
Paper M3, "Classical Minds, Modern Machines" programme.
Author: JP Cacioli
Research assistant: Claude (Anthropic)

Runs RSA analysis on E10 nonce remapping extractions.
Loads pre-computed ordinal-position theoretical RDMs from the stimulus
file (NOT from build_theoretical_rdms, which assumes log-magnitude).

Two conditions: nonce_no_order, nonce_ordered.
6 models × 2 conditions = 12 analysis cells.

Usage:
  python m3_run_nonce_analysis.py
  python m3_run_nonce_analysis.py --models llama3-8b-instruct
  python m3_run_nonce_analysis.py --conditions nonce_ordered
"""

import argparse
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Import analysis functions from the pilot analysis module
sys.path.insert(0, str(Path(__file__).parent))
from m3_pilot_analysis import (
    compute_rdms_all_layers,
    rsa_all_layers,
    compute_precision_gradient,
    plot_rdm_heatmap,
    plot_rsa_comparison,
    plot_precision_gradient,
)

SEED = 42

# Model registry (same as m3_batch_run.py / m3_run_analysis.py)
MODEL_REGISTRY = {
    "llama3-8b-instruct": {
        "primary_layers": (8, 25),
        "n_layers_total": 33,
    },
    "mistral-7b-instruct": {
        "primary_layers": (8, 25),
        "n_layers_total": 33,
    },
    "gemma2-9b-it": {
        "primary_layers": (11, 34),
        "n_layers_total": 43,
    },
    "qwen25-7b-instruct": {
        "primary_layers": (7, 22),
        "n_layers_total": 29,
    },
    "phi35-mini-instruct": {
        "primary_layers": (8, 25),
        "n_layers_total": 33,
    },
    "llama3-8b-base": {
        "primary_layers": (8, 25),
        "n_layers_total": 33,
    },
}

CONDITIONS = ["nonce_no_order", "nonce_ordered"]


def load_nonce_centroids(extraction_dir, condition, model_short):
    """Load RSA centroids for a nonce condition."""
    npz_path = extraction_dir / f"m3_centroids_{condition}_{model_short}.npz"
    meta_path = extraction_dir / f"m3_meta_{condition}_{model_short}.json"

    if not npz_path.exists():
        return None, None
    if not meta_path.exists():
        return None, None

    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return {
        "rsa_centroids": data["rsa_centroids"],
        "values": data["values"],
        "meta": meta,
    }, meta


def load_nonce_theoretical_rdms(stimulus_dir, condition):
    """Load pre-computed ordinal-position theoretical RDMs from stimulus file.

    CRITICAL: Do NOT use build_theoretical_rdms() — that assumes log-magnitude.
    The nonce stimulus file contains ordinal-position RDMs.
    """
    stim_path = stimulus_dir / f"m3_stimuli_{condition}.json"
    with open(stim_path) as f:
        stim_data = json.load(f)

    theoretical_rdms = {}
    for name, rdm_list in stim_data["theoretical_rdms"].items():
        theoretical_rdms[name] = np.array(rdm_list)

    boundary = stim_data["metadata"]["boundary"]
    values = stim_data["probing_values"]

    return theoretical_rdms, boundary, values


def analyse_one_cell(
    extraction_dir, stimulus_dir, output_dir,
    model_short, condition, n_permutations=10_000,
):
    """Run RSA analysis for one model × condition cell."""

    print(f"\n  Loading {model_short} / {condition}...")
    centroid_data, meta = load_nonce_centroids(
        extraction_dir, condition, model_short
    )
    if centroid_data is None:
        print(f"  SKIP: No extraction found for {model_short} / {condition}")
        return None

    rsa_centroids = centroid_data["rsa_centroids"]
    values = centroid_data["values"]

    # Load pre-computed ordinal theoretical RDMs
    theoretical_rdms, boundary, stim_values = load_nonce_theoretical_rdms(
        stimulus_dir, condition
    )

    print(f"  Centroid shape: {rsa_centroids.shape}")
    print(f"  Values: {values}")
    print(f"  Boundary: rank {boundary}")
    print(f"  RDM basis: ordinal position")

    # Get primary layer range for this model
    model_info = MODEL_REGISTRY.get(model_short, {})
    primary_layers = model_info.get("primary_layers", (8, 25))

    # Step 1: Compute empirical RDMs
    empirical_rdms = compute_rdms_all_layers(rsa_centroids, metric="cosine")
    print(f"  Empirical RDMs: {empirical_rdms.shape}")

    # Create output directory for this model
    model_dir = output_dir / model_short
    model_dir.mkdir(parents=True, exist_ok=True)

    # Plot RDM at representative primary layer
    rep_layer = (primary_layers[0] + primary_layers[1]) // 2
    if rep_layer < empirical_rdms.shape[0]:
        plot_rdm_heatmap(
            empirical_rdms[rep_layer], values,
            f"Empirical RDM — {condition} (Layer {rep_layer}, Cosine)\n{model_short}",
            model_dir / f"rdm_heatmap_{condition}_layer{rep_layer}.png",
        )

    # Step 2: RSA with Mantel tests (using pre-computed ordinal RDMs)
    print(f"  RSA with {n_permutations} Mantel permutations...")
    t0 = time.time()
    rsa_results = rsa_all_layers(
        empirical_rdms, theoretical_rdms,
        n_permutations=n_permutations,
        seed=SEED,
    )
    elapsed = time.time() - t0
    print(f"  RSA complete ({elapsed:.1f}s)")

    # Summary at primary layers
    l_start, l_end = primary_layers
    print(f"\n  RSA Summary (primary layers {l_start}-{l_end-1}):")
    cp_advantage_layers = 0
    n_primary = l_end - l_start
    for name in ["continuous", "cp_additive", "categorical", "linear"]:
        if name in rsa_results:
            primary_rhos = rsa_results[name]["rho"][l_start:l_end]
            primary_ps = rsa_results[name]["p"][l_start:l_end]
            print(f"    {name:15s}: mean ρ = {np.mean(primary_rhos):.4f}, "
                  f"max ρ = {np.max(primary_rhos):.4f}, "
                  f"min p = {np.min(primary_ps):.4f}")

    # Count layers where CP-Additive > Continuous
    if "continuous" in rsa_results and "cp_additive" in rsa_results:
        cont_rhos = rsa_results["continuous"]["rho"][l_start:l_end]
        cp_rhos = rsa_results["cp_additive"]["rho"][l_start:l_end]
        cp_advantage_layers = sum(
            1 for c, cp in zip(cont_rhos, cp_rhos) if cp > c
        )
        cp_advantage = np.mean(
            [cp - c for c, cp in zip(cont_rhos, cp_rhos)]
        )
        print(f"\n  CP-Additive > Continuous: {cp_advantage_layers}/{n_primary} "
              f"primary layers")
        print(f"  Mean CP advantage: {cp_advantage:+.4f}")

    # Plot RSA comparison
    plot_rsa_comparison(
        rsa_results, f"{condition} ({model_short})",
        model_dir / f"rsa_comparison_{condition}.png",
        primary_layers,
    )

    # Step 3: Precision gradient
    precision_data = compute_precision_gradient(rsa_centroids, values, metric="cosine")
    plot_precision_gradient(
        precision_data, values, boundary, f"{condition} ({model_short})",
        model_dir / f"precision_gradient_{condition}.png",
        primary_layers,
    )

    # Compute boundary distance ratio (E8 analogue)
    boundary_idx = None
    for i, v in enumerate(values):
        if v == boundary and i > 0:
            boundary_idx = i - 1  # Index into distances array
            break
    # If boundary value is at position 0 of values, boundary_idx stays None

    boundary_ratio = None
    if boundary_idx is not None:
        primary_dists = precision_data["distances"][l_start:l_end, :]
        mean_dists = np.mean(primary_dists, axis=0)
        boundary_dist = mean_dists[boundary_idx]
        non_boundary_dists = np.concatenate([
            mean_dists[:boundary_idx],
            mean_dists[boundary_idx + 1:]
        ])
        if len(non_boundary_dists) > 0 and np.mean(non_boundary_dists) > 0:
            boundary_ratio = float(boundary_dist / np.mean(non_boundary_dists))

    # Assemble results
    result = {
        "model": model_short,
        "condition": condition,
        "boundary": int(boundary),
        "rdm_basis": "ordinal_position",
        "n_values": len(values),
        "n_layers": int(empirical_rdms.shape[0]),
        "primary_layers": list(primary_layers),
        "rsa_results": rsa_results,
        "cp_advantage_layers": f"{cp_advantage_layers}/{n_primary}",
        "cp_advantage_mean": float(cp_advantage) if "continuous" in rsa_results else None,
        "boundary_distance_ratio": boundary_ratio,
    }

    # Save per-model results
    result_path = model_dir / f"m3_nonce_analysis_{condition}_{model_short}.json"

    def _serialise(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(v) for v in obj]
        return obj

    with open(result_path, "w") as f:
        json.dump(_serialise(result), f, indent=2)
    print(f"  Results saved: {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="M3 E10 Nonce Remapping Analysis",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model short names (default: all 6)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Conditions (default: both nonce_no_order, nonce_ordered)",
    )
    parser.add_argument(
        "--permutations", type=int, default=10_000,
        help="Number of Mantel permutations (default: 10000)",
    )
    parser.add_argument(
        "--extraction-dir", type=str, default="extractions",
    )
    parser.add_argument(
        "--stimulus-dir", type=str, default="stimuli",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
    )
    args = parser.parse_args()

    models = args.models or list(MODEL_REGISTRY.keys())
    conditions = args.conditions or CONDITIONS
    extraction_dir = Path(args.extraction_dir)
    stimulus_dir = Path(args.stimulus_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("M3 E10 — Nonce Remapping RSA Analysis")
    print(f"Models: {models}")
    print(f"Conditions: {conditions}")
    print(f"Permutations: {args.permutations}")
    print("=" * 70)

    all_results = []
    t0_total = time.time()

    for condition in conditions:
        # Verify stimulus file exists
        stim_path = stimulus_dir / f"m3_stimuli_{condition}.json"
        if not stim_path.exists():
            print(f"\nERROR: Stimulus file not found: {stim_path}")
            print("Run m3_stimuli_nonce.py first.")
            continue

        for model_short in models:
            result = analyse_one_cell(
                extraction_dir, stimulus_dir, output_dir,
                model_short, condition,
                n_permutations=args.permutations,
            )
            if result is not None:
                all_results.append(result)

    # Cross-model summary
    if all_results:
        print(f"\n{'='*70}")
        print("CROSS-MODEL SUMMARY — E10 Nonce Remapping")
        print(f"{'='*70}")
        print(f"\n{'Model':<25} {'Condition':<18} {'CP>Cont':>8} "
              f"{'CP Adv':>8} {'Bnd Ratio':>10}")
        print("-" * 75)

        for r in all_results:
            bnd_ratio = f"{r['boundary_distance_ratio']:.3f}" if r.get(
                "boundary_distance_ratio") else "N/A"
            cp_adv = f"{r['cp_advantage_mean']:+.4f}" if r.get(
                "cp_advantage_mean") is not None else "N/A"
            print(f"{r['model']:<25} {r['condition']:<18} "
                  f"{r['cp_advantage_layers']:>8} {cp_adv:>8} "
                  f"{bnd_ratio:>10}")

        # Save summary
        summary_path = output_dir / "nonce_cross_model_summary.json"

        def _serialise(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: _serialise(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialise(v) for v in obj]
            return obj

        summary_data = []
        for r in all_results:
            summary_data.append({
                "model": r["model"],
                "condition": r["condition"],
                "cp_advantage_layers": r["cp_advantage_layers"],
                "cp_advantage_mean": r.get("cp_advantage_mean"),
                "boundary_distance_ratio": r.get("boundary_distance_ratio"),
            })

        with open(summary_path, "w") as f:
            json.dump(_serialise(summary_data), f, indent=2)
        print(f"\nSummary saved: {summary_path}")

    elapsed_total = time.time() - t0_total
    print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
