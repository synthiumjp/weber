"""
Weber's Law Project 4.2 — Paradigm A Master Runner
Classical Minds, Modern Machines

Orchestrates the full Paradigm A pipeline:
    1. Extract hidden states (paradigm_a_extract.py)
    2. Analyse: pairwise distances, model fitting, RSA (paradigm_a_analyse.py)
    3. Precision gradient & robustness checks (paradigm_c_robustness.py)
    4. Generate figures (paradigm_a_figures.py)
    5. Print summary report

Usage (recommended start — numerical domain, Llama only):
    python run_paradigm_a.py --model llama_instruct --domain numerical

Full run (all domains, all models):
    python run_paradigm_a.py --model llama_instruct --domain all --include-nouns
    python run_paradigm_a.py --model mistral_instruct --domain all
    python run_paradigm_a.py --model llama_base --domain numerical  # E1 exploratory
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, DOMAINS, RESULTS_DIR, PRIMARY_LAYER_RANGE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_step(script: str, args: list[str], step_name: str) -> bool:
    """Run a pipeline step as a subprocess."""
    cmd = [sys.executable, script] + args
    log.info(f"\n{'─'*60}")
    log.info(f"STEP: {step_name}")
    log.info(f"CMD: {' '.join(cmd)}")
    log.info(f"{'─'*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error(f"FAILED: {step_name} (exit code {result.returncode})")
        return False

    log.info(f"DONE: {step_name} ({elapsed:.1f}s)")
    return True


def print_summary(model_key: str, domains: list[str], results_dir: Path):
    """Print a summary of all Paradigm A results."""
    log.info(f"\n{'='*60}")
    log.info(f"PARADIGM A SUMMARY: {model_key}")
    log.info(f"{'='*60}")

    for domain in domains:
        analysis_path = results_dir / "paradigm_a" / model_key / domain / "paradigm_a_analysis.json"
        if not analysis_path.exists():
            log.warning(f"  {domain}: No analysis results found")
            continue

        with open(analysis_path) as f:
            analysis = json.load(f)

        log.info(f"\n--- {domain.upper()} ---")

        # ICC
        icc = analysis.get("icc_per_layer", [])
        if icc:
            primary_icc = [
                icc[l] for l in range(PRIMARY_LAYER_RANGE[0], min(PRIMARY_LAYER_RANGE[1], len(icc)))
                if icc[l] is not None and not (isinstance(icc[l], float) and icc[l] != icc[l])
            ]
            if primary_icc:
                log.info(f"  ICC (carriers): mean={np.mean(primary_icc):.3f}, "
                         f"min={np.min(primary_icc):.3f} (primary layers)")

        # H1
        h1 = analysis.get("h1_evaluation", {})
        for metric in ["cosine", "euclidean"]:
            h1m = h1.get(metric, {})
            if h1m:
                log.info(
                    f"  H1 ({metric}): {h1m['layers_passing']}/{h1m['layers_tested']} "
                    f"layers pass → {'PASS' if h1m['domain_passes'] else 'FAIL'}"
                )

        # Best model at each primary layer
        for metric in ["cosine"]:  # Just show cosine (RSA default)
            best_counts = {"linear": 0, "weber": 0, "stevens": 0}
            for layer in range(PRIMARY_LAYER_RANGE[0], PRIMARY_LAYER_RANGE[1]):
                layer_key = f"layer_{layer:02d}"
                fits = (analysis.get("layers", {})
                        .get(layer_key, {})
                        .get(metric, {})
                        .get("model_fits", {}))
                best = fits.get("best_aic", "unknown")
                if best in best_counts:
                    best_counts[best] += 1
            log.info(
                f"  Best AIC ({metric}): Weber={best_counts['weber']}, "
                f"Linear={best_counts['linear']}, Stevens={best_counts['stevens']} layers"
            )

        # Model-fit floor
        floor = analysis.get("model_fit_floor_check", {})
        for metric in ["cosine", "euclidean"]:
            fm = floor.get(metric, {})
            if fm.get("triggers_e4"):
                log.warning(f"  ⚠ MODEL-FIT FLOOR TRIGGERED ({metric}): E4 required")

        # Paradigm C / H3
        pc_path = results_dir / "paradigm_a" / model_key / domain / "paradigm_c_robustness.json"
        if pc_path.exists():
            with open(pc_path) as f:
                pc = json.load(f)
            h3 = pc.get("paradigm_c", {}).get("h3_evaluation", {})
            if h3:
                log.info(
                    f"  H3 (precision): {h3['significant_layers']}/{h3['total_layers']} "
                    f"layers sig → {'PASS' if h3['domain_passes'] else 'FAIL'}"
                )

            # Digit boundary
            if domain == "numerical":
                db = pc.get("robustness", {}).get("digit_boundary", {})
                if "primary_median_cohens_d" in db and db["primary_median_cohens_d"] is not None:
                    log.info(
                        f"  Digit boundary: median d={db['primary_median_cohens_d']:.3f} "
                        f"({db['interpretation']})"
                    )

    log.info(f"\n{'='*60}")
    log.info("Results directory: " + str(results_dir / "paradigm_a" / model_key))
    log.info("Figures directory: " + str(results_dir / "figures" / model_key))
    log.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Paradigm A: Master Runner")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--domain", default="numerical",
                        choices=list(DOMAINS.keys()) + ["all"])
    parser.add_argument("--include-nouns", action="store_true",
                        help="Also extract frequency-matched nouns (Llama only)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction (use existing hidden states)")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    domains = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]

    log.info(f"{'='*60}")
    log.info(f"PARADIGM A PIPELINE")
    log.info(f"Model: {args.model} ({MODELS[args.model]['hf_id']})")
    log.info(f"Domains: {domains}")
    log.info(f"Results: {args.results_dir}")
    log.info(f"{'='*60}")

    t_start = time.time()
    success = True

    # Step 1: Extraction
    if not args.skip_extract:
        extract_args = [
            "--model", args.model,
            "--domain", args.domain,
            "--results-dir", str(args.results_dir),
        ]
        if args.include_nouns:
            extract_args.append("--include-nouns")

        if not run_step(
            str(scripts_dir / "paradigm_a_extract.py"),
            extract_args,
            "Hidden State Extraction",
        ):
            log.error("Extraction failed. Aborting pipeline.")
            sys.exit(1)

    # Step 2: Analysis
    for domain in domains:
        if not run_step(
            str(scripts_dir / "paradigm_a_analyse.py"),
            ["--model", args.model, "--domain", domain,
             "--results-dir", str(args.results_dir)],
            f"Analysis ({domain})",
        ):
            log.error(f"Analysis failed for {domain}")
            success = False

    # Step 3: Paradigm C & Robustness
    for domain in domains:
        if not run_step(
            str(scripts_dir / "paradigm_c_robustness.py"),
            ["--model", args.model, "--domain", domain,
             "--results-dir", str(args.results_dir)],
            f"Paradigm C & Robustness ({domain})",
        ):
            log.warning(f"Paradigm C failed for {domain}")

    # Step 4: Figures
    if not args.skip_figures:
        if not run_step(
            str(scripts_dir / "paradigm_a_figures.py"),
            ["--model", args.model, "--domain", args.domain,
             "--results-dir", str(args.results_dir)],
            "Figure Generation",
        ):
            log.warning("Figure generation failed (non-fatal)")

    # Step 5: Summary
    print_summary(args.model, domains, args.results_dir)

    elapsed = time.time() - t_start
    log.info(f"\nTotal pipeline time: {elapsed:.1f}s")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
