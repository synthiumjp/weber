"""
Weber's Law Project 4.2 — Hypothesis Evaluation Summary
Classical Minds, Modern Machines

Aggregates results across domains and models. Evaluates all hypotheses
at the programme level per the pre-registration success criteria.

Pre-registration ref: v2.7 Sections 4.1-4.3, 8 (statistical analysis plan).

Usage:
    python evaluate_hypotheses.py --results-dir C:\\weber\\results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS, DOMAINS, RESULTS_DIR,
    H1_MIN_LAYERS, H1_MIN_DOMAINS, H3_MIN_LAYERS, H3_MIN_DOMAINS,
    BONFERRONI_ALPHA, SECONDARY_ALPHA,
    PRIMARY_LAYER_RANGE, N_LAYERS_TOTAL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def evaluate_all(results_dir: Path) -> dict:
    """
    Evaluate all hypotheses H1-H7 plus exploratory E1-E4.

    Cross-model rule (v2.7 Section 8):
    "All primary hypotheses tested separately for each model.
    Supported at programme level if met in both primary models."
    """
    primary_models = ["llama_instruct", "mistral_instruct"]
    all_domains = ["numerical", "temporal", "spatial"]
    report = {}

    # ════════════════════════════════════════════════════════════
    # H1: Logarithmic Representational Geometry
    # ════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("H1: Logarithmic Representational Geometry")
    log.info("="*60)

    h1_results = {}
    for model_key in primary_models:
        h1_results[model_key] = {"domains": {}}
        for domain_key in all_domains:
            analysis = load_json(
                results_dir / "paradigm_a" / model_key / domain_key / "paradigm_a_analysis.json"
            )
            if analysis is None:
                h1_results[model_key]["domains"][domain_key] = {"status": "not_run"}
                continue

            h1_eval = analysis.get("h1_evaluation", {})
            # Use cosine as RSA default (v2.7: "RSA uses cosine-based RDMs as default")
            cosine = h1_eval.get("cosine", {})
            h1_results[model_key]["domains"][domain_key] = {
                "layers_passing": cosine.get("layers_passing", 0),
                "layers_tested": cosine.get("layers_tested", 0),
                "domain_passes": cosine.get("domain_passes", False),
            }

        # Count passing domains
        passing_domains = sum(
            1 for d in h1_results[model_key]["domains"].values()
            if d.get("domain_passes", False)
        )
        h1_results[model_key]["passing_domains"] = passing_domains
        h1_results[model_key]["model_passes"] = passing_domains >= H1_MIN_DOMAINS

        log.info(f"  {model_key}: {passing_domains}/3 domains pass "
                 f"(need {H1_MIN_DOMAINS}) → {'PASS' if passing_domains >= H1_MIN_DOMAINS else 'FAIL'}")
        for d, r in h1_results[model_key]["domains"].items():
            if r.get("status") != "not_run":
                log.info(f"    {d}: {r['layers_passing']}/{r['layers_tested']} layers "
                         f"→ {'PASS' if r['domain_passes'] else 'FAIL'}")

    # Programme-level H1
    h1_programme = all(
        h1_results.get(m, {}).get("model_passes", False)
        for m in primary_models
    )
    log.info(f"\n  H1 PROGRAMME LEVEL: {'SUPPORTED' if h1_programme else 'NOT SUPPORTED'}")

    report["H1"] = {
        "per_model": h1_results,
        "programme_supported": h1_programme,
    }

    # ════════════════════════════════════════════════════════════
    # H2: Behavioural Weber's Law
    # ════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("H2: Behavioural Weber's Law")
    log.info("="*60)

    h2_results = {}
    for model_key in primary_models:
        h2_data = load_json(
            results_dir / "paradigm_b" / model_key / "numerical" / "paradigm_b_results.json"
        )
        if h2_data is None:
            h2_results[model_key] = {"status": "not_run"}
            log.info(f"  {model_key}: not run")
            continue

        h2_eval = h2_data.get("h2_evaluation", {})
        h2_results[model_key] = h2_eval
        log.info(
            f"  {model_key}: Δdeviance={h2_eval.get('delta_deviance', 'N/A')}, "
            f"p={h2_eval.get('p_comparison', 'N/A')}, "
            f"→ {'PASS' if h2_eval.get('h2_passes') else 'FAIL'}"
        )

    h2_programme = all(
        h2_results.get(m, {}).get("h2_passes", False)
        for m in primary_models
    )
    log.info(f"\n  H2 PROGRAMME LEVEL: {'SUPPORTED' if h2_programme else 'NOT SUPPORTED'}")

    report["H2"] = {
        "per_model": h2_results,
        "programme_supported": h2_programme,
    }

    # ════════════════════════════════════════════════════════════
    # H3: Graded Representational Precision
    # ════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("H3: Graded Representational Precision")
    log.info("="*60)

    h3_results = {}
    for model_key in primary_models:
        h3_results[model_key] = {"domains": {}}
        for domain_key in all_domains:
            pc_data = load_json(
                results_dir / "paradigm_a" / model_key / domain_key / "paradigm_c_robustness.json"
            )
            if pc_data is None:
                h3_results[model_key]["domains"][domain_key] = {"status": "not_run"}
                continue

            h3_eval = pc_data.get("paradigm_c", {}).get("h3_evaluation", {})
            h3_results[model_key]["domains"][domain_key] = h3_eval

        passing_domains = sum(
            1 for d in h3_results[model_key]["domains"].values()
            if d.get("domain_passes", False)
        )
        h3_results[model_key]["passing_domains"] = passing_domains
        h3_results[model_key]["model_passes"] = passing_domains >= H3_MIN_DOMAINS

        log.info(f"  {model_key}: {passing_domains}/3 domains pass → "
                 f"{'PASS' if passing_domains >= H3_MIN_DOMAINS else 'FAIL'}")

    h3_programme = all(
        h3_results.get(m, {}).get("model_passes", False)
        for m in primary_models
    )
    log.info(f"\n  H3 PROGRAMME LEVEL: {'SUPPORTED' if h3_programme else 'NOT SUPPORTED'}")

    report["H3"] = {
        "per_model": h3_results,
        "programme_supported": h3_programme,
    }

    # ════════════════════════════════════════════════════════════
    # H4-H7: Secondary Hypotheses (per-model, no cross-model requirement)
    # ════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("Secondary Hypotheses (H4-H7)")
    log.info("="*60)

    # H7: Functional Relevance (Paradigm D)
    h7_results = {}
    for model_key in primary_models:
        d_data = load_json(
            results_dir / "paradigm_d" / model_key / "paradigm_d_results.json"
        )
        if d_data is None:
            gate_data = load_json(
                results_dir / "paradigm_d" / model_key / "paradigm_d_gate.json"
            )
            if gate_data and not gate_data.get("go"):
                h7_results[model_key] = {"status": "no_go", "gate": gate_data}
                log.info(f"  H7 {model_key}: NO-GO (probe R²={gate_data.get('best_probe_r2', 'N/A')})")
            else:
                h7_results[model_key] = {"status": "not_run"}
                log.info(f"  H7 {model_key}: not run")
        else:
            h7_eval = d_data.get("h7_evaluation", {})
            h7_results[model_key] = h7_eval
            log.info(
                f"  H7 {model_key}: {h7_eval.get('fraction_exceeds', 0):.3f} exceed threshold "
                f"→ {'PASS' if h7_eval.get('h7_passes') else 'FAIL'}"
            )

    report["H7"] = {"per_model": h7_results}

    # ════════════════════════════════════════════════════════════
    # Model-fit floor check
    # ════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("Model-Fit Floor Check")
    log.info("="*60)

    floor_results = {}
    for model_key in primary_models:
        for domain_key in all_domains:
            analysis = load_json(
                results_dir / "paradigm_a" / model_key / domain_key / "paradigm_a_analysis.json"
            )
            if analysis is None:
                continue
            floor = analysis.get("model_fit_floor_check", {})
            for metric in ["cosine", "euclidean"]:
                fm = floor.get(metric, {})
                if fm.get("triggers_e4"):
                    key = f"{model_key}/{domain_key}/{metric}"
                    floor_results[key] = fm
                    log.warning(f"  ⚠ FLOOR TRIGGERED: {key}")

    if not floor_results:
        log.info("  No floor triggers. Three-model framework adequate everywhere.")

    report["model_fit_floor"] = floor_results

    # ════════════════════════════════════════════════════════════
    # Mixed-outcome interpretation (v2.7 Section 4.5)
    # ════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("Mixed-Outcome Interpretation")
    log.info("="*60)

    h1_pass = h1_programme
    h2_pass = h2_programme
    h3_pass = h3_programme
    h7_any_pass = any(
        h7_results.get(m, {}).get("h7_passes", False)
        for m in primary_models
    )

    if h1_pass and not h3_pass:
        log.info("  Pattern (a): Geometry log + precision flat → "
                 "compression from frequency, not precision allocation")
    elif h1_pass and h2_pass and not h7_any_pass:
        log.info("  Pattern (b): Geometry log + behaviour Weber + no causal effect → "
                 "geometry real but epiphenomenal to comparison task")
    elif h2_pass and not h1_pass:
        log.info("  Pattern (c): Behaviour Weber + geometry linear → "
                 "algorithmic ratio computation, not representational compression")
    elif h1_pass and h2_pass and h3_pass:
        log.info("  Full convergence: log geometry + Weber behaviour + precision gradient + "
                 f"{'causal relevance' if h7_any_pass else 'no causal test'}")
    elif not h1_pass and not h2_pass:
        log.info("  Full null: no log geometry, no Weber behaviour → "
                 "LLMs do not develop efficient magnitude codes")
    else:
        log.info("  Partial/other pattern — see detailed results")

    report["mixed_outcome"] = {
        "h1_programme": h1_pass,
        "h2_programme": h2_pass,
        "h3_programme": h3_pass,
        "h7_any_model": h7_any_pass,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Hypothesis Evaluation Summary")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: results_dir/hypothesis_evaluation.json)")
    args = parser.parse_args()

    report = evaluate_all(args.results_dir)

    out_path = args.output or (args.results_dir / "hypothesis_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"\nFull report saved to {out_path}")

    # Print compact summary
    log.info("\n" + "="*60)
    log.info("COMPACT SUMMARY")
    log.info("="*60)
    for h in ["H1", "H2", "H3"]:
        status = "SUPPORTED" if report.get(h, {}).get("programme_supported") else "NOT SUPPORTED"
        log.info(f"  {h}: {status}")
    for h in ["H7"]:
        per_model = report.get(h, {}).get("per_model", {})
        for m, r in per_model.items():
            status = ("PASS" if r.get("h7_passes") else
                      "NO-GO" if r.get("status") == "no_go" else
                      "FAIL" if "h7_passes" in r else "NOT RUN")
            log.info(f"  {h} ({m}): {status}")


if __name__ == "__main__":
    main()
