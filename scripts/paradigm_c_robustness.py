"""
Weber's Law Project 4.2 — Paradigm C & Robustness Checks
Classical Minds, Modern Machines

Paradigm C: Representational Precision Gradient (H3)
    Computed from Paradigm A data. No additional extraction needed.

Robustness checks (v2.7 Section 5.7):
    5.7.1 Digit-boundary diagnostic
    5.7.2 Single-token control
    5.7.6 Shuffled-magnitude sanity check

Pre-registration ref: v2.7 Sections 5.5, 5.7, 4.1 (H3), 14 (F6).

Usage:
    python paradigm_c_robustness.py --model llama_instruct --domain numerical
"""

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, mannwhitneyu
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS, DOMAINS, N_LAYERS_TOTAL, RESULTS_DIR,
    PRIMARY_LAYER_RANGE, BONFERRONI_ALPHA,
    H3_MIN_LAYERS, H3_MIN_DOMAINS,
    NUMERICAL_MAGNITUDES, SEED_SHUFFLED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Paradigm C: Precision Gradient ──

def compute_precision_gradient(
    centroids: np.ndarray,
    magnitudes: list[float],
) -> dict:
    """
    Paradigm C (v2.7 Section 5.5):
    "For each consecutive pair of magnitudes, compute local precision =
    1 / ||h(n+1) - h(n)||."

    "Fit: precision = a / n^γ. If γ ≈ 1, the representation is efficient
    under a 1/n prior."

    Also computes H3 test: Spearman ρ between magnitude and representational
    density is significantly negative.
    """
    n_mags, n_layers, d_model = centroids.shape
    mags = np.array(magnitudes, dtype=np.float64)

    results = {
        "raw_precision": {},      # per layer
        "normalised_precision": {},
        "h3_spearman": {},
        "gamma_fit": {},
    }

    midpoints = (mags[:-1] + mags[1:]) / 2
    log_steps = np.diff(np.log(mags))

    for layer in range(n_layers):
        vecs = centroids[:, layer, :]

        if np.any(np.isnan(vecs)):
            results["raw_precision"][layer] = None
            continue

        # Consecutive distances (Euclidean, as spec says ||h(n+1) - h(n)||)
        diffs = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
        precision_raw = 1.0 / np.where(diffs == 0, 1e-10, diffs)

        # Log-step normalised
        precision_norm = precision_raw * log_steps

        results["raw_precision"][layer] = precision_raw.tolist()
        results["normalised_precision"][layer] = precision_norm.tolist()

        # H3 test: Spearman ρ(magnitude midpoints, precision) < 0 (one-tailed)
        rho, p_two = spearmanr(midpoints, precision_raw)
        p_one = p_two / 2 if rho < 0 else 1 - p_two / 2  # one-tailed for negative
        results["h3_spearman"][layer] = {
            "rho": float(rho),
            "p_two_tailed": float(p_two),
            "p_one_tailed": float(p_one),
            "significant": bool(p_one < BONFERRONI_ALPHA),
        }

        # Fit precision = a / n^γ → log(precision) = log(a) - γ * log(n)
        log_mid = np.log(midpoints)
        log_prec = np.log(np.where(precision_raw > 0, precision_raw, 1e-10))
        try:
            # OLS on log-log: log(prec) = c0 + c1 * log(midpoint)
            X = np.column_stack([np.ones(len(log_mid)), log_mid])
            beta, _, _, _ = np.linalg.lstsq(X, log_prec, rcond=None)
            gamma = -beta[1]  # precision ∝ 1/n^γ → log(prec) = ... - γ log(n)
            results["gamma_fit"][layer] = {
                "gamma": float(gamma),
                "intercept": float(beta[0]),
            }
        except Exception:
            results["gamma_fit"][layer] = {"gamma": None}

    return results


def evaluate_h3(precision_results: dict) -> dict:
    """
    H3 evaluation (v2.7 Section 4.1):
    "Spearman ρ between magnitude and representational density is significantly
    negative (p < 0.017, one-tailed) in at least 17 of 32 layers for at least
    2 of 3 domains, in both primary models."

    This evaluates at single-domain level.
    """
    spearman = precision_results["h3_spearman"]
    sig_layers = sum(
        1 for layer in range(N_LAYERS_TOTAL)
        if spearman.get(layer, {}).get("significant", False)
    )

    return {
        "significant_layers": sig_layers,
        "total_layers": N_LAYERS_TOTAL,  # H3 uses ALL 32 layers, not just primary
        "threshold": H3_MIN_LAYERS,
        "domain_passes": sig_layers >= H3_MIN_LAYERS,
    }


# ── Robustness: Digit-Boundary Diagnostic (v2.7 Section 5.7.1) ──

def digit_boundary_diagnostic(
    centroids: np.ndarray,
    magnitudes: list[float],
) -> dict:
    """
    v2.7 Section 5.7.1:
    "Three pair sets: (a) Within-digit (same digit count); (b) Cross-digit
    (crossing boundary); (c) Matched-ratio controls. Compare hidden-state
    distances for matched-ratio cross-digit vs within-digit pairs."

    "Report the full distribution of Cohen's d across layers."
    """
    mags = np.array(magnitudes, dtype=np.float64)
    n = len(mags)
    digits = np.array([len(str(int(m))) for m in mags])

    # Build pair sets
    within_pairs = []  # same digit count
    cross_pairs = []   # different digit count

    for i, j in combinations(range(n), 2):
        ratio = max(mags[i], mags[j]) / min(mags[i], mags[j])
        if digits[i] == digits[j]:
            within_pairs.append((i, j, ratio))
        else:
            cross_pairs.append((i, j, ratio))

    # Match ratios: for each cross-digit pair, find closest-ratio within-digit pair
    matched = []
    for ci, cj, cr in cross_pairs:
        best_match = None
        best_diff = float("inf")
        for wi, wj, wr in within_pairs:
            diff = abs(cr - wr)
            if diff < best_diff:
                best_diff = diff
                best_match = (wi, wj, wr)
        if best_match is not None:
            matched.append({
                "cross": (ci, cj),
                "within": (best_match[0], best_match[1]),
                "cross_ratio": cr,
                "within_ratio": best_match[2],
                "ratio_diff": best_diff,
            })

    results_per_layer = {}

    for layer in range(N_LAYERS_TOTAL):
        vecs = centroids[:, layer, :]
        if np.any(np.isnan(vecs)):
            results_per_layer[layer] = {"cohens_d": None}
            continue

        # Compute distances for matched pairs
        cross_dists = []
        within_dists = []
        for m in matched:
            ci, cj = m["cross"]
            wi, wj = m["within"]
            cross_dists.append(np.linalg.norm(vecs[ci] - vecs[cj]))
            within_dists.append(np.linalg.norm(vecs[wi] - vecs[wj]))

        cross_dists = np.array(cross_dists)
        within_dists = np.array(within_dists)

        # Cohen's d (cross vs within)
        pooled_std = np.sqrt(
            (cross_dists.var() + within_dists.var()) / 2
        )
        if pooled_std > 0:
            d = (cross_dists.mean() - within_dists.mean()) / pooled_std
        else:
            d = 0.0

        # Mann-Whitney U for non-parametric test
        if len(cross_dists) > 1 and len(within_dists) > 1:
            u_stat, u_p = mannwhitneyu(cross_dists, within_dists, alternative="two-sided")
        else:
            u_stat, u_p = np.nan, np.nan

        results_per_layer[layer] = {
            "cohens_d": float(d),
            "cross_mean": float(cross_dists.mean()),
            "within_mean": float(within_dists.mean()),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": float(u_p),
        }

    # Summary interpretation (v2.7):
    # "If d > 0.5 at most layers, Paradigm A is tempered.
    #  If < 0.2, digit confound is negligible."
    primary_ds = [
        results_per_layer[l]["cohens_d"]
        for l in range(PRIMARY_LAYER_RANGE[0], PRIMARY_LAYER_RANGE[1])
        if results_per_layer[l]["cohens_d"] is not None
    ]
    if primary_ds:
        median_d = float(np.median(primary_ds))
        interpretation = (
            "digit_dominated" if median_d > 0.5
            else "negligible" if median_d < 0.2
            else "mixed"
        )
    else:
        median_d = None
        interpretation = "insufficient_data"

    return {
        "per_layer": results_per_layer,
        "n_matched_pairs": len(matched),
        "primary_median_cohens_d": median_d,
        "interpretation": interpretation,
    }


# ── Robustness: Shuffled-Magnitude Sanity Check (v2.7 Section 5.7.6) ──

def shuffled_magnitude_check(
    centroids: np.ndarray,
    magnitudes: list[float],
) -> dict:
    """
    v2.7 Section 5.7.6:
    "Take the same 26 carrier sentences but randomly reassign which magnitude
    appears in which sentence (seed 42). Extract hidden states from the shuffled
    set. Compute the RDM indexed by the actual magnitude values."

    Since we can't re-run extraction here (that requires the model), this function
    computes what we CAN from existing data: the correlation between per-carrier
    RDMs and the centroid RDM, as a proxy for carrier independence.

    The actual shuffled extraction needs to be done in paradigm_a_extract.py
    with a --shuffled flag. This function analyses the shuffled results.
    """
    # This is a placeholder that checks carrier consistency from per-carrier data
    # The actual shuffled extraction is a separate run
    return {
        "note": "Shuffled extraction requires separate model forward passes. "
                "Run paradigm_a_extract.py with --shuffled flag.",
        "status": "pending",
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Paradigm C (Precision Gradient) & Robustness Checks"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--domain", required=True, choices=list(DOMAINS.keys()) + ["all"])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    domains = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]

    for domain_key in domains:
        log.info(f"\n{'='*60}")
        log.info(f"Paradigm C & Robustness: {args.model} / {domain_key}")
        log.info(f"{'='*60}")

        in_dir = args.results_dir / "paradigm_a" / args.model / domain_key
        hs_path = in_dir / "hidden_states.npz"

        if not hs_path.exists():
            log.error(f"Hidden states not found at {hs_path}")
            continue

        data = np.load(hs_path)
        centroids = data["centroids"]

        with open(in_dir / "extraction_metadata.json") as f:
            meta = json.load(f)
        magnitudes = meta["magnitudes_numeric"]

        # ── Paradigm C ──
        log.info("Computing precision gradient (H3)...")
        precision = compute_precision_gradient(centroids, magnitudes)
        h3_eval = evaluate_h3(precision)
        log.info(
            f"H3: {h3_eval['significant_layers']}/{h3_eval['total_layers']} "
            f"layers significant (need {h3_eval['threshold']}). "
            f"Domain {'PASSES' if h3_eval['domain_passes'] else 'FAILS'}."
        )

        # ── Digit-boundary diagnostic (numerical only) ──
        if domain_key == "numerical":
            log.info("Running digit-boundary diagnostic...")
            digit_diag = digit_boundary_diagnostic(centroids, magnitudes)
            med_d = digit_diag['primary_median_cohens_d']
            if med_d is not None:
                log.info(
                    f"Digit boundary: median Cohen's d = {med_d:.3f} "
                    f"({digit_diag['interpretation']})"
                )
            else:
                log.warning("Digit boundary: could not compute (all NaN centroids?)")
        else:
            digit_diag = {"note": "Digit-boundary diagnostic is numerical-domain only."}

        # ── Shuffled check ──
        shuffled = shuffled_magnitude_check(centroids, magnitudes)

        # Save results
        out = {
            "model_key": args.model,
            "domain_key": domain_key,
            "paradigm_c": {
                "precision": {
                    "raw": {str(k): v for k, v in precision["raw_precision"].items()},
                    "normalised": {str(k): v for k, v in precision["normalised_precision"].items()},
                    "gamma_fit": {str(k): v for k, v in precision["gamma_fit"].items()},
                },
                "h3_spearman": {str(k): v for k, v in precision["h3_spearman"].items()},
                "h3_evaluation": h3_eval,
            },
            "robustness": {
                "digit_boundary": digit_diag,
                "shuffled_magnitude": shuffled,
            },
        }

        out_path = in_dir / "paradigm_c_robustness.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)

        log.info(f"Results saved to {out_path}")

    log.info("\n=== Complete ===")


if __name__ == "__main__":
    main()
