"""
Weber's Law Project 4.2 — Paradigm A: Analysis Pipeline
Classical Minds, Modern Machines

Step 3: Compute pairwise distances (Euclidean + cosine).
Step 4: Fit geometric models (Linear, Weber, Stevens).
Step 5: Representational Similarity Analysis (RSA with Mantel test).

Pre-registration ref: v2.7 Sections 5.3 Steps 3-5, Section 8.

Usage:
    python paradigm_a_analyse.py --model llama_instruct --domain numerical
    python paradigm_a_analyse.py --model llama_instruct --domain all --run-all
"""

import argparse
import json
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS, DOMAINS, N_LAYERS_TOTAL, RESULTS_DIR,
    PRIMARY_LAYER_RANGE, MANTEL_PERMUTATIONS,
    STEVENS_BETA_INIT, STEVENS_BETA_BOUNDS,
    BONFERRONI_ALPHA, BOOTSTRAP_SEED,
    MODEL_FIT_FLOOR_R2, H1_MIN_LAYERS, H1_MIN_DOMAINS,
    NUMERICAL_MAGNITUDES, FREQUENCY_MATCHED_NOUNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress curve_fit convergence warnings (we handle them explicitly)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")


# ── Theoretical RDMs (v2.7 Section 5.3 Step 5) ──

def build_theoretical_rdms(magnitudes: list[float]) -> dict[str, np.ndarray]:
    """
    Construct three theoretical RDMs, z-scored (v2.7: "Theoretical RDMs are
    z-scored (mean-centred, unit-variance) before computing Spearman correlations").

    Returns dict with condensed distance vectors (upper triangle).
    """
    mags = np.array(magnitudes, dtype=np.float64)
    n = len(mags)

    # Linear: |ni - nj|
    linear_rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            linear_rdm[i, j] = abs(mags[i] - mags[j])
            linear_rdm[j, i] = linear_rdm[i, j]

    # Weber/Log: |log(ni) - log(nj)|
    log_mags = np.log(mags)
    weber_rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            weber_rdm[i, j] = abs(log_mags[i] - log_mags[j])
            weber_rdm[j, i] = weber_rdm[i, j]

    # Stevens/Power: |ni^β - nj^β| — we use β from the data, but for the
    # theoretical RDM we need a fixed β. Use β=0.5 as the canonical Stevens
    # form. The actual Stevens model fitting (Step 4) estimates β from data.
    # For RSA, we compare against the theoretical form with β=0.5.
    stevens_rdm = np.zeros((n, n))
    pow_mags = np.power(mags, 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            stevens_rdm[i, j] = abs(pow_mags[i] - pow_mags[j])
            stevens_rdm[j, i] = stevens_rdm[i, j]

    def zscore_condensed(rdm_square):
        """Z-score the upper triangle (condensed form)."""
        v = squareform(rdm_square)
        if v.std() == 0:
            return v
        return (v - v.mean()) / v.std()

    return {
        "linear": zscore_condensed(linear_rdm),
        "weber": zscore_condensed(weber_rdm),
        "stevens": zscore_condensed(stevens_rdm),
    }


# ── Pairwise distance computation (v2.7 Section 5.3 Step 3) ──

def compute_pairwise_distances(centroids: np.ndarray) -> dict:
    """
    Compute pairwise Euclidean and cosine distances for all magnitude pairs.

    Input: (n_mags, n_layers, d_model)
    Output: dict with 'euclidean' and 'cosine', each (n_layers, n_pairs)
    """
    n_mags, n_layers, d_model = centroids.shape
    n_pairs = n_mags * (n_mags - 1) // 2

    euclidean = np.zeros((n_layers, n_pairs))
    cosine = np.zeros((n_layers, n_pairs))

    for layer in range(n_layers):
        vecs = centroids[:, layer, :]
        if np.any(np.isnan(vecs)):
            euclidean[layer] = np.nan
            cosine[layer] = np.nan
            continue
        euclidean[layer] = pdist(vecs, metric="euclidean")
        cosine[layer] = pdist(vecs, metric="cosine")

    return {"euclidean": euclidean, "cosine": cosine}


# ── Geometric model fitting (v2.7 Section 5.3 Step 4) ──

def fit_geometric_models(
    distances: np.ndarray,
    magnitudes: list[float],
    layer: int,
    metric_name: str,
) -> dict:
    """
    Fit Linear, Weber, and Stevens models to pairwise distances at one layer.

    Models (v2.7):
        Linear: d = a + b * |n1 - n2|
        Weber:  d = a + b * |log(n1) - log(n2)|
        Stevens: d = a + b * |n1^β - n2^β|  (β estimated by NLS)

    Compare via R², AIC (primary) / BIC (robustness).
    """
    mags = np.array(magnitudes, dtype=np.float64)
    n = len(mags)
    n_pairs = n * (n - 1) // 2
    y = distances  # condensed distance vector

    if np.any(np.isnan(y)):
        return {"status": "nan_in_distances"}

    # Build predictor vectors
    pairs = list(combinations(range(n), 2))

    # Linear predictor: |n1 - n2|
    x_linear = np.array([abs(mags[i] - mags[j]) for i, j in pairs])

    # Weber predictor: |log(n1) - log(n2)|
    log_mags = np.log(mags)
    x_weber = np.array([abs(log_mags[i] - log_mags[j]) for i, j in pairs])

    results = {}

    # ── Linear model (OLS) ──
    X_lin = np.column_stack([np.ones(n_pairs), x_linear])
    try:
        beta_lin, residuals_lin, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
        y_pred_lin = X_lin @ beta_lin
        ss_res_lin = np.sum((y - y_pred_lin) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0
        k_lin = 2  # a, b
        aic_lin = n_pairs * np.log(ss_res_lin / n_pairs) + 2 * k_lin
        bic_lin = n_pairs * np.log(ss_res_lin / n_pairs) + k_lin * np.log(n_pairs)
        results["linear"] = {
            "r2": float(r2_lin),
            "aic": float(aic_lin),
            "bic": float(bic_lin),
            "params": {"a": float(beta_lin[0]), "b": float(beta_lin[1])},
            "ss_res": float(ss_res_lin),
        }
    except Exception as e:
        results["linear"] = {"status": "error", "error": str(e)}

    # ── Weber model (OLS) ──
    X_web = np.column_stack([np.ones(n_pairs), x_weber])
    try:
        beta_web, _, _, _ = np.linalg.lstsq(X_web, y, rcond=None)
        y_pred_web = X_web @ beta_web
        ss_res_web = np.sum((y - y_pred_web) ** 2)
        r2_web = 1 - ss_res_web / ss_tot if ss_tot > 0 else 0
        k_web = 2
        aic_web = n_pairs * np.log(ss_res_web / n_pairs) + 2 * k_web
        bic_web = n_pairs * np.log(ss_res_web / n_pairs) + k_web * np.log(n_pairs)
        results["weber"] = {
            "r2": float(r2_web),
            "aic": float(aic_web),
            "bic": float(bic_web),
            "params": {"a": float(beta_web[0]), "b": float(beta_web[1])},
            "ss_res": float(ss_res_web),
        }
    except Exception as e:
        results["weber"] = {"status": "error", "error": str(e)}

    # ── Stevens model (NLS) ──
    # d = a + b * |n1^β - n2^β|
    # v2.7: init β0 = 0.5, bounds [0.01, 2.0]
    def stevens_model(pair_indices, a, b, beta):
        x = np.array([
            abs(mags[int(i)] ** beta - mags[int(j)] ** beta)
            for i, j in pair_indices
        ])
        return a + b * x

    pair_arr = np.array(pairs, dtype=np.float64)
    try:
        popt, pcov = curve_fit(
            lambda idx, a, b, beta: stevens_model(idx, a, b, beta),
            pair_arr,
            y,
            p0=[y.mean(), 1.0, STEVENS_BETA_INIT],
            bounds=([-np.inf, -np.inf, STEVENS_BETA_BOUNDS[0]],
                    [np.inf, np.inf, STEVENS_BETA_BOUNDS[1]]),
            maxfev=10000,
        )
        a_s, b_s, beta_s = popt
        y_pred_stev = stevens_model(pair_arr, a_s, b_s, beta_s)
        ss_res_stev = np.sum((y - y_pred_stev) ** 2)
        r2_stev = 1 - ss_res_stev / ss_tot if ss_tot > 0 else 0
        k_stev = 3  # a, b, β
        aic_stev = n_pairs * np.log(ss_res_stev / n_pairs) + 2 * k_stev
        bic_stev = n_pairs * np.log(ss_res_stev / n_pairs) + k_stev * np.log(n_pairs)
        results["stevens"] = {
            "r2": float(r2_stev),
            "aic": float(aic_stev),
            "bic": float(bic_stev),
            "params": {"a": float(a_s), "b": float(b_s), "beta": float(beta_s)},
            "ss_res": float(ss_res_stev),
        }
    except Exception as e:
        results["stevens"] = {"status": "error", "error": str(e)}

    # ── Model comparison ──
    valid_models = {k: v for k, v in results.items() if "aic" in v}
    if valid_models:
        best_aic = min(valid_models, key=lambda k: valid_models[k]["aic"])
        best_bic = min(valid_models, key=lambda k: valid_models[k]["bic"])
        best_r2 = max(valid_models, key=lambda k: valid_models[k]["r2"])
        results["best_aic"] = best_aic
        results["best_bic"] = best_bic
        results["best_r2"] = best_r2
        results["best_r2_value"] = valid_models[best_r2]["r2"]

        # AIC vs BIC disagreement (v2.7 Section 7: "AIC primary")
        results["aic_bic_agree"] = (best_aic == best_bic)

    results["layer"] = layer
    results["metric"] = metric_name

    return results


# ── RSA with Mantel test (v2.7 Section 5.3 Step 5) ──

def mantel_test(
    model_rdm: np.ndarray,
    theoretical_rdm: np.ndarray,
    n_permutations: int = MANTEL_PERMUTATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Mantel permutation test (Kriegeskorte et al., 2008).

    Computes Spearman rank correlation between model RDM and theoretical RDM.
    Permutation null: shuffle magnitude labels and recompute correlation.

    v2.7: "Mantel permutation test (10,000 permutations of magnitude labels),
    which properly accounts for non-independence of pairwise distances."
    """
    # Both inputs are condensed distance vectors (upper triangle)
    rho_obs, _ = spearmanr(model_rdm, theoretical_rdm)

    # Permutation test: we need to permute the rows/columns of the square RDM
    # Convert condensed to square, permute, convert back
    n_items = int(0.5 * (1 + np.sqrt(1 + 8 * len(model_rdm))))
    model_square = squareform(model_rdm)

    rng = np.random.default_rng(seed)
    null_rhos = np.zeros(n_permutations)

    for p in range(n_permutations):
        perm = rng.permutation(n_items)
        perm_square = model_square[np.ix_(perm, perm)]
        perm_condensed = squareform(perm_square)
        null_rhos[p], _ = spearmanr(perm_condensed, theoretical_rdm)

    p_value = np.mean(null_rhos >= rho_obs)

    return {
        "rho": float(rho_obs),
        "p_value": float(p_value),
        "null_mean": float(null_rhos.mean()),
        "null_std": float(null_rhos.std()),
        "null_95": float(np.percentile(null_rhos, 95)),
        "n_permutations": n_permutations,
    }


def run_rsa(
    pairwise_distances: np.ndarray,
    magnitudes: list[float],
    layer: int,
    metric_name: str,
) -> dict:
    """
    Run full RSA analysis at one layer.

    v2.7: "Compute Spearman rank correlation between the model RDM and each
    theoretical RDM. Test significance with a Mantel permutation test."
    """
    model_rdm = pairwise_distances  # condensed vector

    # Build z-scored theoretical RDMs
    theoretical = build_theoretical_rdms(magnitudes)

    results = {"layer": layer, "metric": metric_name}

    for model_name, theo_rdm in theoretical.items():
        mantel_result = mantel_test(model_rdm, theo_rdm)
        results[model_name] = mantel_result

    # Compare Weber vs Linear (H1 primary comparison)
    rho_weber = results["weber"]["rho"]
    rho_linear = results["linear"]["rho"]
    results["weber_minus_linear_rho"] = float(rho_weber - rho_linear)

    # H1 test at this layer: Weber > Linear (Mantel p < alpha) AND AIC Weber < AIC Linear
    # (The AIC comparison comes from the model fitting step)

    return results


# ── Full analysis pipeline ──

def analyse_domain(
    model_key: str,
    domain_key: str,
    results_dir: Path,
) -> dict:
    """
    Run the complete Paradigm A analysis for one model × one domain.

    Steps 3-5 from v2.7 Section 5.3.
    """
    in_dir = results_dir / "paradigm_a" / model_key / domain_key
    hs_path = in_dir / "hidden_states.npz"

    if not hs_path.exists():
        raise FileNotFoundError(
            f"Hidden states not found at {hs_path}. Run paradigm_a_extract.py first."
        )

    # Load data
    data = np.load(hs_path)
    centroids = data["centroids"]  # (n_mags, n_layers, d_model)
    icc = data["icc_per_layer"]

    with open(in_dir / "extraction_metadata.json") as f:
        metadata = json.load(f)
    magnitudes = metadata["magnitudes_numeric"]

    log.info(
        f"Loaded {domain_key}: {centroids.shape[0]} magnitudes, "
        f"{centroids.shape[1]} layers, {centroids.shape[2]} dims"
    )

    # Step 3: Pairwise distances
    log.info("Computing pairwise distances (Euclidean + cosine)...")
    distances = compute_pairwise_distances(centroids)

    # Steps 4 & 5 for each layer and metric
    all_results = {
        "model_key": model_key,
        "domain_key": domain_key,
        "magnitudes": magnitudes,
        "icc_per_layer": icc.tolist(),
        "layers": {},
    }

    for metric_name in ["cosine", "euclidean"]:
        log.info(f"\nAnalysing {metric_name} distances...")
        all_results[f"pairwise_{metric_name}"] = distances[metric_name].tolist()

        for layer in range(N_LAYERS_TOTAL):
            layer_key = f"layer_{layer:02d}"
            if layer_key not in all_results["layers"]:
                all_results["layers"][layer_key] = {}

            dist_vec = distances[metric_name][layer]
            if np.any(np.isnan(dist_vec)):
                log.warning(f"  Layer {layer} has NaN distances, skipping")
                all_results["layers"][layer_key][metric_name] = {"status": "nan"}
                continue

            # Step 4: Model fitting
            fit_results = fit_geometric_models(dist_vec, magnitudes, layer, metric_name)

            # Step 5: RSA
            rsa_results = run_rsa(dist_vec, magnitudes, layer, metric_name)

            all_results["layers"][layer_key][metric_name] = {
                "model_fits": fit_results,
                "rsa": rsa_results,
            }

        log.info(f"  Completed {metric_name} analysis for all layers")

    # ── H1 evaluation (v2.7 Section 4.1) ──
    h1_evaluation = evaluate_h1(all_results)
    all_results["h1_evaluation"] = h1_evaluation

    # ── Model-fit floor check (v2.7 Section 4.1, H1 addendum) ──
    floor_check = check_model_fit_floor(all_results)
    all_results["model_fit_floor_check"] = floor_check

    # Save results
    out_dir = in_dir
    with open(out_dir / "paradigm_a_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save distance matrices as npz for figure generation
    np.savez_compressed(
        out_dir / "pairwise_distances.npz",
        euclidean=distances["euclidean"],
        cosine=distances["cosine"],
    )

    log.info(f"\nAnalysis saved to {out_dir}")
    return all_results


def evaluate_h1(results: dict) -> dict:
    """
    Evaluate H1 at this domain.

    v2.7: "H1 is supported if BOTH of the following hold, in at least 2 of 3
    domains, at ≥9 of 17 layers in the 16-32 range:
    (a) RSA Mantel test: Spearman rank correlation between the model RDM and
        the Weber theoretical RDM significantly exceeds the correlation with
        the Linear theoretical RDM (Mantel permutation test, p < 0.017);
    (b) AIC model selection: The Weber model achieves a lower AIC than the
        Linear model."

    This function evaluates at the domain level; cross-domain evaluation
    is done in the summary script.
    """
    evaluation = {"cosine": {}, "euclidean": {}}

    for metric in ["cosine", "euclidean"]:
        layers_pass = 0
        layer_details = []

        for layer in range(PRIMARY_LAYER_RANGE[0], PRIMARY_LAYER_RANGE[1]):
            layer_key = f"layer_{layer:02d}"
            layer_data = results["layers"].get(layer_key, {}).get(metric, {})

            if not layer_data or layer_data.get("status") == "nan":
                layer_details.append({"layer": layer, "pass": False, "reason": "no_data"})
                continue

            rsa = layer_data.get("rsa", {})
            fits = layer_data.get("model_fits", {})

            # (a) Mantel: Weber rho > Linear rho AND Weber p < alpha
            weber_rho = rsa.get("weber", {}).get("rho", 0)
            linear_rho = rsa.get("linear", {}).get("rho", 0)
            weber_p = rsa.get("weber", {}).get("p_value", 1)
            mantel_pass = (weber_rho > linear_rho) and (weber_p < BONFERRONI_ALPHA)

            # (b) AIC: Weber < Linear
            weber_aic = fits.get("weber", {}).get("aic", float("inf"))
            linear_aic = fits.get("linear", {}).get("aic", float("inf"))
            aic_pass = weber_aic < linear_aic

            layer_passes = mantel_pass and aic_pass
            if layer_passes:
                layers_pass += 1

            layer_details.append({
                "layer": layer,
                "pass": layer_passes,
                "mantel_pass": mantel_pass,
                "aic_pass": aic_pass,
                "weber_rho": weber_rho,
                "linear_rho": linear_rho,
                "weber_p": weber_p,
                "weber_aic": weber_aic,
                "linear_aic": linear_aic,
            })

        n_primary_layers = PRIMARY_LAYER_RANGE[1] - PRIMARY_LAYER_RANGE[0]
        domain_passes = layers_pass >= H1_MIN_LAYERS

        evaluation[metric] = {
            "layers_passing": layers_pass,
            "layers_tested": n_primary_layers,
            "threshold": H1_MIN_LAYERS,
            "domain_passes": domain_passes,
            "layer_details": layer_details,
        }

    return evaluation


def check_model_fit_floor(results: dict) -> dict:
    """
    Model-fit floor check (v2.7):
    "If the best-fitting model achieves R² < 0.20 at a majority of layers
    in the 16-32 range for a given domain, the three-model framework is
    inadequate."
    """
    floor = {}
    for metric in ["cosine", "euclidean"]:
        below_floor = 0
        n_tested = 0
        for layer in range(PRIMARY_LAYER_RANGE[0], PRIMARY_LAYER_RANGE[1]):
            layer_key = f"layer_{layer:02d}"
            fits = results["layers"].get(layer_key, {}).get(metric, {}).get("model_fits", {})
            best_r2 = fits.get("best_r2_value", None)
            if best_r2 is not None:
                n_tested += 1
                if best_r2 < MODEL_FIT_FLOOR_R2:
                    below_floor += 1

        floor[metric] = {
            "n_below_floor": below_floor,
            "n_tested": n_tested,
            "majority_below": below_floor > n_tested / 2 if n_tested > 0 else False,
            "triggers_e4": below_floor > n_tested / 2 if n_tested > 0 else False,
        }

    return floor


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Paradigm A: Analysis Pipeline")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--domain", required=True, choices=list(DOMAINS.keys()) + ["all"])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    domains = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]

    for domain_key in domains:
        log.info(f"\n{'='*60}")
        log.info(f"Paradigm A Analysis: {args.model} / {domain_key}")
        log.info(f"{'='*60}")

        try:
            results = analyse_domain(args.model, domain_key, args.results_dir)

            # Print summary
            for metric in ["cosine", "euclidean"]:
                h1 = results["h1_evaluation"][metric]
                log.info(
                    f"\nH1 ({metric}): {h1['layers_passing']}/{h1['layers_tested']} "
                    f"layers pass (need {h1['threshold']}). "
                    f"Domain {'PASSES' if h1['domain_passes'] else 'FAILS'}."
                )

                floor = results["model_fit_floor_check"][metric]
                if floor["triggers_e4"]:
                    log.warning(
                        f"  MODEL-FIT FLOOR TRIGGERED ({metric}): "
                        f"{floor['n_below_floor']}/{floor['n_tested']} layers "
                        f"have best R² < {MODEL_FIT_FLOOR_R2}. "
                        f"E4 residual periodicity analysis required."
                    )

        except FileNotFoundError as e:
            log.error(str(e))
            continue

    log.info("\n=== Analysis complete ===")


if __name__ == "__main__":
    main()
