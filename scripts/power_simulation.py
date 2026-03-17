#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — Pre-Registration Step 5: Power Analysis

Monte Carlo power simulation (5,000 iterations, seed 42) using the exact 
stimulus structure and analysis pipeline from v2.6.

Tests recovery of log geometry via Mantel test + AIC under noise.
Calibrates noise to produce the smallest effect we consider meaningful.

Author: JP Cacioli
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import json
import math
from pathlib import Path
from datetime import datetime

SEED = 42
N_SIMULATIONS = 5000
OUTPUT_DIR = Path("results/power_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Magnitudes (from pre-registration) ──
NUMERICAL_MAGNITUDES = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    150, 200, 300, 500, 700, 1000
], dtype=float)

N_MAGS = len(NUMERICAL_MAGNITUDES)
N_PAIRS = N_MAGS * (N_MAGS - 1) // 2  # 325


def compute_theoretical_rdm(magnitudes, model_type, beta=None):
    """Compute theoretical RDM for a given model."""
    n = len(magnitudes)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if model_type == "linear":
                rdm[i, j] = abs(magnitudes[i] - magnitudes[j])
            elif model_type == "log":
                rdm[i, j] = abs(np.log(magnitudes[i]) - np.log(magnitudes[j]))
            elif model_type == "power":
                rdm[i, j] = abs(magnitudes[i]**beta - magnitudes[j]**beta)
    return rdm


def rdm_to_vector(rdm):
    """Extract upper triangle as a vector."""
    n = rdm.shape[0]
    indices = np.triu_indices(n, k=1)
    return rdm[indices]


def mantel_test(rdm1, rdm2, n_perms=10000, seed=None):
    """
    Mantel test: Spearman correlation between two RDMs,
    significance via label permutation.
    """
    rng = np.random.RandomState(seed)
    
    vec1 = rdm_to_vector(rdm1)
    vec2 = rdm_to_vector(rdm2)
    
    observed_r, _ = stats.spearmanr(vec1, vec2)
    
    n = rdm1.shape[0]
    perm_rs = np.zeros(n_perms)
    
    for p in range(n_perms):
        perm = rng.permutation(n)
        rdm1_perm = rdm1[np.ix_(perm, perm)]
        vec1_perm = rdm_to_vector(rdm1_perm)
        perm_rs[p], _ = stats.spearmanr(vec1_perm, vec2)
    
    p_value = np.mean(perm_rs >= observed_r)
    return observed_r, p_value


def compute_aic(n, rss, k):
    """AIC = n * ln(RSS/n) + 2k"""
    return n * np.log(rss / n) + 2 * k


def fit_models(observed_distances, magnitudes):
    """
    Fit Linear, Log, and Stevens models to observed pairwise distances.
    Returns R², AIC for each.
    """
    n = len(magnitudes)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    
    y = observed_distances
    n_obs = len(y)
    
    # Linear predictor: |n_i - n_j|
    x_linear = np.array([abs(magnitudes[i] - magnitudes[j]) for i, j in pairs])
    
    # Log predictor: |log(n_i) - log(n_j)|
    log_mags = np.log(magnitudes)
    x_log = np.array([abs(log_mags[i] - log_mags[j]) for i, j in pairs])
    
    # Fit linear: d = a + b * x
    def linear_model(x, a, b):
        return a + b * x
    
    # Linear fit
    try:
        popt_lin, _ = curve_fit(linear_model, x_linear, y)
        y_pred_lin = linear_model(x_linear, *popt_lin)
        ss_res_lin = np.sum((y - y_pred_lin) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot
        aic_lin = compute_aic(n_obs, ss_res_lin, 2)
    except:
        r2_lin, aic_lin = 0, 1e10
    
    # Log fit
    try:
        popt_log, _ = curve_fit(linear_model, x_log, y)
        y_pred_log = linear_model(x_log, *popt_log)
        ss_res_log = np.sum((y - y_pred_log) ** 2)
        r2_log = 1 - ss_res_log / ss_tot
        aic_log = compute_aic(n_obs, ss_res_log, 2)
    except:
        r2_log, aic_log = 0, 1e10
    
    # Stevens fit: d = a + b * |n_i^beta - n_j^beta|
    def stevens_model(x_pair_indices, a, b, beta):
        result = []
        for idx in range(len(x_pair_indices)):
            i, j = pairs[idx]
            pred = a + b * abs(magnitudes[i]**beta - magnitudes[j]**beta)
            result.append(pred)
        return np.array(result)
    
    try:
        x_indices = np.arange(len(pairs))
        popt_stev, _ = curve_fit(
            stevens_model, x_indices, y,
            p0=[0, 1, 0.5], bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 2.0]),
            maxfev=5000,
        )
        y_pred_stev = stevens_model(x_indices, *popt_stev)
        ss_res_stev = np.sum((y - y_pred_stev) ** 2)
        r2_stev = 1 - ss_res_stev / ss_tot
        aic_stev = compute_aic(n_obs, ss_res_stev, 3)
        beta_est = popt_stev[2]
    except:
        r2_stev, aic_stev, beta_est = 0, 1e10, None
    
    return {
        "r2_linear": r2_lin, "aic_linear": aic_lin,
        "r2_log": r2_log, "aic_log": aic_log,
        "r2_stevens": r2_stev, "aic_stevens": aic_stev,
        "beta_stevens": beta_est,
    }


# ═══════════════════════════════════════════════════
# POWER SIMULATION: H1 (Geometry)
# ═══════════════════════════════════════════════════

def simulate_h1(n_sims=N_SIMULATIONS, noise_sd=0.3, alpha=0.017):
    """
    Simulate H1: Generate RDMs under log model + noise,
    test recovery via Mantel test + AIC.
    
    noise_sd calibrated to produce Mantel r_diff ≈ 0.05 (smallest meaningful effect).
    """
    print(f"\n{'='*60}")
    print(f"H1 Power Simulation: {n_sims} iterations, noise_sd={noise_sd}")
    print(f"{'='*60}")
    
    rng = np.random.RandomState(SEED)
    
    # Theoretical RDMs
    rdm_log = compute_theoretical_rdm(NUMERICAL_MAGNITUDES, "log")
    rdm_linear = compute_theoretical_rdm(NUMERICAL_MAGNITUDES, "linear")
    
    # Normalise theoretical RDMs
    rdm_log_vec = rdm_to_vector(rdm_log)
    rdm_linear_vec = rdm_to_vector(rdm_linear)
    
    successes_mantel = 0
    successes_aic = 0
    successes_both = 0
    
    for sim in range(n_sims):
        if (sim + 1) % 500 == 0:
            print(f"  Simulation {sim+1}/{n_sims}")
        
        # Generate observed RDM: log model + Gaussian noise
        noise = rng.normal(0, noise_sd, size=rdm_log.shape)
        noise = (noise + noise.T) / 2  # symmetrise
        np.fill_diagonal(noise, 0)
        
        observed_rdm = rdm_log + noise
        observed_rdm = np.maximum(observed_rdm, 0)  # non-negative distances
        
        observed_vec = rdm_to_vector(observed_rdm)
        
        # Mantel test: is log correlation > linear correlation?
        r_log, _ = stats.spearmanr(observed_vec, rdm_log_vec)
        r_linear, _ = stats.spearmanr(observed_vec, rdm_linear_vec)
        
        # Permutation test for the difference
        n = observed_rdm.shape[0]
        perm_diffs = np.zeros(1000)  # reduced perms for speed
        for p in range(1000):
            perm = rng.permutation(n)
            obs_perm = observed_rdm[np.ix_(perm, perm)]
            obs_perm_vec = rdm_to_vector(obs_perm)
            r_log_p, _ = stats.spearmanr(obs_perm_vec, rdm_log_vec)
            r_lin_p, _ = stats.spearmanr(obs_perm_vec, rdm_linear_vec)
            perm_diffs[p] = r_log_p - r_lin_p
        
        observed_diff = r_log - r_linear
        p_mantel = np.mean(perm_diffs >= observed_diff)
        
        mantel_sig = p_mantel < alpha
        
        # AIC comparison
        fits = fit_models(observed_vec, NUMERICAL_MAGNITUDES)
        aic_log_wins = fits["aic_log"] < fits["aic_linear"]
        
        if mantel_sig:
            successes_mantel += 1
        if aic_log_wins:
            successes_aic += 1
        if mantel_sig and aic_log_wins:
            successes_both += 1
    
    power_mantel = successes_mantel / n_sims
    power_aic = successes_aic / n_sims
    power_both = successes_both / n_sims
    
    print(f"\n  Power (Mantel only):     {power_mantel:.3f}")
    print(f"  Power (AIC only):        {power_aic:.3f}")
    print(f"  Power (Both — H1 criterion): {power_both:.3f}")
    
    return {
        "hypothesis": "H1",
        "n_simulations": n_sims,
        "noise_sd": noise_sd,
        "alpha": alpha,
        "power_mantel": power_mantel,
        "power_aic": power_aic,
        "power_both": power_both,
    }


# ═══════════════════════════════════════════════════
# POWER SIMULATION: H2 (Behavioural Weber's Law)
# ═══════════════════════════════════════════════════

def simulate_h2(n_sims=N_SIMULATIONS, weber_fraction=0.15, alpha=0.017):
    """
    Simulate H2: Generate comparison accuracy data under Weber's law,
    test via delta-deviance (log(ratio) vs abs diff).
    """
    print(f"\n{'='*60}")
    print(f"H2 Power Simulation: {n_sims} iterations, w={weber_fraction}")
    print(f"{'='*60}")
    
    rng = np.random.RandomState(SEED + 100)
    
    baselines = [10, 30, 100, 300, 1000]
    ratios = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
    pairs_per_cell = 50
    
    successes = 0
    
    for sim in range(n_sims):
        if (sim + 1) % 500 == 0:
            print(f"  Simulation {sim+1}/{n_sims}")
        
        log_ratios = []
        abs_diffs = []
        accuracies = []
        
        for baseline in baselines:
            for ratio in ratios:
                for _ in range(pairs_per_cell):
                    comparison = baseline * ratio
                    log_r = np.log(ratio)
                    abs_d = comparison - baseline
                    
                    # Weber's law: d' = log(ratio) / weber_fraction
                    d_prime = log_r / weber_fraction
                    p_correct = stats.norm.cdf(d_prime / np.sqrt(2))
                    
                    # Add noise
                    p_correct = np.clip(p_correct + rng.normal(0, 0.02), 0.01, 0.99)
                    correct = rng.binomial(1, p_correct)
                    
                    log_ratios.append(log_r)
                    abs_diffs.append(abs_d)
                    accuracies.append(correct)
        
        log_ratios = np.array(log_ratios)
        abs_diffs = np.array(abs_diffs)
        accuracies = np.array(accuracies)
        
        # Logistic regression: log(ratio) vs abs_diff
        from scipy.special import expit
        
        # Fit with log(ratio)
        def neg_log_lik(params, x, y):
            p = expit(params[0] + params[1] * x)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        from scipy.optimize import minimize
        
        # Model 1: log(ratio)
        res_log = minimize(neg_log_lik, [0, 1], args=(log_ratios, accuracies), method='Nelder-Mead')
        ll_log = -res_log.fun
        
        # Model 2: abs_diff (standardised)
        abs_diffs_std = (abs_diffs - abs_diffs.mean()) / (abs_diffs.std() + 1e-10)
        res_abs = minimize(neg_log_lik, [0, 1], args=(abs_diffs_std, accuracies), method='Nelder-Mead')
        ll_abs = -res_abs.fun
        
        # Delta deviance test
        delta_deviance = 2 * (ll_log - ll_abs)
        p_value = 1 - stats.chi2.cdf(max(0, delta_deviance), df=0)
        # Use df=0 is wrong; the models have same number of params
        # Actually we compare deviance: larger log-lik = better fit
        # Chi-sq test with df = difference in parameters = 0 doesn't work
        # Instead: compare using AIC or BIC
        aic_log = -2 * ll_log + 4  # 2 params
        aic_abs = -2 * ll_abs + 4  # 2 params
        
        # For the actual test: is log(ratio) a better predictor?
        # Since same df, just check which has higher log-likelihood
        if ll_log > ll_abs:
            # Additionally do a likelihood ratio test vs null
            res_null = minimize(neg_log_lik, [0, 0], args=(np.zeros_like(log_ratios), accuracies), method='Nelder-Mead')
            ll_null = -res_null.fun
            chi2_log = 2 * (ll_log - ll_null)
            p_log = 1 - stats.chi2.cdf(max(0, chi2_log), df=1)
            
            if p_log < alpha:
                successes += 1
    
    power = successes / n_sims
    print(f"\n  Power (H2): {power:.3f}")
    
    return {
        "hypothesis": "H2",
        "n_simulations": n_sims,
        "weber_fraction": weber_fraction,
        "alpha": alpha,
        "power": power,
    }


# ═══════════════════════════════════════════════════
# POWER SIMULATION: H3 (Precision Gradient)
# ═══════════════════════════════════════════════════

def simulate_h3(n_sims=N_SIMULATIONS, rho_true=-0.20, alpha=0.017):
    """
    Simulate H3: Generate precision gradients with true negative Spearman rho,
    test significance.
    """
    print(f"\n{'='*60}")
    print(f"H3 Power Simulation: {n_sims} iterations, rho={rho_true}")
    print(f"{'='*60}")
    
    rng = np.random.RandomState(SEED + 200)
    
    n_adjacent = len(NUMERICAL_MAGNITUDES) - 1  # 25
    magnitudes = NUMERICAL_MAGNITUDES[:-1]  # midpoints
    
    successes = 0
    
    for sim in range(n_sims):
        if (sim + 1) % 500 == 0:
            print(f"  Simulation {sim+1}/{n_sims}")
        
        # Generate precision values with negative correlation to magnitude
        # Under log geometry: precision ∝ 1/n
        true_precision = 1.0 / magnitudes
        
        # Add noise calibrated to achieve target rho
        noise_scale = np.std(true_precision) * abs(1 / rho_true - 1) ** 0.5
        noise = rng.normal(0, noise_scale, size=n_adjacent)
        observed_precision = true_precision + noise
        
        # Spearman correlation
        r, p = stats.spearmanr(magnitudes, observed_precision)
        
        if r < 0 and p / 2 < alpha:  # one-tailed
            successes += 1
    
    power = successes / n_sims
    print(f"\n  Power (H3): {power:.3f}")
    
    return {
        "hypothesis": "H3",
        "n_simulations": n_sims,
        "rho_true": rho_true,
        "n_adjacent_pairs": n_adjacent,
        "alpha": alpha,
        "power": power,
    }


# ═══════════════════════════════════════════════════
# POWER SIMULATION: H7 (Causal Intervention)
# ═══════════════════════════════════════════════════

def simulate_h7(n_sims=N_SIMULATIONS, effect_d=0.5, alpha=0.025):
    """
    Simulate H7: Magnitude patching produces larger shift than random.
    """
    print(f"\n{'='*60}")
    print(f"H7 Power Simulation: {n_sims} iterations, d={effect_d}")
    print(f"{'='*60}")
    
    rng = np.random.RandomState(SEED + 300)
    
    n_prompts = 200
    n_random_dirs = 10
    
    successes = 0
    
    for sim in range(n_sims):
        if (sim + 1) % 500 == 0:
            print(f"  Simulation {sim+1}/{n_sims}")
        
        # Random direction shifts: mean 0, sd 1
        random_shifts = rng.normal(0, 1, size=(n_random_dirs, n_prompts))
        
        # Magnitude direction shifts: mean = effect_d * sd, same sd
        mag_shifts = rng.normal(effect_d, 1, size=n_prompts)
        
        # For each prompt, check if magnitude shift > 97.5th percentile of random
        all_random = random_shifts.flatten()
        threshold = np.percentile(np.abs(all_random), 97.5)
        
        prop_exceeding = np.mean(np.abs(mag_shifts) > threshold)
        
        if prop_exceeding >= 0.75:
            successes += 1
    
    power = successes / n_sims
    print(f"\n  Power (H7): {power:.3f}")
    
    return {
        "hypothesis": "H7",
        "n_simulations": n_sims,
        "effect_d": effect_d,
        "n_prompts": n_prompts,
        "n_random_directions": n_random_dirs,
        "alpha": alpha,
        "power": power,
    }


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    print("Weber's Law Project 4.2 — Monte Carlo Power Analysis")
    print(f"Seed: {SEED}, Simulations: {N_SIMULATIONS}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    results = {}
    
    # H1: Geometry
    results["H1"] = simulate_h1(n_sims=N_SIMULATIONS, noise_sd=0.3)
    
    # H2: Behaviour
    results["H2"] = simulate_h2(n_sims=N_SIMULATIONS, weber_fraction=0.15)
    
    # H3: Precision
    results["H3"] = simulate_h3(n_sims=N_SIMULATIONS, rho_true=-0.20)
    
    # H7: Causal
    results["H7"] = simulate_h7(n_sims=N_SIMULATIONS, effect_d=0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("POWER ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  H1 (Geometry — Mantel + AIC): {results['H1']['power_both']:.1%}")
    print(f"  H2 (Behaviour — delta deviance): {results['H2']['power']:.1%}")
    print(f"  H3 (Precision — Spearman rho): {results['H3']['power']:.1%}")
    print(f"  H7 (Causal — 97.5th percentile): {results['H7']['power']:.1%}")
    print()
    
    all_above_80 = all([
        results["H1"]["power_both"] >= 0.80,
        results["H2"]["power"] >= 0.80,
        results["H3"]["power"] >= 0.80,
        results["H7"]["power"] >= 0.80,
    ])
    
    if all_above_80:
        print("  ✓ All power estimates ≥ 80%. Design is adequately powered.")
    else:
        print("  ⚠ Some power estimates < 80%. Consider adjusting parameters.")
    
    # Save
    output = {
        "project": "Classical Minds, Modern Machines — Project 4.2",
        "version": "v2.6",
        "seed": SEED,
        "n_simulations": N_SIMULATIONS,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    
    out_path = OUTPUT_DIR / "power_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
