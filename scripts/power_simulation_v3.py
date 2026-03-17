"""
Weber's Law Project 4.2 — Power Analysis v3 (analytic + Monte Carlo)

H1, H2: Monte Carlo (confirmed at ceiling)
H3: Analytic power for Spearman correlation test (n=25)
H7: Analytic characterisation of the detection threshold

Run: python scripts\power_simulation_v3.py
"""
import numpy as np
from scipy import stats
import json
from pathlib import Path
from datetime import datetime

SEED = 42
OUTPUT_DIR = Path("results/power_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def power_spearman_one_tailed(n, rho, alpha):
    """
    Analytic approximation of power for one-tailed Spearman test.
    Uses the Fisher z-transform approximation.
    
    For Spearman with n observations at significance alpha (one-tailed),
    testing H0: rho >= 0 vs H1: rho < 0.
    """
    # Fisher z-transform of the population correlation
    z_rho = np.arctanh(rho)
    # Standard error of Fisher z
    se = 1.0 / np.sqrt(n - 3)
    # Critical z for one-tailed alpha
    z_crit = stats.norm.ppf(alpha)  # negative for left tail
    # Power = P(z_obs < z_crit | rho)
    # z_obs ~ N(z_rho, se)
    power = stats.norm.cdf((z_crit * se - z_rho) / se)
    # Simpler: power = P(Z < z_crit - z_rho/se) where Z is standard normal
    power = stats.norm.cdf(z_crit - z_rho / se)
    return power


def h3_power_analysis():
    """
    H3: Precision gradient.
    n = 25 adjacent pairs.
    Alpha = 0.017 (one-tailed, Bonferroni corrected).
    
    Under log geometry, true precision ~ 1/n, giving rho with magnitude
    very strongly negative. We compute power across a range of rho values.
    """
    print("=" * 60)
    print("H3 Power Analysis (analytic)")
    print("=" * 60)
    
    n = 25  # adjacent pairs
    alpha = 0.017  # one-tailed, Bonferroni
    
    print(f"  n = {n} adjacent pairs")
    print(f"  alpha = {alpha} (one-tailed, Bonferroni)")
    print(f"  Test: Spearman rho(magnitude, precision) < 0")
    print()
    
    # Power curve
    rhos = [-0.20, -0.30, -0.40, -0.50, -0.60, -0.70, -0.80, -0.90]
    print(f"  {'rho':>8}  {'power':>8}")
    print(f"  {'-'*8}  {'-'*8}")
    
    results = {}
    for rho in rhos:
        pwr = power_spearman_one_tailed(n, rho, alpha)
        print(f"  {rho:>8.2f}  {pwr:>8.3f}")
        results[f"rho_{rho}"] = pwr
    
    # What rho gives 80% power?
    from scipy.optimize import brentq
    rho_80 = brentq(lambda r: power_spearman_one_tailed(n, r, alpha) - 0.80, -0.99, -0.01)
    print(f"\n  Minimum |rho| for 80% power: {rho_80:.3f}")
    print(f"  Minimum |rho| for 90% power: {brentq(lambda r: power_spearman_one_tailed(n, r, alpha) - 0.90, -0.99, -0.01):.3f}")
    
    # Verify with Monte Carlo
    print(f"\n  Monte Carlo verification (rho = -0.60, 5000 sims):")
    rng = np.random.RandomState(SEED)
    
    # Generate correlated ranks
    successes = 0
    n_sims = 5000
    target_rho = -0.60
    
    for _ in range(n_sims):
        # Generate bivariate normal with target correlation, then rank
        x = rng.normal(0, 1, n)
        y = target_rho * x + np.sqrt(1 - target_rho**2) * rng.normal(0, 1, n)
        r, p = stats.spearmanr(x, y)
        if r < 0 and p/2 < alpha:
            successes += 1
    
    mc_power = successes / n_sims
    analytic_power = power_spearman_one_tailed(n, target_rho, alpha)
    print(f"    Analytic: {analytic_power:.3f}")
    print(f"    Monte Carlo: {mc_power:.3f}")
    print(f"    Agreement: {'GOOD' if abs(analytic_power - mc_power) < 0.05 else 'CHECK'}")
    
    # What we expect under log geometry
    print(f"\n  Expected effect under log geometry:")
    print(f"    True precision ~ 1/n, so rho(n, 1/n) is strongly negative.")
    
    # Compute the actual Spearman correlation for the exact stimulus set
    mags = np.array([1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,
                     150,200,300,500,700,1000], dtype=float)
    midpoints = (mags[:-1] + mags[1:]) / 2
    true_precision = 1.0 / (mags[1:] - mags[:-1])  # raw precision (not log-normalised)
    true_rho, _ = stats.spearmanr(midpoints, true_precision)
    print(f"    Spearman rho(midpoints, 1/step_size) = {true_rho:.3f}")
    print(f"    Power at this rho: {power_spearman_one_tailed(n, true_rho, alpha):.3f}")
    
    # Under log geometry: precision proportional to 1/n
    true_precision_log = 1.0 / midpoints
    true_rho_log, _ = stats.spearmanr(midpoints, true_precision_log)
    print(f"    Spearman rho(midpoints, 1/midpoint) = {true_rho_log:.3f}")
    print(f"    Power at this rho: {power_spearman_one_tailed(n, true_rho_log, alpha):.3f}")
    
    return {
        "hypothesis": "H3",
        "method": "analytic (Fisher z-transform)",
        "n_adjacent_pairs": n,
        "alpha": alpha,
        "power_curve": results,
        "min_rho_80pct_power": round(rho_80, 3),
        "expected_rho_log_geometry": round(true_rho_log, 3),
        "expected_power_log_geometry": round(power_spearman_one_tailed(n, true_rho_log, alpha), 3),
        "mc_verification": {"rho": -0.60, "analytic": round(analytic_power, 3), "mc": round(mc_power, 3)},
    }


def h7_power_analysis():
    """
    H7: Causal intervention.
    
    The criterion: magnitude-direction shift exceeds 97.5th percentile
    of 2000 pooled random-direction shifts, for >= 75% of 200 prompts.
    
    The 97.5th percentile of |N(0,1)| with 2000 draws is ~2.24.
    For magnitude shifts from |N(d,1)|, P(|N(d,1)| > 2.24) depends on d.
    We need this probability >= 0.75.
    
    This is an analytic calculation.
    """
    print(f"\n{'='*60}")
    print("H7 Power Analysis (analytic)")
    print("=" * 60)
    
    n_random = 2000
    n_prompts = 200
    prop_threshold = 0.75
    
    # 97.5th percentile of |N(0,1)| with 2000 draws
    # The 97.5th percentile of the absolute standard normal is ~1.96
    # But with 2000 draws, the empirical 97.5th percentile of |N(0,1)| ≈ 1.96
    # (since the theoretical 97.5th percentile of |N(0,1)| = qnorm(0.9875) ≈ 2.24)
    # Wait: |N(0,1)| is a half-normal. P(|Z| > x) = 2*(1 - Phi(x)).
    # The 97.5th percentile of |N(0,1)| means P(|Z| <= t) = 0.975
    # P(|Z| <= t) = P(-t <= Z <= t) = 2*Phi(t) - 1 = 0.975
    # Phi(t) = 0.9875, t = 2.24
    
    threshold = stats.norm.ppf(0.9875)  # ~2.24
    print(f"  Pooled null threshold (97.5th pctile of |N(0,1)|): {threshold:.3f}")
    print(f"  Required exceedance proportion: {prop_threshold}")
    print(f"  n_prompts: {n_prompts}")
    
    print(f"\n  Per-prompt exceedance probability P(|N(d,1)| > {threshold:.2f}):")
    print(f"  {'d':>6}  {'P(exceed)':>10}  {'E[prompts]':>12}  {'P(>=75%)':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*10}")
    
    results = {}
    for d in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        # P(|N(d,1)| > threshold) = P(N(d,1) > threshold) + P(N(d,1) < -threshold)
        p_exceed = 1 - stats.norm.cdf(threshold - d) + stats.norm.cdf(-threshold - d)
        expected_prompts = p_exceed * n_prompts
        
        # P(>= 75% of prompts exceed) = P(Binomial(200, p_exceed) >= 150)
        p_75pct = 1 - stats.binom.cdf(int(prop_threshold * n_prompts) - 1, n_prompts, p_exceed)
        
        print(f"  {d:>6.1f}  {p_exceed:>10.4f}  {expected_prompts:>12.1f}  {p_75pct:>10.4f}")
        results[f"d_{d}"] = {
            "d": d,
            "p_exceed_per_prompt": round(p_exceed, 4),
            "expected_prompts_exceeding": round(expected_prompts, 1),
            "p_75pct_criterion": round(p_75pct, 4),
        }
    
    # What d gives 80% power?
    from scipy.optimize import brentq
    try:
        def power_at_d(d):
            p_ex = 1 - stats.norm.cdf(threshold - d) + stats.norm.cdf(-threshold - d)
            return stats.binom.sf(int(prop_threshold * n_prompts) - 1, n_prompts, p_ex) - 0.80
        d_80 = brentq(power_at_d, 0.1, 10.0)
        print(f"\n  Minimum d for 80% power: {d_80:.2f}")
    except:
        d_80 = None
        print(f"\n  Could not find d for 80% power in range [0.1, 10.0]")
    
    print(f"\n  INTERPRETATION:")
    print(f"  The 75%-of-prompts criterion requires d ≈ {d_80:.1f} to achieve 80% power." if d_80 else "")
    print(f"  In activation patching, effect sizes are typically measured in")
    print(f"  comparison-probability shifts (delta_p), not standardised d.")
    print(f"  The power depends on how much variance the magnitude direction")
    print(f"  captures relative to random directions — this is unknown a priori.")
    print(f"  H7 power is therefore stated as conditional on the true effect size,")
    print(f"  with the minimum detectable effect reported.")
    
    return {
        "hypothesis": "H7",
        "method": "analytic (binomial + normal)",
        "null_threshold": round(threshold, 3),
        "power_curve": results,
        "min_d_80pct_power": round(d_80, 2) if d_80 else None,
        "note": "Power is conditional on true effect size, which is unknown a priori for activation patching.",
    }


def main():
    print("Weber's Law Project 4.2 — Power Analysis v3")
    print(f"Seed: {SEED}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = {}
    
    # H1, H2: confirmed at ceiling from v1
    results["H1"] = {"hypothesis": "H1", "power": 1.000, "method": "Monte Carlo (5000 sims)", 
                      "note": "Power at ceiling; design massively overpowered for geometry detection"}
    results["H2"] = {"hypothesis": "H2", "power": 1.000, "method": "Monte Carlo (5000 sims)",
                      "note": "Power at ceiling; 1500 items provides overwhelming power"}
    
    print(f"\n  H1 (Geometry): 100.0% (Monte Carlo, 5000 sims)")
    print(f"  H2 (Behaviour): 100.0% (Monte Carlo, 5000 sims)")
    
    # H3: analytic
    results["H3"] = h3_power_analysis()
    
    # H7: analytic
    results["H7"] = h7_power_analysis()
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL POWER SUMMARY")
    print(f"{'='*60}")
    print(f"  H1 (Geometry):  ≥99% at r_diff=0.05 (N=325 pairs)")
    print(f"  H2 (Behaviour): ≥99% at w=0.15 (N=1500 items)")
    print(f"  H3 (Precision): {results['H3']['expected_power_log_geometry']:.0%} at expected rho={results['H3']['expected_rho_log_geometry']}")
    print(f"                  ≥80% at |rho| ≥ {abs(results['H3']['min_rho_80pct_power'])}")
    if results["H7"]["min_d_80pct_power"]:
        print(f"  H7 (Causal):    ≥80% at d ≥ {results['H7']['min_d_80pct_power']}")
    print(f"                  Power conditional on true effect size (unknown a priori)")
    
    out_path = OUTPUT_DIR / "power_analysis_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
