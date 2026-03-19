"""
qwen_h2_analysis.py — Δ deviance test + Weber fraction estimation for Qwen B1
Matches the primary models' analysis pipeline exactly.
"""

import json
import numpy as np
from scipy import stats
from collections import defaultdict

# Load Qwen B1 cross-format results
with open(r"C:\weber\results\exploratory\qwen25_7b\paradigm_b1_crossformat.json") as f:
    data = json.load(f)

items = data["items"]
print(f"Loaded {len(items)} B1 items for Qwen-2.5-7B-Instruct")
print(f"Overall accuracy: {data['overall_accuracy']:.3f}")
print()

# ═══════════════════════════════════════════════════════════════
# 1. Δ DEVIANCE TEST (H2 criterion)
# Pre-reg: logistic regression on log(ratio) vs absolute difference
# ═══════════════════════════════════════════════════════════════

from scipy.optimize import minimize

def neg_log_likelihood(beta, X, y):
    """Negative log-likelihood for logistic regression."""
    z = X @ beta
    z = np.clip(z, -500, 500)
    p = 1 / (1 + np.exp(-z))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

# Build data arrays
correct = np.array([1 if it["correct"] else 0 for it in items], dtype=float)
ratios = np.array([it["ratio"] for it in items], dtype=float)
baselines = np.array([it["baseline"] for it in items], dtype=float)

# Compute absolute differences and log ratios
# For items from comparison_numerical.json via paradigm_b_raw.json format:
# The actual values are first_presented and second_presented
# But we have ratio and baseline, so: larger = baseline * ratio, abs_diff = baseline * (ratio - 1)
abs_diffs = baselines * (ratios - 1)
log_ratios = np.log(ratios)

# Model 1: intercept + abs_diff
X_abs = np.column_stack([np.ones(len(items)), abs_diffs])
res_abs = minimize(neg_log_likelihood, np.zeros(2), args=(X_abs, correct), method="L-BFGS-B")
nll_abs = res_abs.fun
deviance_abs = 2 * nll_abs

# Model 2: intercept + log_ratio
X_log = np.column_stack([np.ones(len(items)), log_ratios])
res_log = minimize(neg_log_likelihood, np.zeros(2), args=(X_log, correct), method="L-BFGS-B")
nll_log = res_log.fun
deviance_log = 2 * nll_log

# Δ deviance = deviance(abs_diff model) - deviance(log_ratio model)
# Positive Δ means log_ratio is better (Weber-like)
delta_deviance = deviance_abs - deviance_log

# Chi-squared test (1 df — same number of parameters)
from scipy.stats import chi2
p_value = 1 - chi2.cdf(abs(delta_deviance), df=1)

print("=" * 50)
print("Δ DEVIANCE TEST (H2)")
print("=" * 50)
print(f"  Deviance (abs diff model):  {deviance_abs:.2f}")
print(f"  Deviance (log ratio model): {deviance_log:.2f}")
print(f"  Δ deviance:                 {delta_deviance:.2f}")
print(f"  p-value:                    {p_value:.6f}")
print(f"  Direction:                  {'log_ratio BETTER (Weber-like)' if delta_deviance > 0 else 'abs_diff BETTER (not Weber)'}")
print(f"  H2 criterion (p < .017):    {'PASS' if p_value < 0.017 and delta_deviance > 0 else 'FAIL'}")
print()

# ═══════════════════════════════════════════════════════════════
# 2. WEBER FRACTION ESTIMATION (position-corrected, BCa bootstrap)
# ═══════════════════════════════════════════════════════════════

# Group by baseline
baseline_data = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
for it in items:
    b = it["baseline"]
    r = it["ratio"]
    baseline_data[b][r]["total"] += 1
    if it["correct"]:
        baseline_data[b][r]["correct"] += 1

print("=" * 50)
print("ACCURACY BY BASELINE × RATIO")
print("=" * 50)
for b in sorted(baseline_data.keys()):
    print(f"\n  Baseline {b}:")
    for r in sorted(baseline_data[b].keys()):
        d = baseline_data[b][r]
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0
        print(f"    Ratio {r:.2f}: {acc:.3f} ({d['correct']}/{d['total']})")

# Estimate position bias (overall proportion choosing A vs B regardless of correctness)
a_count = sum(1 for it in items if it["predicted"] == "A")
b_count = sum(1 for it in items if it["predicted"] == "B")
position_bias = a_count / len(items)
print(f"\n  Position bias (prop choosing A): {position_bias:.3f}")

# Weber fraction estimation per baseline
# WF = ratio at 75% correct threshold
# Using logistic fit per baseline: accuracy = f(log_ratio)
print("\n" + "=" * 50)
print("WEBER FRACTION PER BASELINE")
print("=" * 50)

def estimate_wf_at_baseline(baseline_items, threshold=0.75):
    """Estimate Weber fraction at a given baseline using logistic fit."""
    ratios_b = []
    accs_b = []
    
    ratio_groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for it in baseline_items:
        r = it["ratio"]
        ratio_groups[r]["total"] += 1
        if it["correct"]:
            ratio_groups[r]["correct"] += 1
    
    for r in sorted(ratio_groups.keys()):
        d = ratio_groups[r]
        if d["total"] >= 3:  # minimum trials
            ratios_b.append(r)
            accs_b.append(d["correct"] / d["total"])
    
    if len(ratios_b) < 3:
        return None, ratios_b, accs_b
    
    log_ratios_b = np.log(ratios_b)
    accs_b = np.array(accs_b)
    
    # Fit logistic: p = 1/(1+exp(-(a + b*log_ratio)))
    # Find ratio where p = threshold
    X = np.column_stack([np.ones(len(log_ratios_b)), log_ratios_b])
    
    # Weight by n per cell
    from scipy.optimize import curve_fit
    def logistic(x, a, b):
        z = a + b * x
        return 1 / (1 + np.exp(-z))
    
    try:
        popt, _ = curve_fit(logistic, np.array(np.log(ratios_b)), accs_b, 
                           p0=[0, 5], maxfev=5000)
        # Threshold: threshold = 1/(1+exp(-(a + b*log_r)))
        # => log_r = (log(threshold/(1-threshold)) - a) / b
        log_r_threshold = (np.log(threshold / (1 - threshold)) - popt[0]) / popt[1]
        wf = np.exp(log_r_threshold) - 1  # WF = ratio - 1 at threshold
        return wf, ratios_b, accs_b
    except Exception:
        return None, ratios_b, accs_b

wfs = {}
for b in sorted(baseline_data.keys()):
    b_items = [it for it in items if it["baseline"] == b]
    wf, ratios_b, accs_b = estimate_wf_at_baseline(b_items)
    wfs[b] = wf
    status = f"{wf:.3f}" if wf is not None else "N/A (fit failed)"
    print(f"  Baseline {b:.0f}: WF = {status}")

valid_wfs = [v for v in wfs.values() if v is not None and 0 < v < 5]
if valid_wfs:
    aggregate_wf = np.median(valid_wfs)
    print(f"\n  Aggregate WF (median): {aggregate_wf:.3f}")
    print(f"  In human range (0.10-0.25)? {'YES' if 0.10 <= aggregate_wf <= 0.25 else 'NO'}")
else:
    aggregate_wf = None
    print("\n  Could not estimate aggregate WF")

# ═══════════════════════════════════════════════════════════════
# 3. BCa BOOTSTRAP for aggregate WF
# ═══════════════════════════════════════════════════════════════

if valid_wfs:
    print("\n" + "=" * 50)
    print("BCa BOOTSTRAP (10,000 iterations, seed 42)")
    print("=" * 50)
    
    rng = np.random.RandomState(42)
    n_boot = 10000
    boot_wfs = []
    
    for i in range(n_boot):
        # Resample items with replacement
        boot_items = [items[j] for j in rng.randint(0, len(items), len(items))]
        
        boot_baseline_wfs = []
        for b in sorted(baseline_data.keys()):
            b_items = [it for it in boot_items if it["baseline"] == b]
            if len(b_items) < 10:
                continue
            wf, _, _ = estimate_wf_at_baseline(b_items)
            if wf is not None and 0 < wf < 5:
                boot_baseline_wfs.append(wf)
        
        if boot_baseline_wfs:
            boot_wfs.append(np.median(boot_baseline_wfs))
    
    boot_wfs = np.array(boot_wfs)
    
    # BCa intervals
    # Bias correction
    z0 = stats.norm.ppf(np.mean(boot_wfs < aggregate_wf))
    
    # Acceleration (jackknife)
    jack_wfs = []
    for i in range(len(items)):
        jack_items = items[:i] + items[i+1:]
        jack_baseline_wfs = []
        for b in sorted(baseline_data.keys()):
            b_items = [it for it in jack_items if it["baseline"] == b]
            if len(b_items) < 10:
                continue
            wf, _, _ = estimate_wf_at_baseline(b_items)
            if wf is not None and 0 < wf < 5:
                jack_baseline_wfs.append(wf)
        if jack_baseline_wfs:
            jack_wfs.append(np.median(jack_baseline_wfs))
    
    if len(jack_wfs) > 2:
        jack_mean = np.mean(jack_wfs)
        a_hat = np.sum((jack_mean - np.array(jack_wfs))**3) / (6 * np.sum((jack_mean - np.array(jack_wfs))**2)**1.5 + 1e-15)
    else:
        a_hat = 0
    
    # BCa percentiles
    alpha = 0.05
    z_low = stats.norm.ppf(alpha / 2)
    z_high = stats.norm.ppf(1 - alpha / 2)
    
    a1 = stats.norm.cdf(z0 + (z0 + z_low) / (1 - a_hat * (z0 + z_low)))
    a2 = stats.norm.cdf(z0 + (z0 + z_high) / (1 - a_hat * (z0 + z_high)))
    
    ci_low = np.percentile(boot_wfs, 100 * a1) if len(boot_wfs) > 0 else np.nan
    ci_high = np.percentile(boot_wfs, 100 * a2) if len(boot_wfs) > 0 else np.nan
    
    print(f"  Aggregate WF: {aggregate_wf:.3f}")
    print(f"  BCa 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  Bootstrap samples: {len(boot_wfs)}/{n_boot}")
    print(f"  In human range (0.10-0.25)? Point estimate: {'YES' if 0.10 <= aggregate_wf <= 0.25 else 'NO'}")

# ═══════════════════════════════════════════════════════════════
# 4. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════

results = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "task": "B1_cross_format",
    "delta_deviance": {
        "value": float(delta_deviance),
        "p_value": float(p_value),
        "direction": "log_ratio_better" if delta_deviance > 0 else "abs_diff_better",
        "h2_pass": bool(p_value < 0.017 and delta_deviance > 0),
        "deviance_abs": float(deviance_abs),
        "deviance_log": float(deviance_log),
    },
    "weber_fraction": {
        "aggregate": float(aggregate_wf) if aggregate_wf else None,
        "per_baseline": {str(k): float(v) if v else None for k, v in wfs.items()},
        "bca_ci_low": float(ci_low) if valid_wfs else None,
        "bca_ci_high": float(ci_high) if valid_wfs else None,
        "in_human_range": bool(aggregate_wf and 0.10 <= aggregate_wf <= 0.25),
    },
    "position_bias": float(position_bias),
    "mean_entropy": float(data["mean_entropy"]),
    "overall_accuracy": float(data["overall_accuracy"]),
}

out_path = r"C:\weber\results\exploratory\qwen25_7b\qwen_h2_analysis.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

# Summary
print("\n" + "=" * 50)
print("SUMMARY FOR MANUSCRIPT")
print("=" * 50)
print(f"  Qwen B1 accuracy:     {data['overall_accuracy']:.3f}")
print(f"  Δ deviance:            {delta_deviance:.2f} (p = {p_value:.4f})")
print(f"  H2 criterion:          {'PASS' if p_value < 0.017 and delta_deviance > 0 else 'FAIL'}")
print(f"  Aggregate WF:          {aggregate_wf:.3f}" if aggregate_wf else "  Aggregate WF: N/A")
print(f"  BCa 95% CI:            [{ci_low:.3f}, {ci_high:.3f}]" if valid_wfs else "")
print(f"  Mean entropy:          {data['mean_entropy']:.3f}")
