#!/usr/bin/env python3
"""
Psychometric Function Fitting and Weber Fraction Estimation
Weber's Law in Transformer Magnitude Representations (Project 4.2)

Fixes the failed psychometric fits from the initial run by:
1. Position-bias correction: averaging accuracy across A-correct and B-correct trials
2. Fitting on log(ratio) as the independent variable
3. Bootstrap CIs (1000 iterations, non-parametric, per pre-reg)

Pre-registration reference: v2.7 Section 5.4 (Paradigm B)
Output: Weber fractions per baseline, psychometric curves for figures F2/F3

Author: JP Cacioli
Date: March 2026
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Position-Bias Correction
# ---------------------------------------------------------------------------

def compute_position_corrected_accuracy(trials):
    """Compute accuracy corrected for A/B position bias.
    
    For each ratio × baseline cell, average accuracy across A-correct
    and B-correct trials separately, then average the two. This cancels
    any systematic bias toward answering A or B.
    
    Args:
        trials: list of dicts with 'correct_answer', 'correct', 'ratio', 'baseline'
        
    Returns:
        cells: dict of (baseline, ratio) → {acc_a, acc_b, acc_corrected, n_a, n_b}
    """
    # Group by baseline × ratio
    groups = {}
    for t in trials:
        key = (t['baseline'], t['ratio'])
        if key not in groups:
            groups[key] = {'A': [], 'B': []}
        groups[key][t['correct_answer']].append(t['correct'])
    
    cells = {}
    for (bl, ratio), positions in groups.items():
        acc_a = np.mean(positions['A']) if positions['A'] else 0.5
        acc_b = np.mean(positions['B']) if positions['B'] else 0.5
        acc_corrected = (acc_a + acc_b) / 2.0
        cells[(bl, ratio)] = {
            'acc_a': float(acc_a),
            'acc_b': float(acc_b),
            'acc_corrected': float(acc_corrected),
            'n_a': len(positions['A']),
            'n_b': len(positions['B']),
            'n_total': len(positions['A']) + len(positions['B']),
        }
    
    return cells


# ---------------------------------------------------------------------------
# Psychometric Function
# ---------------------------------------------------------------------------

def psychometric_function(log_ratio, mu, sigma, lapse):
    """Psychometric function: cumulative normal with lapse rate.
    
    P(correct) = lapse/2 + (1 - lapse) * Φ((log_ratio - μ) / σ)
    
    where Φ is the standard normal CDF.
    
    Parameters:
        log_ratio: log(comparison/baseline) — the stimulus intensity
        mu: threshold (log-ratio at 75% correct for lapse=0)
        sigma: slope (spread of the psychometric function)
        lapse: lapse rate (probability of random response)
    """
    return lapse / 2.0 + (1.0 - lapse) * norm.cdf((log_ratio - mu) / sigma)


def neg_log_likelihood(params, log_ratios, n_correct, n_total):
    """Negative log-likelihood for psychometric function fitting.
    
    Uses binomial likelihood at each ratio level.
    """
    mu, sigma, lapse = params
    
    # Bounds enforcement
    if sigma <= 0.001 or lapse < 0 or lapse > 0.5:
        return 1e10
    
    p_correct = psychometric_function(log_ratios, mu, sigma, lapse)
    p_correct = np.clip(p_correct, 1e-10, 1 - 1e-10)
    
    # Binomial log-likelihood
    ll = np.sum(n_correct * np.log(p_correct) + (n_total - n_correct) * np.log(1 - p_correct))
    return -ll


def fit_psychometric(log_ratios, accuracies, n_per_cell, fast=False):
    """Fit psychometric function to accuracy data at one baseline.
    
    Args:
        log_ratios: array of log(ratio) values
        accuracies: array of position-corrected accuracies
        n_per_cell: array of total trials per cell (for binomial weighting)
        fast: if True, use single start and lower maxiter (for bootstrap)
        
    Returns:
        dict with mu, sigma, lapse, weber_fraction, converged, etc.
    """
    n_correct = np.round(accuracies * n_per_cell).astype(int)
    
    if fast:
        starts = [[np.log(1.3), 0.5, 0.05]]
        maxiter = 2000
    else:
        starts = [
            [np.log(1.1), 0.3, 0.02],
            [np.log(1.3), 0.5, 0.05],
            [np.log(1.5), 0.8, 0.10],
            [np.log(2.0), 1.0, 0.05],
            [0.0, 0.5, 0.05],
        ]
        maxiter = 10000
    
    best_result = None
    best_nll = np.inf
    
    for start in starts:
        try:
            result = minimize(
                neg_log_likelihood,
                start,
                args=(log_ratios, n_correct, n_per_cell),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        return {
            'mu': None, 'sigma': None, 'lapse': None,
            'weber_fraction': None, 'converged': False,
        }
    
    mu, sigma, lapse = best_result.x
    
    # Weber fraction: the ratio at which accuracy = 75% (threshold)
    # From the psychometric function: P = 0.75 when
    # lapse/2 + (1-lapse) * Φ((log_r - μ)/σ) = 0.75
    # Φ((log_r - μ)/σ) = (0.75 - lapse/2) / (1 - lapse)
    # log_r = μ + σ * Φ⁻¹((0.75 - lapse/2) / (1 - lapse))
    target_p = 0.75
    inner = (target_p - lapse / 2.0) / (1.0 - lapse)
    
    if 0 < inner < 1:
        log_threshold = mu + sigma * norm.ppf(inner)
        weber_fraction = np.exp(log_threshold) - 1.0  # ratio - 1
    else:
        weber_fraction = None
    
    # Predicted curve for plotting (skip in fast/bootstrap mode)
    if not fast:
        log_ratio_fine = np.linspace(-0.1, np.log(4), 100)
        predicted = psychometric_function(log_ratio_fine, mu, sigma, lapse)
        extra = {
            'predicted_log_ratios': log_ratio_fine.tolist(),
            'predicted_accuracy': predicted.tolist(),
        }
    else:
        extra = {}
    
    result = {
        'mu': float(mu),
        'sigma': float(sigma),
        'lapse': float(lapse),
        'weber_fraction': float(weber_fraction) if weber_fraction is not None else None,
        'converged': bool(best_result.success),
        'neg_loglik': float(best_nll),
    }
    result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Bootstrap Weber Fractions
# ---------------------------------------------------------------------------

def bootstrap_weber_fractions(trials_at_baseline, n_bootstrap=1000, seed=42):
    """Bootstrap CI for Weber fraction at one baseline.
    
    Resamples trials within each ratio × position cell, recomputes
    position-corrected accuracy, refits psychometric function.
    
    Pre-reg: 1000 iterations, non-parametric bootstrap.
    """
    rng = np.random.RandomState(seed)
    
    # Group trials by ratio × position
    groups = {}
    for t in trials_at_baseline:
        key = (t['ratio'], t['correct_answer'])
        if key not in groups:
            groups[key] = []
        groups[key].append(t['correct'])
    
    ratios = sorted(set(t['ratio'] for t in trials_at_baseline))
    log_ratios = np.log(np.array(ratios))
    
    weber_fracs = []
    
    for b in range(n_bootstrap):
        # Resample within each cell
        accs = []
        ns = []
        for ratio in ratios:
            a_trials = groups.get((ratio, 'A'), [])
            b_trials = groups.get((ratio, 'B'), [])
            
            if a_trials:
                a_boot = rng.choice(a_trials, size=len(a_trials), replace=True)
                acc_a = np.mean(a_boot)
            else:
                acc_a = 0.5
            
            if b_trials:
                b_boot = rng.choice(b_trials, size=len(b_trials), replace=True)
                acc_b = np.mean(b_boot)
            else:
                acc_b = 0.5
            
            accs.append((acc_a + acc_b) / 2.0)
            ns.append(len(a_trials) + len(b_trials))
        
        accs = np.array(accs)
        ns = np.array(ns)
        
        fit = fit_psychometric(log_ratios, accs, ns, fast=True)
        if fit['weber_fraction'] is not None and 0 < fit['weber_fraction'] < 10:
            weber_fracs.append(fit['weber_fraction'])
    
    if len(weber_fracs) < 10:
        return {
            'n_valid': len(weber_fracs),
            'n_bootstrap': n_bootstrap,
            'median': None,
            'ci_low': None,
            'ci_high': None,
            'mean': None,
            'std': None,
        }
    
    weber_fracs = np.array(weber_fracs)
    
    return {
        'n_valid': len(weber_fracs),
        'n_bootstrap': n_bootstrap,
        'median': float(np.median(weber_fracs)),
        'mean': float(np.mean(weber_fracs)),
        'std': float(np.std(weber_fracs)),
        'ci_low': float(np.percentile(weber_fracs, 2.5)),
        'ci_high': float(np.percentile(weber_fracs, 97.5)),
        'all_fractions': weber_fracs.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_psychometric_fitting(model_key, project_root):
    """Run position-corrected psychometric fitting for one model."""
    
    results_dir = Path(project_root) / "results" / "paradigm_b" / f"{model_key}_instruct" / "numerical"
    raw_path = results_dir / "paradigm_b_raw.json"
    
    print(f"\n{'='*70}")
    print(f"PSYCHOMETRIC FUNCTION FITTING — {model_key}")
    print(f"{'='*70}")
    
    # Load raw data
    with open(raw_path) as f:
        raw = json.load(f)
    
    b1_trials = [r for r in raw if r['task_type'] == 'B1']
    print(f"  B1 trials: {len(b1_trials)}")
    
    # Position-corrected accuracy
    cells = compute_position_corrected_accuracy(b1_trials)
    
    baselines = sorted(set(t['baseline'] for t in b1_trials))
    ratios = sorted(set(t['ratio'] for t in b1_trials))
    log_ratios = np.log(np.array(ratios))
    
    print(f"\n  Position-corrected accuracy:")
    print(f"  {'Baseline':>8s} | " + " | ".join(f"{r:.2f}" for r in ratios))
    print(f"  {'-'*8}-+-" + "-+-".join("-"*4 for _ in ratios))
    
    for bl in baselines:
        accs = []
        for r in ratios:
            cell = cells.get((bl, r))
            accs.append(f"{cell['acc_corrected']:.2f}" if cell else " -- ")
        print(f"  {bl:8.0f} | " + " | ".join(accs))
    
    # Fit psychometric functions per baseline
    print(f"\n  Psychometric fits:")
    fits = {}
    
    for bl in baselines:
        bl_cells = {r: cells[(bl, r)] for r in ratios if (bl, r) in cells}
        accs = np.array([bl_cells[r]['acc_corrected'] for r in ratios])
        ns = np.array([bl_cells[r]['n_total'] for r in ratios])
        
        fit = fit_psychometric(log_ratios, accs, ns)
        fits[str(bl)] = fit
        
        wf_str = f"{fit['weber_fraction']:.3f}" if fit['weber_fraction'] is not None else "N/A"
        print(f"    baseline {bl:5.0f}: μ={fit['mu']:.3f}, σ={fit['sigma']:.3f}, "
              f"λ={fit['lapse']:.3f}, WF={wf_str}")
    
    # Aggregate fit (all baselines pooled)
    print(f"\n  Aggregate fit (all baselines):")
    all_accs = []
    all_ns = []
    for r in ratios:
        ratio_cells = [(bl, r) for bl in baselines if (bl, r) in cells]
        if ratio_cells:
            acc = np.mean([cells[k]['acc_corrected'] for k in ratio_cells])
            n = sum(cells[k]['n_total'] for k in ratio_cells)
            all_accs.append(acc)
            all_ns.append(n)
    
    all_accs = np.array(all_accs)
    all_ns = np.array(all_ns)
    agg_fit = fit_psychometric(log_ratios, all_accs, all_ns)
    fits['aggregate'] = agg_fit
    
    agg_wf = f"{agg_fit['weber_fraction']:.3f}" if agg_fit['weber_fraction'] is not None else "N/A"
    print(f"    μ={agg_fit['mu']:.3f}, σ={agg_fit['sigma']:.3f}, "
          f"λ={agg_fit['lapse']:.3f}, WF={agg_wf}")
    
    # Bootstrap CIs
    print(f"\n  Bootstrap Weber fractions (1000 iterations):")
    bootstrap_results = {}
    
    for bl in baselines:
        bl_trials = [t for t in b1_trials if t['baseline'] == bl]
        boot = bootstrap_weber_fractions(bl_trials, n_bootstrap=1000, seed=42)
        bootstrap_results[str(bl)] = boot
        
        if boot['median'] is not None:
            print(f"    baseline {bl:5.0f}: WF = {boot['median']:.3f} "
                  f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] "
                  f"({boot['n_valid']}/1000 valid)")
        else:
            print(f"    baseline {bl:5.0f}: FAILED ({boot['n_valid']}/1000 valid)")
    
    # Aggregate bootstrap
    all_trials = b1_trials
    agg_boot = bootstrap_weber_fractions(all_trials, n_bootstrap=1000, seed=42)
    bootstrap_results['aggregate'] = agg_boot
    
    if agg_boot['median'] is not None:
        print(f"    AGGREGATE:      WF = {agg_boot['median']:.3f} "
              f"[{agg_boot['ci_low']:.3f}, {agg_boot['ci_high']:.3f}] "
              f"({agg_boot['n_valid']}/1000 valid)")
    
    # Weber's Law test: are Weber fractions constant across baselines?
    print(f"\n  Weber's Law consistency test:")
    valid_wfs = [(bl, fits[str(bl)]['weber_fraction']) 
                  for bl in baselines if fits[str(bl)]['weber_fraction'] is not None]
    
    if len(valid_wfs) >= 3:
        wf_values = [wf for _, wf in valid_wfs]
        wf_mean = np.mean(wf_values)
        wf_std = np.std(wf_values)
        wf_cv = wf_std / wf_mean if wf_mean > 0 else float('inf')
        print(f"    Valid baselines: {len(valid_wfs)}/5")
        print(f"    WF mean: {wf_mean:.3f}, std: {wf_std:.3f}, CV: {wf_cv:.3f}")
        print(f"    Human WF range: 0.10–0.25")
        print(f"    Within human range: {0.10 <= wf_mean <= 0.25}")
    else:
        print(f"    Insufficient valid fits ({len(valid_wfs)}/5)")
    
    # Save results
    output = {
        'model': model_key,
        'method': 'position_corrected_psychometric',
        'note': 'Position-bias corrected: accuracy averaged across A-correct and B-correct trials',
        'position_corrected_cells': {f"{bl},{r}": cells[(bl, r)] 
                                      for bl, r in cells},
        'psychometric_fits': fits,
        'bootstrap_weber_fractions': bootstrap_results,
        'aggregate_accuracy': {str(r): float(a) for r, a in zip(ratios, all_accs)},
    }
    
    out_path = results_dir / "psychometric_corrected.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved: {out_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Psychometric function fitting (position-corrected)")
    parser.add_argument("--model", choices=["llama", "mistral"], required=True)
    parser.add_argument("--project-root", type=str, default=r"C:\weber")
    args = parser.parse_args()
    
    run_psychometric_fitting(args.model, args.project_root)


if __name__ == "__main__":
    main()
