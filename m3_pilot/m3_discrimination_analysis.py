"""
M3 Paradigm B: Discrimination Analysis
=======================================
Analyses discrimination results from m3_discrimination.py.

Computes:
  1. d′ by position (cross-boundary vs within-category) × log-distance
  2. H2 test: d′ higher for cross-boundary than within-category, controlling for log-distance
  3. McMurray strict test (H2b): conditional on identification producing a crossover
  4. Meta-d′ preparation (Paradigm D): SDT operationalisation using Δlogit as evidence

Statistical tests:
  - Mixed-effects-style bootstrap: d′ difference with 10K bootstrap CIs
  - Log-distance controlled: compare cross vs within AT EACH distance level
  - McMurray: predicted d′ from identification vs observed d′

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# SDT functions for 2AFC
# ==============================================================================

def dprime_2afc(p_correct: float) -> float:
    """
    Compute d′ for 2AFC from proportion correct.
    
    d′ = √2 · Φ⁻¹(p_correct)
    
    Applies Hautus (2005) log-linear correction: if p = 0 or 1,
    use (0 + 0.5) / (N + 1) style correction.
    """
    # Clamp to avoid infinite d'
    p = max(0.001, min(0.999, p_correct))
    return math.sqrt(2) * stats.norm.ppf(p)


def dprime_from_trials(trials: list[dict]) -> float:
    """Compute d′ from a list of trial results."""
    if not trials:
        return 0.0
    p_correct = np.mean([t['p_correct_cb'] for t in trials])
    return dprime_2afc(p_correct)


def bootstrap_dprime_difference(
    cross_trials: list[dict],
    within_trials: list[dict],
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap the d′ difference (cross-boundary − within-category).
    
    Returns:
        dict with mean_diff, ci_95, p_value (proportion of bootstrap diffs ≤ 0)
    """
    rng = np.random.default_rng(seed)
    
    cross_correct = np.array([t['p_correct_cb'] for t in cross_trials])
    within_correct = np.array([t['p_correct_cb'] for t in within_trials])
    
    diffs = []
    for _ in range(n_boot):
        cross_boot = rng.choice(cross_correct, size=len(cross_correct), replace=True)
        within_boot = rng.choice(within_correct, size=len(within_correct), replace=True)
        
        d_cross = dprime_2afc(np.mean(cross_boot))
        d_within = dprime_2afc(np.mean(within_boot))
        diffs.append(d_cross - d_within)
    
    diffs = np.array(diffs)
    
    return {
        'mean_diff': round(float(np.mean(diffs)), 4),
        'ci_95_lower': round(float(np.percentile(diffs, 2.5)), 4),
        'ci_95_upper': round(float(np.percentile(diffs, 97.5)), 4),
        'p_value': round(float(np.mean(diffs <= 0)), 4),
        'n_boot': n_boot,
    }


# ==============================================================================
# Analysis functions
# ==============================================================================

def compute_cell_stats(trials: list[dict]) -> dict:
    """Compute accuracy and d′ for a cell of trials."""
    if not trials:
        return {'n': 0, 'accuracy': None, 'dprime': None}
    
    p_correct = np.mean([t['p_correct_cb'] for t in trials])
    d = dprime_2afc(p_correct)
    
    # Mean evidence and confidence
    mean_evidence = np.mean([t['evidence_cb'] for t in trials])
    mean_confidence = np.mean([t['confidence_cb'] for t in trials])
    
    return {
        'n': len(trials),
        'accuracy': round(float(p_correct), 4),
        'dprime': round(d, 4),
        'mean_evidence': round(float(mean_evidence), 4),
        'mean_confidence': round(float(mean_confidence), 4),
    }


def analyse_by_factorial(trials: list[dict]) -> dict:
    """
    Compute stats for the full 2 × 6 factorial.
    
    Returns dict keyed by position_collapsed × target_log_distance.
    """
    results = {}
    
    positions = ['cross_boundary', 'within_category']
    log_dists = sorted(set(t['target_log_distance'] for t in trials))
    
    for pos in positions:
        for ld in log_dists:
            subset = [t for t in trials 
                      if t['position_collapsed'] == pos and t['target_log_distance'] == ld]
            key = f"{pos}_logdist_{ld:.2f}"
            results[key] = compute_cell_stats(subset)
    
    # Marginals
    for pos in positions:
        subset = [t for t in trials if t['position_collapsed'] == pos]
        results[f"{pos}_marginal"] = compute_cell_stats(subset)
    
    for ld in log_dists:
        subset = [t for t in trials if t['target_log_distance'] == ld]
        results[f"all_logdist_{ld:.2f}"] = compute_cell_stats(subset)
    
    results['overall'] = compute_cell_stats(trials)
    
    return results


def test_h2(trials: list[dict], n_boot: int = 10000) -> dict:
    """
    Test H2: d′ higher for cross-boundary than within-category.
    
    Two analyses:
    1. Marginal: collapse across log-distances
    2. Distance-controlled: compare at each log-distance level
    """
    cross = [t for t in trials if t['position_collapsed'] == 'cross_boundary']
    within = [t for t in trials if t['position_collapsed'] == 'within_category']
    
    # 1. Marginal test
    marginal = bootstrap_dprime_difference(cross, within, n_boot=n_boot)
    
    # 2. Distance-controlled tests
    log_dists = sorted(set(t['target_log_distance'] for t in trials))
    distance_controlled = {}
    
    for ld in log_dists:
        cross_ld = [t for t in cross if t['target_log_distance'] == ld]
        within_ld = [t for t in within if t['target_log_distance'] == ld]
        
        if cross_ld and within_ld:
            distance_controlled[f"logdist_{ld:.2f}"] = bootstrap_dprime_difference(
                cross_ld, within_ld, n_boot=n_boot
            )
    
    # H2 verdict
    sig_marginal = marginal['ci_95_lower'] > 0
    n_sig_controlled = sum(
        1 for v in distance_controlled.values() if v['ci_95_lower'] > 0
    )
    
    return {
        'marginal': marginal,
        'distance_controlled': distance_controlled,
        'sig_marginal': sig_marginal,
        'n_sig_controlled': n_sig_controlled,
        'n_distance_levels': len(distance_controlled),
        'verdict': 'SUPPORTED' if sig_marginal else 'NOT_SUPPORTED',
    }


def test_h2_confidence(trials: list[dict], n_boot: int = 10000) -> dict:
    """
    Test H2 using confidence magnitude (|Δlogit|) instead of accuracy.
    
    Rationale: When accuracy is at ceiling, d′ is uninformative. But confidence
    magnitude (|Δlogit|) varies even at 100% accuracy — analogous to reaction
    time in human CP studies. If the model finds cross-boundary pairs "easier"
    (as CP predicts), |Δlogit| should be LARGER for cross-boundary than within.
    
    This is the RT-analogue test: higher confidence = faster/easier discrimination.
    """
    cross = [t for t in trials if t['position_collapsed'] == 'cross_boundary']
    within = [t for t in trials if t['position_collapsed'] == 'within_category']
    
    rng = np.random.default_rng(42)
    
    # Marginal test: bootstrap confidence difference
    cross_conf = np.array([t['confidence_cb'] for t in cross])
    within_conf = np.array([t['confidence_cb'] for t in within])
    
    observed_diff = float(np.mean(cross_conf) - np.mean(within_conf))
    
    diffs = []
    for _ in range(n_boot):
        c_boot = rng.choice(cross_conf, size=len(cross_conf), replace=True)
        w_boot = rng.choice(within_conf, size=len(within_conf), replace=True)
        diffs.append(np.mean(c_boot) - np.mean(w_boot))
    diffs = np.array(diffs)
    
    marginal = {
        'mean_cross_confidence': round(float(np.mean(cross_conf)), 4),
        'mean_within_confidence': round(float(np.mean(within_conf)), 4),
        'mean_diff': round(float(np.mean(diffs)), 4),
        'ci_95_lower': round(float(np.percentile(diffs, 2.5)), 4),
        'ci_95_upper': round(float(np.percentile(diffs, 97.5)), 4),
        'p_value': round(float(np.mean(diffs <= 0)), 4),
        'cohens_d': round(float(observed_diff / np.sqrt((np.var(cross_conf) + np.var(within_conf)) / 2)), 4)
            if (np.var(cross_conf) + np.var(within_conf)) > 0 else 0.0,
    }
    
    # Distance-controlled tests
    log_dists = sorted(set(t['target_log_distance'] for t in trials))
    distance_controlled = {}
    
    for ld in log_dists:
        cross_ld = [t['confidence_cb'] for t in cross if t['target_log_distance'] == ld]
        within_ld = [t['confidence_cb'] for t in within if t['target_log_distance'] == ld]
        
        if cross_ld and within_ld:
            cross_arr = np.array(cross_ld)
            within_arr = np.array(within_ld)
            
            ld_diffs = []
            for _ in range(n_boot):
                c = rng.choice(cross_arr, size=len(cross_arr), replace=True)
                w = rng.choice(within_arr, size=len(within_arr), replace=True)
                ld_diffs.append(np.mean(c) - np.mean(w))
            ld_diffs = np.array(ld_diffs)
            
            distance_controlled[f"logdist_{ld:.2f}"] = {
                'mean_cross': round(float(np.mean(cross_arr)), 4),
                'mean_within': round(float(np.mean(within_arr)), 4),
                'mean_diff': round(float(np.mean(ld_diffs)), 4),
                'ci_95_lower': round(float(np.percentile(ld_diffs, 2.5)), 4),
                'ci_95_upper': round(float(np.percentile(ld_diffs, 97.5)), 4),
            }
    
    # Also do Mann-Whitney U for a non-parametric check
    if len(cross_conf) > 0 and len(within_conf) > 0:
        u_stat, u_p = stats.mannwhitneyu(cross_conf, within_conf, alternative='greater')
        mann_whitney = {'U': round(float(u_stat), 2), 'p': round(float(u_p), 6)}
    else:
        mann_whitney = {'U': None, 'p': None}
    
    sig_marginal = marginal['ci_95_lower'] > 0
    n_sig = sum(1 for v in distance_controlled.values() if v['ci_95_lower'] > 0)
    
    return {
        'marginal': marginal,
        'distance_controlled': distance_controlled,
        'mann_whitney': mann_whitney,
        'sig_marginal': sig_marginal,
        'n_sig_controlled': n_sig,
        'n_distance_levels': len(distance_controlled),
        'verdict': 'SUPPORTED' if sig_marginal else 'NOT_SUPPORTED',
        'note': 'Confidence-based test (RT analogue). Positive diff = cross-boundary '
                'pairs elicit higher confidence, consistent with CP prediction.',
    }


def plot_confidence_curves(trials: list[dict], model_tag: str, out_dir: Path) -> None:
    """Plot confidence (|Δlogit|) by position × log-distance — the RT-analogue figure."""
    if not HAS_MATPLOTLIB:
        return
    
    log_dists = sorted(set(t['target_log_distance'] for t in trials))
    
    cross_conf_means = []
    within_conf_means = []
    cross_conf_sems = []
    within_conf_sems = []
    
    for ld in log_dists:
        cross = [t['confidence_cb'] for t in trials 
                 if t['position_collapsed'] == 'cross_boundary' and t['target_log_distance'] == ld]
        within = [t['confidence_cb'] for t in trials 
                  if t['position_collapsed'] == 'within_category' and t['target_log_distance'] == ld]
        
        cross_conf_means.append(np.mean(cross) if cross else 0)
        within_conf_means.append(np.mean(within) if within else 0)
        cross_conf_sems.append(stats.sem(cross) if len(cross) > 1 else 0)
        within_conf_sems.append(stats.sem(within) if len(within) > 1 else 0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.errorbar(log_dists, cross_conf_means, yerr=cross_conf_sems,
                fmt='o-', color='#e74c3c', label='Cross-boundary', linewidth=2, capsize=4)
    ax.errorbar(log_dists, within_conf_means, yerr=within_conf_sems,
                fmt='s-', color='#3498db', label='Within-category', linewidth=2, capsize=4)
    
    ax.set_xlabel('Log-distance')
    ax.set_ylabel('Confidence (|Δlogit|)')
    ax.set_title(f'Discrimination Confidence (RT analogue)\n{model_tag}')
    ax.legend()
    
    plt.tight_layout()
    out_path = out_dir / f'confidence_curves_{model_tag}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def prepare_metad_data(trials: list[dict], boundary: int = 10) -> dict:
    """
    Prepare data for Paradigm D (meta-d′) analysis.
    
    SDT operationalisation (pre-registered):
      - Type-1 decision: which is larger (correct/incorrect based on counterbalanced scoring)
      - Type-1 stimulus: cross-boundary (S2) vs within-category (S1)
      - Type-2 confidence: |Δlogit| (counterbalanced evidence magnitude)
    
    For meta-d′, we need:
      - nR_S1: confidence counts for "stimulus 1" (within-category) trials
      - nR_S2: confidence counts for "stimulus 2" (cross-boundary) trials
    
    Following M1 metadpy bin format: 2 × nRatings bins.
    """
    # Separate by position
    cross = [t for t in trials if t['position_collapsed'] == 'cross_boundary']
    within = [t for t in trials if t['position_collapsed'] == 'within_category']
    
    # Collect confidence values and correctness
    def extract_sdt_data(trial_list):
        correct = [t['p_correct_cb'] for t in trial_list]  # 0, 0.5, or 1
        confidence = [t['confidence_cb'] for t in trial_list]
        evidence = [t['evidence_cb'] for t in trial_list]
        return correct, confidence, evidence
    
    cross_correct, cross_conf, cross_ev = extract_sdt_data(cross)
    within_correct, within_conf, within_ev = extract_sdt_data(within)
    
    # Bin confidence into quartiles for metadpy format
    all_conf = cross_conf + within_conf
    if all_conf:
        quartiles = np.percentile(all_conf, [25, 50, 75])
    else:
        quartiles = [0, 0, 0]
    
    return {
        'cross_boundary': {
            'n': len(cross),
            'correct': cross_correct,
            'confidence': cross_conf,
            'evidence': cross_ev,
        },
        'within_category': {
            'n': len(within),
            'correct': within_correct,
            'confidence': within_conf,
            'evidence': within_ev,
        },
        'confidence_quartiles': [round(float(q), 4) for q in quartiles],
        'note': 'For meta-d analysis, use confidence quartiles as bin edges. '
                'nR_S1 = within-category counts, nR_S2 = cross-boundary counts.',
    }


# ==============================================================================
# Plotting
# ==============================================================================

def plot_discrimination_factorial(factorial_results: dict, model_tag: str, out_dir: Path) -> None:
    """Plot the 2 × 6 factorial: d′ by position × log-distance."""
    if not HAS_MATPLOTLIB:
        return
    
    log_dists = sorted(set(
        float(k.split('logdist_')[1]) 
        for k in factorial_results 
        if 'logdist_' in k and 'marginal' not in k and 'all_' not in k
    ))
    
    cross_dprime = []
    within_dprime = []
    
    for ld in log_dists:
        cross_key = f"cross_boundary_logdist_{ld:.2f}"
        within_key = f"within_category_logdist_{ld:.2f}"
        
        cd = factorial_results.get(cross_key, {}).get('dprime', None)
        wd = factorial_results.get(within_key, {}).get('dprime', None)
        
        cross_dprime.append(cd if cd is not None else 0)
        within_dprime.append(wd if wd is not None else 0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(log_dists))
    width = 0.35
    
    ax.bar(x - width/2, cross_dprime, width, label='Cross-boundary', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, within_dprime, width, label='Within-category', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Log-distance')
    ax.set_ylabel("d′")
    ax.set_title(f'Discrimination d′ by Position × Log-distance\n{model_tag}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{ld:.2f}' for ld in log_dists])
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    out_path = out_dir / f'discrimination_factorial_{model_tag}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_discrimination_curve(factorial_results: dict, model_tag: str, out_dir: Path) -> None:
    """Plot d′ as a function of log-distance for both conditions (line plot)."""
    if not HAS_MATPLOTLIB:
        return
    
    log_dists = sorted(set(
        float(k.split('logdist_')[1]) 
        for k in factorial_results 
        if 'logdist_' in k and 'marginal' not in k and 'all_' not in k
    ))
    
    cross_dprime = []
    within_dprime = []
    cross_acc = []
    within_acc = []
    
    for ld in log_dists:
        cross_key = f"cross_boundary_logdist_{ld:.2f}"
        within_key = f"within_category_logdist_{ld:.2f}"
        
        cr = factorial_results.get(cross_key, {})
        wr = factorial_results.get(within_key, {})
        
        cross_dprime.append(cr.get('dprime', 0))
        within_dprime.append(wr.get('dprime', 0))
        cross_acc.append(cr.get('accuracy', 0.5))
        within_acc.append(wr.get('accuracy', 0.5))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # d′ plot
    ax1.plot(log_dists, cross_dprime, 'o-', color='#e74c3c', label='Cross-boundary', linewidth=2)
    ax1.plot(log_dists, within_dprime, 's-', color='#3498db', label='Within-category', linewidth=2)
    ax1.set_xlabel('Log-distance')
    ax1.set_ylabel("d′")
    ax1.set_title(f"d′ by Position × Log-distance")
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Accuracy plot
    ax2.plot(log_dists, cross_acc, 'o-', color='#e74c3c', label='Cross-boundary', linewidth=2)
    ax2.plot(log_dists, within_acc, 's-', color='#3498db', label='Within-category', linewidth=2)
    ax2.set_xlabel('Log-distance')
    ax2.set_ylabel('P(correct)')
    ax2.set_title('Accuracy by Position × Log-distance')
    ax2.legend()
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, label='Chance')
    ax2.set_ylim(0.4, 1.05)
    
    fig.suptitle(f'Paradigm B: Behavioural Discrimination — {model_tag}', fontsize=12)
    plt.tight_layout()
    
    out_path = out_dir / f'discrimination_curves_{model_tag}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ==============================================================================
# Main analysis pipeline
# ==============================================================================

def analyse_model(results_path: Path, out_dir: Path, n_boot: int = 10000) -> dict:
    """Full analysis pipeline for one model."""
    
    with open(results_path) as f:
        data = json.load(f)
    
    model_tag = data['tag']
    trials = data['trials']
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {model_tag}")
    print(f"{'='*60}")
    print(f"Trials: {len(trials)}")
    
    # 1. Factorial analysis
    print(f"\n--- Factorial Analysis ---")
    factorial = analyse_by_factorial(trials)
    
    for key, val in sorted(factorial.items()):
        if val and val.get('dprime') is not None:
            print(f"  {key}: n={val['n']}, acc={val['accuracy']:.4f}, d′={val['dprime']:.4f}")
    
    # 2. H2 test
    print(f"\n--- H2 Test: Cross-boundary > Within-category ---")
    h2 = test_h2(trials, n_boot=n_boot)
    
    m = h2['marginal']
    print(f"  Marginal d′ diff: {m['mean_diff']:.4f} "
          f"[{m['ci_95_lower']:.4f}, {m['ci_95_upper']:.4f}]")
    print(f"  p-value: {m['p_value']:.4f}")
    print(f"  Verdict: {h2['verdict']}")
    
    print(f"\n  Distance-controlled:")
    for key, val in sorted(h2['distance_controlled'].items()):
        sig = '*' if val['ci_95_lower'] > 0 else ' '
        print(f"    {sig} {key}: diff={val['mean_diff']:.4f} "
              f"[{val['ci_95_lower']:.4f}, {val['ci_95_upper']:.4f}]")
    print(f"  Significant at {h2['n_sig_controlled']}/{h2['n_distance_levels']} distance levels")
    
    # 3. Confidence-based H2 test (RT analogue — key analysis when accuracy is at ceiling)
    print(f"\n--- H2 Confidence Test (RT analogue) ---")
    h2_conf = test_h2_confidence(trials, n_boot=n_boot)
    
    mc = h2_conf['marginal']
    print(f"  Mean confidence: cross={mc['mean_cross_confidence']:.4f}, "
          f"within={mc['mean_within_confidence']:.4f}")
    print(f"  Difference: {mc['mean_diff']:.4f} "
          f"[{mc['ci_95_lower']:.4f}, {mc['ci_95_upper']:.4f}]")
    print(f"  Cohen's d: {mc['cohens_d']:.4f}")
    mw = h2_conf['mann_whitney']
    print(f"  Mann-Whitney U={mw['U']}, p={mw['p']}")
    print(f"  Verdict: {h2_conf['verdict']}")
    
    print(f"\n  Distance-controlled:")
    for key, val in sorted(h2_conf['distance_controlled'].items()):
        sig = '*' if val['ci_95_lower'] > 0 else ' '
        print(f"    {sig} {key}: cross={val['mean_cross']:.4f}, "
              f"within={val['mean_within']:.4f}, diff={val['mean_diff']:.4f} "
              f"[{val['ci_95_lower']:.4f}, {val['ci_95_upper']:.4f}]")
    print(f"  Significant at {h2_conf['n_sig_controlled']}/{h2_conf['n_distance_levels']} distance levels")
    
    # 4. Meta-d′ preparation
    metad_data = prepare_metad_data(trials)
    
    # 5. Plots
    model_out_dir = out_dir / model_tag
    model_out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_discrimination_factorial(factorial, model_tag, model_out_dir)
    plot_discrimination_curve(factorial, model_tag, model_out_dir)
    plot_confidence_curves(trials, model_tag, model_out_dir)
    
    # Detect ceiling
    overall_acc = factorial.get('overall', {}).get('accuracy', 0)
    at_ceiling = overall_acc > 0.95
    
    # 6. Compile results
    analysis = {
        'model': data['model'],
        'tag': model_tag,
        'n_trials': len(trials),
        'at_ceiling': at_ceiling,
        'factorial': factorial,
        'h2_test': h2,
        'h2_confidence_test': h2_conf,
        'metad_preparation': {
            'cross_n': metad_data['cross_boundary']['n'],
            'within_n': metad_data['within_category']['n'],
            'confidence_quartiles': metad_data['confidence_quartiles'],
        },
    }
    
    # Save
    analysis_path = model_out_dir / f'm3_discrimination_analysis_{model_tag}.json'
    
    # Make JSON serialisable
    def make_serialisable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=make_serialisable)
    
    print(f"\n  Saved analysis: {analysis_path}")
    
    return analysis


def run_cross_model_summary(analyses: list[dict], out_dir: Path) -> None:
    """Print and save cross-model summary table."""
    
    print(f"\n{'='*60}")
    print(f"CROSS-MODEL SUMMARY")
    print(f"{'='*60}")
    
    # Accuracy-based table
    header = f"{'Model':<25} {'Acc':<8} {'d′(C)':<8} {'d′(W)':<8} {'Δd′':<8} {'H2(acc)':<12}"
    print(header)
    print('-' * len(header))
    
    rows = []
    for a in analyses:
        tag = a['tag']
        f = a['factorial']
        h2 = a['h2_test']
        h2c = a.get('h2_confidence_test', {})
        
        cross = f.get('cross_boundary_marginal', {})
        within = f.get('within_category_marginal', {})
        
        row = {
            'model': tag,
            'at_ceiling': a.get('at_ceiling', False),
            'acc_overall': f.get('overall', {}).get('accuracy', None),
            'acc_cross': cross.get('accuracy', None),
            'acc_within': within.get('accuracy', None),
            'dprime_cross': cross.get('dprime', None),
            'dprime_within': within.get('dprime', None),
            'dprime_diff': h2['marginal']['mean_diff'],
            'h2_acc_verdict': h2['verdict'],
            # Confidence-based
            'conf_cross': h2c.get('marginal', {}).get('mean_cross_confidence', None),
            'conf_within': h2c.get('marginal', {}).get('mean_within_confidence', None),
            'conf_diff': h2c.get('marginal', {}).get('mean_diff', None),
            'conf_cohens_d': h2c.get('marginal', {}).get('cohens_d', None),
            'h2_conf_verdict': h2c.get('verdict', None),
        }
        rows.append(row)
        
        acc = row['acc_overall'] or 0
        print(f"{tag:<25} {acc:.4f}  "
              f"{row['dprime_cross'] or 0:.4f}  {row['dprime_within'] or 0:.4f}  "
              f"{row['dprime_diff']:.4f}  {row['h2_acc_verdict']}")
    
    # Confidence-based table (the informative one for ceiling models)
    print(f"\n--- Confidence-based H2 (RT analogue) ---")
    header2 = f"{'Model':<25} {'Conf(C)':<10} {'Conf(W)':<10} {'ΔConf':<10} {'d':<8} {'H2(conf)':<12}"
    print(header2)
    print('-' * len(header2))
    
    for row in rows:
        conf_c = row['conf_cross'] or 0
        conf_w = row['conf_within'] or 0
        conf_d = row['conf_diff'] or 0
        d = row['conf_cohens_d'] or 0
        v = row['h2_conf_verdict'] or 'N/A'
        ceiling = ' [CEILING]' if row['at_ceiling'] else ''
        print(f"{row['model']:<25} {conf_c:<10.4f} {conf_w:<10.4f} {conf_d:<10.4f} {d:<8.4f} {v}{ceiling}")
    
    # Save summary
    summary_path = out_dir / 'discrimination_cross_model_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='M3 Paradigm B: Discrimination Analysis')
    parser.add_argument('--results-dir', type=str, default='discrimination_results',
                        help='Directory containing discrimination result JSONs')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for analysis results')
    parser.add_argument('--model', type=str, default=None,
                        help='Analyse specific model (by tag). Default: all found.')
    parser.add_argument('--n-boot', type=int, default=10000,
                        help='Number of bootstrap resamples')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Find result files
    if args.model:
        files = list(results_dir.glob(f'm3_discrimination_*{args.model}*.json'))
    else:
        files = sorted(results_dir.glob('m3_discrimination_*.json'))
    
    if not files:
        print(f"No result files found in {results_dir}")
        return
    
    print(f"Found {len(files)} result files")
    
    analyses = []
    for f in files:
        try:
            analysis = analyse_model(f, out_dir, n_boot=args.n_boot)
            analyses.append(analysis)
        except Exception as e:
            print(f"  ERROR analysing {f}: {e}")
            import traceback
            traceback.print_exc()
    
    # Cross-model summary
    if len(analyses) > 1:
        run_cross_model_summary(analyses, out_dir)


if __name__ == '__main__':
    main()
