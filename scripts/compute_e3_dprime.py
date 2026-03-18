#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — E3: Bridge to SDT (d' Computation)
==============================================================
Computes d' (sensitivity) for magnitude comparison at each baseline level,
bridging to Project 4.1 (SDT Calibration).

If Weber's Law holds, d' should be constant when the comparison ratio is constant,
regardless of baseline magnitude. This is the SDT analogue of Weber fraction constancy.

No GPU needed — reads from existing paradigm_b_raw.json.

Usage: python scripts/compute_e3_dprime.py [--results-dir C:\\weber\\results]

Pre-registration ref: v2.7 E3, Section 6 (Bridge to SDT)
"""

import json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import norm

MODELS = ['llama_instruct', 'mistral_instruct']
BASELINES = [10, 30, 100, 300, 1000]
RATIOS = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]


def compute_dprime(hit_rate, false_alarm_rate=None):
    """
    Compute d' from hit rate in a 2AFC task.
    For 2AFC (forced choice between two alternatives), d' = sqrt(2) * z(accuracy).
    This assumes equal-variance Gaussian signal and noise distributions.
    
    Accuracy is clipped to [0.001, 0.999] to avoid infinite z-scores.
    """
    # For 2AFC, accuracy = Φ(d'/√2), so d' = √2 * Φ⁻¹(accuracy)
    acc_clipped = np.clip(hit_rate, 0.001, 0.999)
    return np.sqrt(2) * norm.ppf(acc_clipped)


def process_model(results_dir, model):
    """Compute d' at each baseline × ratio cell from B1 raw data."""
    raw_path = Path(results_dir) / 'paradigm_b' / model / 'numerical' / 'paradigm_b_raw.json'
    
    if not raw_path.exists():
        print(f"  [SKIP] {raw_path} not found")
        return None

    with open(raw_path) as f:
        raw = json.load(f)
    
    items = raw if isinstance(raw, list) else raw.get('items', [])
    b1_items = [it for it in items if it.get('task_type') == 'B1']
    print(f"  {model}: {len(b1_items)} B1 items")

    # Group by baseline × ratio
    cells = defaultdict(list)
    for it in b1_items:
        bl = it.get('baseline', it.get('nominal_baseline'))
        rt = it.get('ratio', it.get('nominal_ratio'))
        correct = it.get('correct', it.get('is_correct', False))
        if bl is None or rt is None:
            continue
        bl_snap = min(BASELINES, key=lambda b: abs(b - bl))
        rt_snap = min(RATIOS, key=lambda r: abs(r - rt))
        cells[(bl_snap, rt_snap)].append(1 if correct else 0)

    # Compute d' per cell
    results = {
        'model': model,
        'method': '2AFC d-prime: sqrt(2) * Phi_inv(accuracy)',
        'note': 'Weber prediction: d-prime should be constant across baselines at each ratio level',
        'per_cell': {},
        'per_ratio': {},
        'per_baseline': {},
    }

    # Per-cell d'
    for (bl, rt), vals in sorted(cells.items()):
        acc = np.mean(vals)
        dp = compute_dprime(acc)
        key = f'{bl}_{rt}'
        results['per_cell'][key] = {
            'baseline': bl,
            'ratio': rt,
            'n_trials': len(vals),
            'accuracy': round(acc, 4),
            'dprime': round(float(dp), 4),
        }

    # Aggregate by ratio (averaging d' across baselines — Weber prediction: should be equal)
    by_ratio = defaultdict(list)
    for (bl, rt), vals in cells.items():
        acc = np.mean(vals)
        dp = compute_dprime(acc)
        by_ratio[rt].append(dp)

    for rt in sorted(by_ratio.keys()):
        dps = by_ratio[rt]
        results['per_ratio'][str(rt)] = {
            'ratio': rt,
            'mean_dprime': round(float(np.mean(dps)), 4),
            'std_dprime': round(float(np.std(dps)), 4),
            'cv': round(float(np.std(dps) / np.mean(dps)) if np.mean(dps) != 0 else np.nan, 4),
            'n_baselines': len(dps),
            'dprime_values': [round(float(d), 4) for d in dps],
        }

    # Aggregate by baseline (should show same d' pattern at each baseline if Weber holds)
    by_baseline = defaultdict(list)
    for (bl, rt), vals in cells.items():
        acc = np.mean(vals)
        dp = compute_dprime(acc)
        by_baseline[bl].append((rt, dp))

    for bl in sorted(by_baseline.keys()):
        pairs = sorted(by_baseline[bl])
        results['per_baseline'][str(bl)] = {
            'baseline': bl,
            'ratios': [p[0] for p in pairs],
            'dprimes': [round(float(p[1]), 4) for p in pairs],
            'mean_dprime': round(float(np.mean([p[1] for p in pairs])), 4),
        }

    # Weber constancy test: coefficient of variation of d' across baselines at each ratio
    # Low CV = Weber's Law holds (d' is constant across baselines for the same ratio)
    cv_by_ratio = []
    for rt in sorted(by_ratio.keys()):
        dps = by_ratio[rt]
        cv = float(np.std(dps) / np.mean(dps)) if np.mean(dps) != 0 else np.nan
        cv_by_ratio.append(cv)
    
    results['weber_constancy'] = {
        'mean_cv_across_ratios': round(float(np.nanmean(cv_by_ratio)), 4),
        'interpretation': 'Low CV = d-prime is constant across baselines (Weber holds). '
                          'High CV = d-prime varies with baseline (absolute difference dominates).',
    }

    # Print summary
    print(f"  d' by ratio:")
    for rt in sorted(by_ratio.keys()):
        dps = by_ratio[rt]
        print(f"    ratio {rt:.2f}: mean d' = {np.mean(dps):.3f} (SD = {np.std(dps):.3f})")
    print(f"  Weber constancy (mean CV): {np.nanmean(cv_by_ratio):.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default=r'C:\weber\results')
    args = parser.parse_args()

    print("=" * 60)
    print("E3: Bridge to SDT — d' Computation")
    print("=" * 60)

    all_results = {}
    for model in MODELS:
        print(f"\n--- {model} ---")
        result = process_model(args.results_dir, model)
        if result is not None:
            all_results[model] = result

    # Save
    out_path = Path(args.results_dir) / 'e3_bridge_to_sdt.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 60)
    print("DONE. d' values saved. Add to paper as E3 exploratory analysis.")
    print("=" * 60)


if __name__ == '__main__':
    main()
