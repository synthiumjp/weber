"""
M3 100-Boundary Analysis Runner
================================
Runs RSA analysis on decade_100 and control_150 using m3_pilot_analysis.py.

Usage:
  python m3_run_analysis_100.py
  python m3_run_analysis_100.py --perms 1000 --model llama3-8b-instruct

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import m3_pilot_analysis as analysis


MODELS = {
    'llama3-8b-instruct': {'primary_layers': (8, 25), 'heatmap_layer': 16},
    'mistral-7b-instruct': {'primary_layers': (8, 25), 'heatmap_layer': 16},
    'gemma2-9b-it': {'primary_layers': (11, 34), 'heatmap_layer': 22},
    'qwen25-7b-instruct': {'primary_layers': (7, 22), 'heatmap_layer': 14},
    'phi35-mini-instruct': {'primary_layers': (8, 25), 'heatmap_layer': 16},
    'llama3-8b-base': {'primary_layers': (8, 25), 'heatmap_layer': 16},
}

CONDITIONS = ['decade_100', 'control_150']


def analyse_condition(model_key, model_cfg, condition, n_perms, out_dir):
    """Run full RSA + precision analysis for one model × condition."""

    stim_path = Path(f'stimuli/m3_stimuli_{condition}.json')
    cent_path = Path(f'extractions/m3_centroids_{condition}_{model_key}.npz')

    if not stim_path.exists() or not cent_path.exists():
        print(f"  SKIP: missing files for {condition}")
        return None

    with open(stim_path) as f:
        stim = json.load(f)

    data = np.load(cent_path)
    rsa_centroids = data['rsa_centroids']
    values = np.array(stim['probing_values'])
    boundary = stim['boundary']
    primary_lo, primary_hi = model_cfg['primary_layers']
    heatmap_layer = model_cfg['heatmap_layer']

    print(f"\n  --- {condition} (boundary={boundary}) ---")
    print(f"  Values: {values}")
    print(f"  Centroids: {rsa_centroids.shape}")

    # 1. Empirical RDMs at all layers
    empirical_rdms = analysis.compute_rdms_all_layers(rsa_centroids, metric='cosine')
    print(f"  Empirical RDMs: {empirical_rdms.shape}")

    # 2. Theoretical RDMs
    theoretical_rdms = analysis.build_theoretical_rdms(values, boundary)
    for name, rdm in theoretical_rdms.items():
        print(f"    {name}: shape={rdm.shape}")

    # 3. RSA with Mantel tests
    print(f"  RSA ({n_perms} permutations)...")
    rsa_results = analysis.rsa_all_layers(
        empirical_rdms, theoretical_rdms, n_permutations=n_perms
    )

    # 4. Summarise primary layers
    primary_range = range(primary_lo, primary_hi)
    n_primary = len(list(primary_range))

    for model_name in ['continuous', 'cp_additive', 'categorical', 'linear']:
        if model_name in rsa_results and 'rho' in rsa_results[model_name]:
            rhos = rsa_results[model_name]['rho']
            primary_rhos = [rhos[l] for l in primary_range if l < len(rhos)]
            if primary_rhos:
                print(f"    {model_name:<15}: mean ρ = {np.mean(primary_rhos):.4f}, "
                      f"max ρ = {max(primary_rhos):.4f}")

    # CP advantage at primary layers
    cp_rhos = rsa_results.get('cp_additive', {}).get('rho', [])
    cont_rhos = rsa_results.get('continuous', {}).get('rho', [])

    if cp_rhos and cont_rhos:
        cp_wins = sum(1 for l in primary_range
                      if l < len(cp_rhos) and l < len(cont_rhos)
                      and cp_rhos[l] > cont_rhos[l])
        advantages = [cp_rhos[l] - cont_rhos[l]
                      for l in primary_range
                      if l < len(cp_rhos) and l < len(cont_rhos)]
        mean_adv = np.mean(advantages) if advantages else 0
    else:
        cp_wins = 0
        mean_adv = 0

    print(f"  CP-Additive > Continuous: {cp_wins}/{n_primary} layers")
    print(f"  Mean CP advantage: {mean_adv:+.4f}")

    # 5. Precision gradient
    prec = analysis.compute_precision_gradient(rsa_centroids, values)

    # 6. Plots
    model_out = out_dir / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    try:
        rdm_layer = min(heatmap_layer, empirical_rdms.shape[0] - 1)
        analysis.plot_rdm_heatmap(
            empirical_rdms[rdm_layer], values,
            f'{model_key} {condition} layer {rdm_layer}',
            model_out / f'rdm_heatmap_{condition}_layer{rdm_layer}.png'
        )
        print(f"  RDM heatmap saved")
    except Exception as e:
        print(f"  Plot error (RDM): {e}")

    try:
        analysis.plot_rsa_comparison(
            rsa_results, condition,
            model_out / f'rsa_comparison_{condition}.png',
            primary_layers=(primary_lo, primary_hi)
        )
        print(f"  RSA comparison saved")
    except Exception as e:
        print(f"  Plot error (RSA): {e}")

    try:
        analysis.plot_precision_gradient(
            prec, values, boundary, condition,
            model_out / f'precision_gradient_{condition}.png',
            primary_layers=(primary_lo, primary_hi)
        )
        print(f"  Precision gradient saved")
    except Exception as e:
        print(f"  Plot error (precision): {e}")

    return {
        'model': model_key,
        'condition': condition,
        'boundary': boundary,
        'n_values': len(values),
        'cp_wins': cp_wins,
        'n_primary': n_primary,
        'cp_advantage': round(float(mean_adv), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perms', type=int, default=10000)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)

    print(f"M3 100-Boundary Analysis — {args.perms} permutations")
    print('=' * 60)

    models = {args.model: MODELS[args.model]} if args.model else MODELS
    all_results = []

    for model_key, model_cfg in models.items():
        print(f"\n{'#'*60}")
        print(f"  {model_key}")
        print(f"{'#'*60}")
        t0 = time.time()

        for condition in CONDITIONS:
            r = analyse_condition(model_key, model_cfg, condition, args.perms, out_dir)
            if r:
                all_results.append(r)

        print(f"\n  Done in {time.time() - t0:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"100-BOUNDARY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Condition':<15} {'CP>Cont':<10} {'Advantage':<10}")
    print('-' * 60)
    for r in all_results:
        print(f"{r['model']:<25} {r['condition']:<15} "
              f"{r['cp_wins']}/{r['n_primary']:<7} {r['cp_advantage']:+.4f}")

    summary_path = out_dir / 'analysis_100_boundary_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == '__main__':
    main()
