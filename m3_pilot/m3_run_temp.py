"""
M3 Temperature Domain Analysis + Identification
================================================
Runs RSA analysis and counterbalanced identification for temp_hotcold
and temp_control conditions.

Uses pre-computed RDMs from stimulus files (handles negative temperatures
correctly, unlike build_theoretical_rdms which uses log transform).

Usage:
  python m3_run_temp.py --phase extract   # already done via m3_extract.py
  python m3_run_temp.py --phase analyse   # RSA analysis
  python m3_run_temp.py --phase identify  # counterbalanced identification
  python m3_run_temp.py --phase all       # analyse + identify

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# Analysis imports
import m3_pilot_analysis as analysis

# Identification imports
try:
    import torch
    from m3_rerun_identification import (
        ExtractionConfig,
        load_model_and_tokenizer,
        run_counterbalanced_trial,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


MODELS = {
    'llama3-8b-instruct': {'primary_layers': (8, 25), 'hf_id': 'meta-llama/Meta-Llama-3-8B-Instruct', 'precision': 'float16', 'chat': True},
    'mistral-7b-instruct': {'primary_layers': (8, 25), 'hf_id': 'mistralai/Mistral-7B-Instruct-v0.3', 'precision': 'float16', 'chat': True},
    'gemma2-9b-it': {'primary_layers': (11, 34), 'hf_id': 'google/gemma-2-9b-it', 'precision': 'bfloat16', 'chat': True},
    'qwen25-7b-instruct': {'primary_layers': (7, 22), 'hf_id': 'Qwen/Qwen2.5-7B-Instruct', 'precision': 'float16', 'chat': True},
    'phi35-mini-instruct': {'primary_layers': (8, 25), 'hf_id': 'Lexius/Phi-3.5-mini-instruct', 'precision': 'float16', 'chat': True},
    'llama3-8b-base': {'primary_layers': (8, 25), 'hf_id': 'meta-llama/Meta-Llama-3-8B', 'precision': 'float16', 'chat': False},
}

CONDITIONS = ['temp_hotcold', 'temp_control']

IDENTIFICATION_FRAMINGS = {
    'hot_cold': {
        'question': 'Is {T} degrees Celsius hot or cold?',
        'category_a_label': 'cold',
        'category_b_label': 'hot',
    },
    'warm_cool': {
        'question': 'Is {T} degrees Celsius warm or cool?',
        'category_a_label': 'cool',
        'category_b_label': 'warm',
    },
}


# ==============================================================================
# RSA Analysis
# ==============================================================================

def run_rsa_analysis(n_perms=10000, model_filter=None):
    """Run RSA on temperature conditions using stimulus-file RDMs."""
    
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    
    models = {model_filter: MODELS[model_filter]} if model_filter else MODELS
    all_results = []
    
    for model_key, cfg in models.items():
        print(f"\n{'#'*60}")
        print(f"  {model_key}")
        print(f"{'#'*60}")
        t0 = time.time()
        
        for condition in CONDITIONS:
            stim_path = Path(f'stimuli/m3_stimuli_{condition}.json')
            cent_path = Path(f'extractions/m3_centroids_{condition}_{model_key}.npz')
            
            if not stim_path.exists() or not cent_path.exists():
                print(f"  SKIP {condition}: missing files")
                continue
            
            with open(stim_path) as f:
                stim = json.load(f)
            
            rsa_centroids = np.load(cent_path)['rsa_centroids']
            values = np.array(stim['probing_values'])
            boundary = stim['boundary']
            primary_lo, primary_hi = cfg['primary_layers']
            
            print(f"\n  --- {condition} (boundary={boundary}°C) ---")
            print(f"  Values: {values}")
            print(f"  Centroids: {rsa_centroids.shape}")
            
            # Use pre-computed RDMs from stimulus file
            rdms = {}
            for key in stim['theoretical_rdms']:
                val = stim['theoretical_rdms'][key]
                if isinstance(val, list):
                    rdms[key] = np.array(val)
            
            # Compute empirical RDMs
            emp = analysis.compute_rdms_all_layers(rsa_centroids)
            print(f"  Empirical RDMs: {emp.shape}")
            
            # RSA
            print(f"  RSA ({n_perms} permutations)...")
            results = analysis.rsa_all_layers(emp, rdms, n_permutations=n_perms)
            
            # Summarise
            primary_range = range(primary_lo, primary_hi)
            n_primary = len(list(primary_range))
            
            for name in ['continuous', 'cp_additive', 'categorical', 'linear']:
                if name in results and 'rho' in results[name]:
                    rhos = results[name]['rho']
                    pr = [rhos[l] for l in primary_range if l < len(rhos)]
                    if pr:
                        print(f"    {name:<15}: mean ρ = {np.mean(pr):.4f}, max ρ = {max(pr):.4f}")
            
            # CP advantage
            cp_rhos = results.get('cp_additive', {}).get('rho', [])
            cont_rhos = results.get('continuous', {}).get('rho', [])
            
            if cp_rhos and cont_rhos:
                wins = sum(1 for l in primary_range
                           if l < len(cp_rhos) and l < len(cont_rhos)
                           and cp_rhos[l] > cont_rhos[l])
                advs = [cp_rhos[l] - cont_rhos[l] for l in primary_range
                        if l < len(cp_rhos) and l < len(cont_rhos)]
                mean_adv = np.mean(advs) if advs else 0
            else:
                wins = 0
                mean_adv = 0
            
            print(f"  CP-Additive > Continuous: {wins}/{n_primary} layers")
            print(f"  Mean CP advantage: {mean_adv:+.4f}")
            
            # Also check log-based models if available
            for alt_name in ['continuous_log', 'cp_additive_log']:
                if alt_name in results and 'rho' in results[alt_name]:
                    rhos = results[alt_name]['rho']
                    pr = [rhos[l] for l in primary_range if l < len(rhos)]
                    if pr:
                        print(f"    {alt_name:<15}: mean ρ = {np.mean(pr):.4f}")
            
            # Plots
            model_out = out_dir / model_key
            model_out.mkdir(parents=True, exist_ok=True)
            
            try:
                hm_layer = min(cfg['primary_layers'][0] + 8, emp.shape[0] - 1)
                analysis.plot_rdm_heatmap(
                    emp[hm_layer], values,
                    f'{model_key} {condition} layer {hm_layer}',
                    model_out / f'rdm_heatmap_{condition}_layer{hm_layer}.png'
                )
            except Exception as e:
                print(f"  Plot error: {e}")
            
            try:
                analysis.plot_rsa_comparison(
                    results, condition,
                    model_out / f'rsa_comparison_{condition}.png',
                    primary_layers=(primary_lo, primary_hi)
                )
            except Exception as e:
                print(f"  Plot error: {e}")
            
            all_results.append({
                'model': model_key,
                'condition': condition,
                'boundary': boundary,
                'cp_wins': wins,
                'n_primary': n_primary,
                'cp_advantage': round(float(mean_adv), 4),
            })
        
        print(f"\n  Done in {time.time() - t0:.0f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEMPERATURE RSA SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Condition':<15} {'CP>Cont':<10} {'Advantage':<10}")
    print('-' * 60)
    for r in all_results:
        print(f"{r['model']:<25} {r['condition']:<15} "
              f"{r['cp_wins']}/{r['n_primary']:<7} {r['cp_advantage']:+.4f}")
    
    summary_path = out_dir / 'analysis_temperature_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {summary_path}")
    
    return all_results


# ==============================================================================
# Counterbalanced Identification
# ==============================================================================

def run_identification(model_filter=None):
    """Run counterbalanced identification for temperature conditions."""
    
    if not HAS_TORCH:
        print("ERROR: torch not available")
        return
    
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    
    models_to_run = {model_filter: MODELS[model_filter]} if model_filter else MODELS
    
    for model_key, cfg in models_to_run.items():
        if not cfg['chat']:
            print(f"\n  SKIP {model_key} — no chat template")
            continue
        
        print(f"\n{'#'*60}")
        print(f"  {model_key}")
        print(f"{'#'*60}")
        
        config = ExtractionConfig()
        config.model_name = cfg['hf_id']
        config.model_short = model_key
        config.precision = cfg['precision']
        
        print(f"  Loading: {cfg['hf_id']}")
        model, tokenizer = load_model_and_tokenizer(config)
        
        for condition in CONDITIONS:
            meta_path = Path(f'extractions/m3_meta_{condition}_{model_key}.json')
            if not meta_path.exists():
                print(f"  SKIP {condition}: no metadata")
                continue
            
            with open(meta_path) as f:
                meta = json.load(f)
            
            values = meta['values']
            print(f"\n  {'='*55}")
            print(f"  {condition} — {len(values)} values × {len(IDENTIFICATION_FRAMINGS)} framings")
            print(f"  {'='*55}")
            
            results = []
            t0 = time.time()
            
            for framing_key, spec in IDENTIFICATION_FRAMINGS.items():
                print(f"\n  Framing: {framing_key}")
                print(f"  {'Value':>6s} | {'O1 P(B)':>8s} | {'O2 P(B)':>8s} | "
                      f"{'Avg P(B)':>8s} | {'Bias':>7s} | Category")
                print(f"  {'-'*68}")
                
                for v in values:
                    question = spec['question'].format(T=str(v))
                    trial = run_counterbalanced_trial(
                        model, tokenizer, question,
                        spec['category_a_label'], spec['category_b_label'],
                        config.device,
                    )
                    
                    choice = (spec['category_b_label']
                              if trial['prob_category_b'] > 0.5
                              else spec['category_a_label'])
                    
                    print(f"  {v:6d} | {trial['order1_p_catb']:.4f}   | "
                          f"{trial['order2_p_catb']:.4f}   | "
                          f"{trial['prob_category_b']:.4f}   | "
                          f"{trial['position_bias']:+.4f}  | {choice}")
                    
                    results.append({
                        'value': v,
                        'framing': framing_key,
                        'prob_category_a': trial['prob_category_a'],
                        'prob_category_b': trial['prob_category_b'],
                        'order1_p_catb': trial['order1_p_catb'],
                        'order2_p_catb': trial['order2_p_catb'],
                        'position_bias': trial['position_bias'],
                        'option_a_label': spec['category_a_label'],
                        'option_b_label': spec['category_b_label'],
                    })
            
            elapsed = time.time() - t0
            
            meta['identification_results_temp'] = results
            meta['identification_method_temp'] = 'chat_template_ab_counterbalanced'
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"\n  Updated: {meta_path} ({elapsed:.1f}s)")
        
        del model
        torch.cuda.empty_cache()
        print(f"  GPU cleared.")


def main():
    parser = argparse.ArgumentParser(description='M3 Temperature Domain Pipeline')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['analyse', 'identify', 'all'])
    parser.add_argument('--perms', type=int, default=10000)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    
    if args.phase in ('analyse', 'all'):
        print("=" * 60)
        print("PHASE 1: RSA ANALYSIS")
        print("=" * 60)
        run_rsa_analysis(n_perms=args.perms, model_filter=args.model)
    
    if args.phase in ('identify', 'all'):
        print("\n" + "=" * 60)
        print("PHASE 2: COUNTERBALANCED IDENTIFICATION")
        print("=" * 60)
        run_identification(model_filter=args.model)


if __name__ == '__main__':
    main()
