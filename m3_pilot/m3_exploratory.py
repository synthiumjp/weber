"""
M3 Exploratory Analyses (E1, E3, E4, E7, E8, E9, E11)
======================================================
All computed from existing centroids and metadata. No new extraction needed.

E1:  Layerwise CP profile — where does CP emerge across depth?
E3:  Boundary sharpness — sigmoid fit to identification functions
E4:  CP × magnitude interaction — decade_100 vs decade_10 effect sizes
E7:  Local manifold analysis — PCA around boundary, dimensionality, rotation
E8:  Phase-reset analysis — hidden-state discontinuity at boundary
E9:  Identification slope predicts CP strength — cross-model correlation
E11: CP as local violation of compression — λ vs global fit anticorrelation

Usage:
  python m3_exploratory.py
  python m3_exploratory.py --model llama3-8b-instruct

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform, cosine

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import m3_pilot_analysis as analysis


MODELS = {
    'llama3-8b-instruct': {'primary': (8, 25), 'role': 'primary'},
    'mistral-7b-instruct': {'primary': (8, 25), 'role': 'primary'},
    'gemma2-9b-it': {'primary': (11, 34), 'role': 'primary'},
    'qwen25-7b-instruct': {'primary': (7, 22), 'role': 'primary'},
    'phi35-mini-instruct': {'primary': (8, 25), 'role': 'primary'},
    'llama3-8b-base': {'primary': (8, 25), 'role': 'exploratory'},
}

NUMERICAL_CONDITIONS = ['decade_10', 'decade_100']
CONTROL_CONDITIONS = ['control_15', 'control_150']
BOUNDARIES = {'decade_10': 10, 'decade_100': 100, 'control_15': 15, 'control_150': 150}


def load_rdms(d):
    """Load theoretical RDMs from stimulus data, handling both formats."""
    rdm_source = d['stim'].get('theoretical_rdms', {})
    rdms = {}
    for key in rdm_source:
        val = rdm_source[key]
        if isinstance(val, list):
            arr = np.array(val)
            if arr.ndim == 2:
                rdms[key] = arr
    
    if not rdms and d['values'] is not None and d['boundary'] is not None:
        try:
            rdms = analysis.build_theoretical_rdms(d['values'], d['boundary'])
        except Exception:
            pass
    
    # Normalise key names (old format uses Title Case)
    key_map = {'Continuous': 'continuous', 'CP-Additive': 'cp_additive',
               'Categorical': 'categorical', 'Linear': 'linear'}
    normalised = {}
    for k, v in rdms.items():
        nk = key_map.get(k, k)
        normalised[nk] = v if isinstance(v, np.ndarray) else np.array(v)
    return normalised if normalised else rdms


def load_data(model, condition):
    """Load centroids, stimuli, and metadata for a model×condition."""
    cent_path = Path(f'extractions/m3_centroids_{condition}_{model}.npz')
    stim_path = Path(f'stimuli/m3_stimuli_{condition}.json')
    meta_path = Path(f'extractions/m3_meta_{condition}_{model}.json')
    
    if not cent_path.exists():
        return None
    
    data = np.load(cent_path)
    rsa_centroids = data['rsa_centroids']
    
    with open(stim_path) as f:
        stim = json.load(f)
    
    # Handle both stimulus file formats:
    # New format (100-boundary, temperature): has 'boundary' key directly
    # Old format (decade_10, control_15): has 'metadata' dict with 'boundary'
    if 'boundary' in stim:
        boundary = stim['boundary']
    elif 'metadata' in stim and 'boundary' in stim['metadata']:
        boundary = stim['metadata']['boundary']
    else:
        boundary = BOUNDARIES.get(condition, None)
    
    meta = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    
    return {
        'centroids': rsa_centroids,
        'values': np.array(stim['probing_values']),
        'boundary': boundary,
        'stim': stim,
        'meta': meta,
    }


# ==============================================================================
# E1: Layerwise CP Profile
# ==============================================================================

def e1_layerwise_profile(model, out_dir):
    """CP advantage (CP-Additive ρ − Continuous ρ) at each layer."""
    print(f"\n  --- E1: Layerwise CP Profile ---")
    
    results = {}
    for condition in NUMERICAL_CONDITIONS:
        d = load_data(model, condition)
        if d is None:
            continue
        
        emp = analysis.compute_rdms_all_layers(d['centroids'])
        rdms = load_rdms(d)
        
        rsa = analysis.rsa_all_layers(emp, rdms, n_permutations=1000)
        
        cp_rhos = rsa.get('cp_additive', {}).get('rho', [])
        cont_rhos = rsa.get('continuous', {}).get('rho', [])
        
        if cp_rhos and cont_rhos:
            n_layers = min(len(cp_rhos), len(cont_rhos))
            advantages = [cp_rhos[l] - cont_rhos[l] for l in range(n_layers)]
            results[condition] = {
                'cp_rhos': cp_rhos[:n_layers],
                'cont_rhos': cont_rhos[:n_layers],
                'advantages': advantages,
                'peak_layer': int(np.argmax(advantages)),
                'peak_advantage': round(float(max(advantages)), 4),
            }
            print(f"    {condition}: peak at layer {results[condition]['peak_layer']} "
                  f"(Δρ = {results[condition]['peak_advantage']:.4f})")
    
    return results


# ==============================================================================
# E3: Boundary Sharpness (Sigmoid Fit)
# ==============================================================================

def sigmoid(x, a, b, c, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return d + (c - d) / (1.0 + np.exp(-a * (x - b)))


def e3_boundary_sharpness(model, out_dir):
    """Fit sigmoid to identification functions, extract slope and crossover."""
    print(f"\n  --- E3: Boundary Sharpness ---")
    
    results = {}
    
    for condition in NUMERICAL_CONDITIONS + ['temp_hotcold']:
        meta_path = Path(f'extractions/m3_meta_{condition}_{model}.json')
        if not meta_path.exists():
            continue
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        # Try multiple identification result keys
        id_results = (meta.get('identification_results_temp') or
                      meta.get('identification_results_100') or
                      meta.get('identification_results') or [])
        
        if not id_results:
            continue
        
        # Group by framing
        framings = {}
        for r in id_results:
            fr = r.get('framing', 'unknown')
            if fr not in framings:
                framings[fr] = {'values': [], 'probs': []}
            framings[fr]['values'].append(r['value'])
            framings[fr]['probs'].append(r.get('prob_category_b', 0.5))
        
        condition_results = {}
        for fr_name, fr_data in framings.items():
            vals = np.array(fr_data['values'], dtype=float)
            probs = np.array(fr_data['probs'])
            
            try:
                popt, _ = curve_fit(sigmoid, vals, probs,
                                    p0=[1.0, np.median(vals), 1.0, 0.0],
                                    maxfev=5000)
                slope, crossover, upper, lower = popt
                predicted = sigmoid(vals, *popt)
                r_squared = 1 - np.sum((probs - predicted)**2) / np.sum((probs - np.mean(probs))**2)
                
                condition_results[fr_name] = {
                    'slope': round(float(slope), 4),
                    'crossover': round(float(crossover), 2),
                    'r_squared': round(float(r_squared), 4),
                    'upper_asymptote': round(float(upper), 4),
                    'lower_asymptote': round(float(lower), 4),
                }
                print(f"    {condition}/{fr_name}: slope={slope:.3f}, "
                      f"crossover={crossover:.1f}, R²={r_squared:.3f}")
            except Exception as e:
                condition_results[fr_name] = {'error': str(e)}
                print(f"    {condition}/{fr_name}: fit failed ({e})")
        
        results[condition] = condition_results
    
    return results


# ==============================================================================
# E4: CP × Magnitude Interaction
# ==============================================================================

def e4_magnitude_interaction(model):
    """Compare CP effect sizes at decade_10 vs decade_100."""
    print(f"\n  --- E4: CP × Magnitude Interaction ---")
    
    primary = MODELS[model]['primary']
    effects = {}
    
    for condition in NUMERICAL_CONDITIONS:
        d = load_data(model, condition)
        if d is None:
            continue
        
        emp = analysis.compute_rdms_all_layers(d['centroids'])
        rdms = load_rdms(d)
        
        rsa = analysis.rsa_all_layers(emp, rdms, n_permutations=100)  # quick
        
        cp_rhos = rsa.get('cp_additive', {}).get('rho', [])
        cont_rhos = rsa.get('continuous', {}).get('rho', [])
        
        if cp_rhos and cont_rhos:
            advs = [cp_rhos[l] - cont_rhos[l] 
                    for l in range(primary[0], primary[1])
                    if l < len(cp_rhos) and l < len(cont_rhos)]
            effects[condition] = {
                'mean_advantage': round(float(np.mean(advs)), 4),
                'max_advantage': round(float(max(advs)), 4),
            }
    
    if 'decade_10' in effects and 'decade_100' in effects:
        ratio = effects['decade_100']['mean_advantage'] / max(effects['decade_10']['mean_advantage'], 0.001)
        print(f"    decade_10:  Δρ = {effects['decade_10']['mean_advantage']:+.4f}")
        print(f"    decade_100: Δρ = {effects['decade_100']['mean_advantage']:+.4f}")
        print(f"    Ratio (100/10): {ratio:.1f}×")
        effects['ratio'] = round(ratio, 2)
    
    return effects


# ==============================================================================
# E7: Local Manifold Analysis
# ==============================================================================

def e7_local_manifold(model, out_dir):
    """PCA around boundary region — rotation, dimensionality change."""
    print(f"\n  --- E7: Local Manifold Analysis ---")
    
    results = {}
    primary = MODELS[model]['primary']
    # Use middle of primary range
    target_layer = (primary[0] + primary[1]) // 2
    
    for condition in NUMERICAL_CONDITIONS:
        d = load_data(model, condition)
        if d is None:
            continue
        
        values = d['values']
        boundary = d['boundary']
        centroids = d['centroids']  # (n_values, n_layers, d_model)
        
        # Get centroids at target layer
        layer_cents = centroids[:, target_layer, :]  # (n_values, d_model)
        
        # Split into below and above boundary
        below_mask = values < boundary
        above_mask = values >= boundary
        below_cents = layer_cents[below_mask]
        above_cents = layer_cents[above_mask]
        
        # PCA on each side
        from numpy.linalg import svd
        
        def local_pca(X, n_components=5):
            X_centered = X - X.mean(axis=0)
            U, S, Vt = svd(X_centered, full_matrices=False)
            explained_var = (S**2) / (S**2).sum()
            return Vt[:n_components], explained_var[:n_components], S[:n_components]
        
        below_pcs, below_var, below_sv = local_pca(below_cents)
        above_pcs, above_var, above_sv = local_pca(above_cents)
        
        # PC1 rotation angle between below and above
        cos_angle = abs(np.dot(below_pcs[0], above_pcs[0]))
        angle_deg = np.degrees(np.arccos(min(cos_angle, 1.0)))
        
        # Effective dimensionality (participation ratio)
        def participation_ratio(eigenvalues):
            ev = eigenvalues / eigenvalues.sum()
            return 1.0 / (ev**2).sum()
        
        below_pr = participation_ratio(below_sv**2)
        above_pr = participation_ratio(above_sv**2)
        
        results[condition] = {
            'layer': target_layer,
            'pc1_rotation_deg': round(float(angle_deg), 2),
            'below_var_explained_pc1': round(float(below_var[0]), 4),
            'above_var_explained_pc1': round(float(above_var[0]), 4),
            'below_effective_dim': round(float(below_pr), 2),
            'above_effective_dim': round(float(above_pr), 2),
            'n_below': int(below_mask.sum()),
            'n_above': int(above_mask.sum()),
        }
        
        print(f"    {condition} (layer {target_layer}):")
        print(f"      PC1 rotation at boundary: {angle_deg:.1f}°")
        print(f"      Var(PC1): below={below_var[0]:.3f}, above={above_var[0]:.3f}")
        print(f"      Eff. dim: below={below_pr:.1f}, above={above_pr:.1f}")
    
    return results


# ==============================================================================
# E8: Phase-Reset Analysis
# ==============================================================================

def e8_phase_reset(model, out_dir):
    """Test for hidden-state discontinuity at boundary (cosine jump)."""
    print(f"\n  --- E8: Phase-Reset Analysis ---")
    
    results = {}
    primary = MODELS[model]['primary']
    
    for condition in NUMERICAL_CONDITIONS:
        d = load_data(model, condition)
        if d is None:
            continue
        
        values = d['values']
        boundary = d['boundary']
        centroids = d['centroids']
        
        # Compute consecutive cosine distances at each layer
        n_values = len(values)
        n_layers = centroids.shape[1]
        
        # Find boundary index (where values cross boundary)
        boundary_idx = None
        for i in range(n_values - 1):
            if values[i] < boundary and values[i+1] >= boundary:
                boundary_idx = i
                break
        
        if boundary_idx is None:
            continue
        
        # Compute cosine distances between consecutive values at primary layers
        layer_jumps = []
        for layer in range(primary[0], primary[1]):
            consecutive_dists = []
            for i in range(n_values - 1):
                d_cos = cosine(centroids[i, layer], centroids[i+1, layer])
                consecutive_dists.append(d_cos)
            
            boundary_dist = consecutive_dists[boundary_idx]
            non_boundary_dists = [consecutive_dists[j] for j in range(len(consecutive_dists)) 
                                  if j != boundary_idx]
            mean_non_boundary = np.mean(non_boundary_dists)
            
            layer_jumps.append({
                'layer': layer,
                'boundary_dist': boundary_dist,
                'mean_non_boundary': mean_non_boundary,
                'ratio': boundary_dist / max(mean_non_boundary, 1e-10),
            })
        
        mean_ratio = np.mean([lj['ratio'] for lj in layer_jumps])
        max_ratio = max(lj['ratio'] for lj in layer_jumps)
        
        # Statistical test: is boundary distance an outlier?
        all_consecutive = []
        for layer in range(primary[0], primary[1]):
            for i in range(n_values - 1):
                if i != boundary_idx:
                    all_consecutive.append(cosine(centroids[i, layer], centroids[i+1, layer]))
        
        boundary_dists_across_layers = [
            cosine(centroids[boundary_idx, l], centroids[boundary_idx+1, l])
            for l in range(primary[0], primary[1])
        ]
        
        # Mann-Whitney: boundary distances vs non-boundary distances
        u_stat, u_p = stats.mannwhitneyu(boundary_dists_across_layers, all_consecutive, 
                                          alternative='greater')
        
        results[condition] = {
            'boundary_idx': boundary_idx,
            'boundary_transition': f"{values[boundary_idx]}→{values[boundary_idx+1]}",
            'mean_ratio': round(float(mean_ratio), 3),
            'max_ratio': round(float(max_ratio), 3),
            'mann_whitney_U': round(float(u_stat), 1),
            'mann_whitney_p': round(float(u_p), 6),
        }
        
        print(f"    {condition}: {values[boundary_idx]}→{values[boundary_idx+1]}")
        print(f"      Mean ratio (boundary/non-boundary): {mean_ratio:.3f}")
        print(f"      Max ratio: {max_ratio:.3f}")
        print(f"      Mann-Whitney p: {u_p:.6f}")
    
    return results


# ==============================================================================
# E9: Identification Slope Predicts CP Strength (Cross-Model)
# ==============================================================================

def e9_id_slope_vs_cp(out_dir):
    """Correlate identification sigmoid slope with CP advantage across models."""
    print(f"\n  --- E9: Identification Slope vs CP Strength ---")
    
    slopes = []
    cp_advs = []
    model_labels = []
    
    for model in MODELS:
        # Get best identification slope for decade_10
        meta_path = Path(f'extractions/m3_meta_decade_10_{model}.json')
        if not meta_path.exists():
            continue
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        id_results = (meta.get('identification_results') or [])
        if not id_results:
            continue
        
        # Use digit_count framing (best signal from Session 3)
        digit_count = [r for r in id_results if r.get('framing') == 'digit_count']
        if not digit_count:
            continue
        
        vals = np.array([r['value'] for r in digit_count], dtype=float)
        probs = np.array([r.get('prob_category_b', 0.5) for r in digit_count])
        
        try:
            popt, _ = curve_fit(sigmoid, vals, probs,
                                p0=[1.0, 10.0, 1.0, 0.0], maxfev=5000)
            slope = abs(popt[0])
        except Exception:
            slope = 0.0
        
        # Get CP advantage from results
        results_path = Path(f'results/{model}/m3_results_{model}.json')
        if results_path.exists():
            with open(results_path) as f:
                res = json.load(f)
            cp_adv = res.get('decade_10', {}).get('cp_advantage', None)
            if cp_adv is None:
                # Try to compute from RSA
                cp_adv = 0.0
        else:
            # Use the 100-boundary summary
            summary_path = Path('results/analysis_100_boundary_summary.json')
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                match = [s for s in summary if s['model'] == model and s['condition'] == 'decade_100']
                cp_adv = match[0]['cp_advantage'] if match else 0.0
            else:
                cp_adv = 0.0
        
        slopes.append(slope)
        cp_advs.append(cp_adv)
        model_labels.append(model)
    
    if len(slopes) >= 3:
        r, p = stats.spearmanr(slopes, cp_advs)
        print(f"    Models: {model_labels}")
        print(f"    Slopes: {[round(s, 3) for s in slopes]}")
        print(f"    CP advantages: {[round(c, 4) for c in cp_advs]}")
        print(f"    Spearman ρ = {r:.3f}, p = {p:.4f}")
        return {'rho': round(r, 4), 'p': round(p, 4), 'n': len(slopes),
                'models': model_labels, 'slopes': slopes, 'cp_advs': cp_advs}
    else:
        print(f"    Insufficient data (n={len(slopes)})")
        return {'error': 'insufficient data'}


# ==============================================================================
# E11: CP as Local Violation of Compression
# ==============================================================================

def e11_local_violation(model):
    """Test whether CP boundary boost (λ) anticorrelates with continuous fit across layers."""
    print(f"\n  --- E11: CP as Local Violation of Compression ---")
    
    results = {}
    primary = MODELS[model]['primary']
    
    for condition in NUMERICAL_CONDITIONS:
        d = load_data(model, condition)
        if d is None:
            continue
        
        emp = analysis.compute_rdms_all_layers(d['centroids'])
        rdms = load_rdms(d)
        
        rsa = analysis.rsa_all_layers(emp, rdms, n_permutations=100)
        
        cp_rhos = rsa.get('cp_additive', {}).get('rho', [])
        cont_rhos = rsa.get('continuous', {}).get('rho', [])
        
        if cp_rhos and cont_rhos:
            # λ proxy: CP advantage at each layer
            # β proxy: continuous fit at each layer
            primary_range = range(primary[0], min(primary[1], len(cp_rhos), len(cont_rhos)))
            lambdas = [cp_rhos[l] - cont_rhos[l] for l in primary_range]
            betas = [cont_rhos[l] for l in primary_range]
            
            if len(lambdas) >= 3:
                r, p = stats.spearmanr(lambdas, betas)
                results[condition] = {
                    'rho': round(float(r), 4),
                    'p': round(float(p), 4),
                    'n_layers': len(lambdas),
                    'interpretation': 'anticorrelated (CP disrupts compression)' if r < -0.3 else
                                     'correlated (CP and compression co-occur)' if r > 0.3 else
                                     'uncorrelated',
                }
                print(f"    {condition}: ρ(λ, β) = {r:.3f}, p = {p:.4f} → {results[condition]['interpretation']}")
    
    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='M3 Exploratory Analyses')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    
    models = {args.model: MODELS[args.model]} if args.model else MODELS
    
    all_results = {}
    
    for model in models:
        print(f"\n{'#'*60}")
        print(f"  {model}")
        print(f"{'#'*60}")
        
        model_results = {}
        
        model_results['E1'] = e1_layerwise_profile(model, out_dir)
        model_results['E3'] = e3_boundary_sharpness(model, out_dir)
        model_results['E4'] = e4_magnitude_interaction(model)
        model_results['E7'] = e7_local_manifold(model, out_dir)
        model_results['E8'] = e8_phase_reset(model, out_dir)
        model_results['E11'] = e11_local_violation(model)
        
        all_results[model] = model_results
    
    # E9 is cross-model
    print(f"\n{'#'*60}")
    print(f"  CROSS-MODEL ANALYSES")
    print(f"{'#'*60}")
    all_results['cross_model'] = {}
    all_results['cross_model']['E9'] = e9_id_slope_vs_cp(out_dir)
    
    # Save
    def make_ser(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    
    save_path = out_dir / 'm3_exploratory_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=make_ser)
    print(f"\nSaved: {save_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EXPLORATORY SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nE4 — CP × Magnitude (decade_100 / decade_10 ratio):")
    for model in models:
        e4 = all_results.get(model, {}).get('E4', {})
        ratio = e4.get('ratio', 'N/A')
        print(f"  {model}: {ratio}×")
    
    print(f"\nE8 — Phase-Reset (boundary/non-boundary distance ratio):")
    for model in models:
        e8 = all_results.get(model, {}).get('E8', {})
        for cond, data in e8.items():
            print(f"  {model}/{cond}: {data.get('mean_ratio', 'N/A')} (p={data.get('mann_whitney_p', 'N/A')})")
    
    print(f"\nE9 — ID Slope vs CP Strength:")
    e9 = all_results.get('cross_model', {}).get('E9', {})
    print(f"  ρ = {e9.get('rho', 'N/A')}, p = {e9.get('p', 'N/A')}")
    
    print(f"\nE11 — λ vs β Anticorrelation:")
    for model in models:
        e11 = all_results.get(model, {}).get('E11', {})
        for cond, data in e11.items():
            print(f"  {model}/{cond}: ρ={data.get('rho', 'N/A')} → {data.get('interpretation', 'N/A')}")


if __name__ == '__main__':
    main()
