#!/usr/bin/env python3
"""
Pre-Registration Compliance: Phase 1 Gap-Fill
Weber's Law in Transformer Magnitude Representations (Project 4.2)

Runs all no-GPU analyses identified in the compliance audit:
1. Semantic structure diagnostic for frequency-matched nouns (Section 5.7.5)
2. Formal H5 evaluation (layer-wise transition)
3. Formal H6 evaluation (distance + ratio effects)
4. Single-token control analysis (Section 5.7.2)
5. Paradigm C details check (γ fit, normalised precision)

All use existing data — no model loading required.

Author: JP Cacioli
Date: March 2026
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


MAGNITUDES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50,
              60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1000]


def load_analysis(results_dir, model_key, domain):
    path = Path(results_dir) / "paradigm_a" / model_key / domain / "paradigm_a_analysis.json"
    with open(path) as f:
        return json.load(f)


def load_hidden_states(results_dir, model_key, domain):
    path = Path(results_dir) / "paradigm_a" / model_key / domain / "hidden_states.npz"
    data = np.load(str(path), allow_pickle=True)
    return data['centroids']  # (n_mags, n_layers, d_model)


# ====================================================================
# 1. Semantic Structure Diagnostic (Section 5.7.5)
# ====================================================================

def semantic_structure_diagnostic(results_dir):
    """Compare noun RDM against semantic (layer 0) vs frequency-rank RDMs.
    
    Pre-reg: "compute a semantic RDM from the static token embeddings (layer 0 cosine
    distances). If the noun model-RDM at layers 16-32 correlates more strongly with the
    semantic RDM than with the frequency-rank RDM (Mantel test comparison), the geometry
    is tracking meaning, not frequency."
    """
    print(f"\n{'='*70}")
    print(f"1. SEMANTIC STRUCTURE DIAGNOSTIC (Section 5.7.5)")
    print(f"{'='*70}")
    
    # Load noun hidden states
    noun_path = Path(results_dir) / "paradigm_a" / "llama_instruct" / "freq_nouns"
    if not noun_path.exists():
        # Try alternative paths
        candidates = list(Path(results_dir).rglob("*freq_noun*"))
        if candidates:
            noun_path = candidates[0].parent if candidates[0].is_file() else candidates[0]
        else:
            print("  WARNING: Frequency-matched noun data not found. Skipping.")
            return None
    
    # Look for hidden states
    hs_file = noun_path / "hidden_states.npz"
    if not hs_file.exists():
        hs_candidates = list(noun_path.glob("*.npz"))
        if hs_candidates:
            hs_file = hs_candidates[0]
        else:
            print(f"  WARNING: No .npz file found in {noun_path}. Skipping.")
            return None
    
    data = np.load(str(hs_file), allow_pickle=True)
    if 'centroids' in data:
        noun_centroids = data['centroids']
    else:
        print(f"  Available keys: {list(data.keys())}")
        print("  WARNING: Cannot load noun centroids. Skipping.")
        return None
    
    n_nouns, n_layers, d_model = noun_centroids.shape
    print(f"  Loaded noun centroids: {noun_centroids.shape}")
    
    # Semantic RDM: layer 0 cosine distances (static embeddings)
    layer0 = noun_centroids[:, 0, :]
    semantic_rdm = squareform(pdist(layer0.astype(np.float64), metric='cosine'))
    
    # Frequency-rank RDM: |rank(i) - rank(j)|
    # Nouns are already ordered by frequency match to magnitudes
    freq_ranks = np.arange(n_nouns, dtype=np.float64)
    freq_rdm = squareform(pdist(freq_ranks.reshape(-1, 1), metric='euclidean'))
    
    # Upper triangle for correlation
    triu = np.triu_indices(n_nouns, k=1)
    sem_flat = semantic_rdm[triu]
    freq_flat = freq_rdm[triu]
    
    results = {"layers": []}
    n_sem_wins = 0
    n_freq_wins = 0
    
    print(f"\n  Layer | ρ(semantic) | ρ(frequency) | Winner")
    for layer in range(n_layers):
        X = noun_centroids[:, layer, :].astype(np.float64)
        model_rdm = squareform(pdist(X, metric='cosine'))
        model_flat = model_rdm[triu]
        
        rho_sem, p_sem = spearmanr(model_flat, sem_flat)
        rho_freq, p_freq = spearmanr(model_flat, freq_flat)
        
        winner = "semantic" if rho_sem > rho_freq else "frequency"
        if layer >= 16:
            if winner == "semantic":
                n_sem_wins += 1
            else:
                n_freq_wins += 1
        
        results["layers"].append({
            "layer": layer,
            "rho_semantic": float(rho_sem),
            "rho_frequency": float(rho_freq),
            "winner": winner,
        })
        
        if layer >= 16 or layer in [0, 5, 10]:
            print(f"    {layer:2d}  | {rho_sem:+.4f}      | {rho_freq:+.4f}       | {winner}")
    
    results["summary"] = {
        "semantic_wins_primary": n_sem_wins,
        "frequency_wins_primary": n_freq_wins,
        "interpretation": "semantic" if n_sem_wins > n_freq_wins else "frequency",
    }
    
    print(f"\n  Primary layers: semantic wins {n_sem_wins}, frequency wins {n_freq_wins}")
    if n_sem_wins > n_freq_wins:
        print(f"  → Noun geometry tracks MEANING, not frequency. Frequency artefact ruled out.")
    else:
        print(f"  → Noun geometry tracks FREQUENCY. Frequency confound is operating.")
    
    return results


# ====================================================================
# 2. Formal H5 Evaluation (Layer-Wise Transition)
# ====================================================================

def evaluate_h5(results_dir):
    """H5: Stevens exponent β decreases across layers (linear → log transition).
    
    Pre-reg: "Spearman ρ between layer index and estimated Stevens exponent β is 
    significantly negative (p < 0.05) for at least 2 of 3 domains in both models."
    """
    print(f"\n{'='*70}")
    print(f"2. H5 EVALUATION (Layer-Wise Transition)")
    print(f"{'='*70}")
    
    results = {}
    
    for model_key in ["llama_instruct", "mistral_instruct"]:
        results[model_key] = {}
        print(f"\n  {model_key}:")
        
        n_pass = 0
        for domain in ["numerical", "temporal", "spatial"]:
            try:
                analysis = load_analysis(results_dir, model_key, domain)
            except FileNotFoundError:
                print(f"    {domain}: data not found")
                results[model_key][domain] = {"status": "not_found"}
                continue
            
            # Extract Stevens β at each layer (cosine metric)
            layers = []
            betas = []
            for lname, ldata in sorted(analysis['layers'].items()):
                layer_num = int(lname.split('_')[1])
                cos = ldata.get('cosine', {})
                mf = cos.get('model_fits', {})
                stevens = mf.get('stevens', {})
                beta = stevens.get('params', {}).get('beta', None)
                if beta is not None:
                    layers.append(layer_num)
                    betas.append(beta)
            
            if len(layers) < 5:
                print(f"    {domain}: insufficient Stevens fits ({len(layers)} layers)")
                results[model_key][domain] = {"status": "insufficient_data"}
                continue
            
            rho, p_val = spearmanr(layers, betas)
            passes = rho < 0 and p_val < 0.05
            if passes:
                n_pass += 1
            
            results[model_key][domain] = {
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "n_layers": len(layers),
                "passes": passes,
                "beta_range": [float(min(betas)), float(max(betas))],
            }
            
            print(f"    {domain}: ρ = {rho:.4f}, p = {p_val:.4f} → {'PASS' if passes else 'FAIL'}")
        
        results[model_key]["n_domains_pass"] = n_pass
        results[model_key]["h5_supported"] = n_pass >= 2
        print(f"    H5 for {model_key}: {n_pass}/3 domains → {'SUPPORTED' if n_pass >= 2 else 'NOT SUPPORTED'}")
    
    # Programme level
    both_pass = all(results[m].get("h5_supported", False) for m in ["llama_instruct", "mistral_instruct"])
    results["programme_level"] = both_pass
    print(f"\n  H5 programme level: {'SUPPORTED' if both_pass else 'NOT SUPPORTED'}")
    
    return results


# ====================================================================
# 3. Formal H6 Evaluation (Distance + Ratio Effects)
# ====================================================================

def evaluate_h6(results_dir):
    """H6: Distance and ratio effects in comparison accuracy.
    
    Pre-reg: "(a) logistic regression on |n1-n2| significant (p < 0.01); (b) adding 
    log((n1+n2)/2) significantly improves model (Δ deviance, p < 0.01); (c) distance × 
    baseline interaction significant (p < 0.05)"
    """
    print(f"\n{'='*70}")
    print(f"3. H6 EVALUATION (Distance + Ratio Effects)")
    print(f"{'='*70}")
    
    results = {}
    
    for model_key in ["llama_instruct", "mistral_instruct"]:
        # Load B1 raw data
        raw_path = Path(results_dir) / "paradigm_b" / model_key / "numerical" / "paradigm_b_raw.json"
        if not raw_path.exists():
            print(f"  {model_key}: B1 data not found")
            results[model_key] = {"status": "not_found"}
            continue
        
        with open(raw_path) as f:
            raw = json.load(f)
        
        b1 = [r for r in raw if r['task_type'] == 'B1']
        print(f"\n  {model_key}: {len(b1)} B1 trials")
        
        # Extract variables
        correct = np.array([r['correct'] for r in b1], dtype=float)
        abs_diff = np.array([abs(r['baseline'] * r['ratio'] - r['baseline']) for r in b1])
        log_mean_mag = np.array([np.log((r['baseline'] + r['baseline'] * r['ratio']) / 2) for r in b1])
        baseline = np.array([r['baseline'] for r in b1])
        
        # (a) Logistic regression on |n1-n2|
        X_a = sm.add_constant(abs_diff)
        model_a = sm.Logit(correct, X_a).fit(disp=0)
        p_distance = model_a.pvalues[1]
        dev_a = model_a.llf  # log-likelihood
        
        # (b) Add log(mean magnitude) — does it improve?
        X_b = sm.add_constant(np.column_stack([abs_diff, log_mean_mag]))
        model_b = sm.Logit(correct, X_b).fit(disp=0)
        
        # Likelihood ratio test
        delta_dev = 2 * (model_b.llf - model_a.llf)
        from scipy.stats import chi2
        p_ratio = chi2.sf(delta_dev, df=1)
        
        # (c) Distance × baseline interaction
        interaction = abs_diff * baseline
        X_c = sm.add_constant(np.column_stack([abs_diff, log_mean_mag, interaction]))
        model_c = sm.Logit(correct, X_c).fit(disp=0)
        p_interaction = model_c.pvalues[3]  # interaction term
        
        passes_a = p_distance < 0.01
        passes_b = p_ratio < 0.01
        passes_c = p_interaction < 0.05
        h6_passes = passes_a and passes_b and passes_c
        
        results[model_key] = {
            "n_trials": len(b1),
            "distance_effect_p": float(p_distance),
            "distance_effect_pass": passes_a,
            "ratio_effect_delta_dev": float(delta_dev),
            "ratio_effect_p": float(p_ratio),
            "ratio_effect_pass": passes_b,
            "interaction_p": float(p_interaction),
            "interaction_pass": passes_c,
            "h6_passes": h6_passes,
        }
        
        print(f"    (a) Distance effect: p = {p_distance:.6f} → {'PASS' if passes_a else 'FAIL'}")
        print(f"    (b) Ratio effect: Δdev = {delta_dev:.2f}, p = {p_ratio:.6f} → {'PASS' if passes_b else 'FAIL'}")
        print(f"    (c) Interaction: p = {p_interaction:.6f} → {'PASS' if passes_c else 'FAIL'}")
        print(f"    H6: {'PASS' if h6_passes else 'FAIL'}")
    
    return results


# ====================================================================
# 4. Single-Token Control (Section 5.7.2)
# ====================================================================

def single_token_control(results_dir):
    """Compare Paradigm A results with vs without multi-token magnitude (1000).
    
    Pre-reg: "If ΔR²(log-linear) differs by >0.10 from full set, tokenisation flagged. 
    If <0.05, ruled out."
    """
    print(f"\n{'='*70}")
    print(f"4. SINGLE-TOKEN CONTROL (Section 5.7.2)")
    print(f"{'='*70}")
    
    # Only Llama has single-token magnitudes (25/26)
    # Magnitude 1000 is the only multi-token one
    
    centroids = load_hidden_states(results_dir, "llama_instruct", "numerical")
    n_mags, n_layers, d_model = centroids.shape
    
    # Full set indices (all 26)
    all_mags = np.array(MAGNITUDES, dtype=np.float64)
    log_all = np.log(all_mags)
    
    # Single-token set (drop 1000 = index 25)
    single_idx = list(range(25))  # 0-24 (magnitudes 1-700)
    single_mags = all_mags[single_idx]
    log_single = np.log(single_mags)
    
    print(f"  Full set: {len(all_mags)} magnitudes")
    print(f"  Single-token set: {len(single_mags)} magnitudes (dropped: 1000)")
    
    results = {"layers": []}
    max_delta = 0
    
    print(f"\n  Layer | Full R²(Weber) | Single R²(Weber) | ΔR²")
    for layer in range(n_layers):
        # Full set
        X_full = centroids[:, layer, :].astype(np.float64)
        dists_full = pdist(X_full, metric='cosine')
        log_diffs_full = pdist(log_all.reshape(-1, 1), metric='euclidean')
        lin_diffs_full = pdist(all_mags.reshape(-1, 1), metric='euclidean')
        
        # Weber R² (full)
        from numpy.polynomial.polynomial import polyfit as np_polyfit
        coeffs_w = np.polyfit(log_diffs_full, dists_full, 1)
        pred_w = np.polyval(coeffs_w, log_diffs_full)
        ss_res_w = np.sum((dists_full - pred_w)**2)
        ss_tot = np.sum((dists_full - dists_full.mean())**2)
        r2_weber_full = 1 - ss_res_w / ss_tot if ss_tot > 0 else 0
        
        # Single-token set
        X_single = centroids[single_idx, layer, :].astype(np.float64)
        dists_single = pdist(X_single, metric='cosine')
        log_diffs_single = pdist(log_single.reshape(-1, 1), metric='euclidean')
        
        coeffs_ws = np.polyfit(log_diffs_single, dists_single, 1)
        pred_ws = np.polyval(coeffs_ws, log_diffs_single)
        ss_res_ws = np.sum((dists_single - pred_ws)**2)
        ss_tot_s = np.sum((dists_single - dists_single.mean())**2)
        r2_weber_single = 1 - ss_res_ws / ss_tot_s if ss_tot_s > 0 else 0
        
        delta_r2 = abs(r2_weber_full - r2_weber_single)
        max_delta = max(max_delta, delta_r2)
        
        results["layers"].append({
            "layer": layer,
            "r2_weber_full": float(r2_weber_full),
            "r2_weber_single": float(r2_weber_single),
            "delta_r2": float(delta_r2),
        })
        
        if layer >= 16 and layer % 4 == 0:
            print(f"    {layer:2d}  | {r2_weber_full:.4f}          | {r2_weber_single:.4f}           | {delta_r2:.4f}")
    
    primary_deltas = [r["delta_r2"] for r in results["layers"] if r["layer"] >= 16]
    mean_delta = np.mean(primary_deltas)
    
    if mean_delta > 0.10:
        verdict = "FLAGGED — tokenisation is a major limitation"
    elif mean_delta < 0.05:
        verdict = "RULED OUT — tokenisation effect is negligible"
    else:
        verdict = "MARGINAL — tokenisation has a moderate effect"
    
    results["summary"] = {
        "mean_delta_r2_primary": float(mean_delta),
        "max_delta_r2": float(max_delta),
        "verdict": verdict,
    }
    
    print(f"\n  Mean ΔR² (primary layers): {mean_delta:.4f}")
    print(f"  Max ΔR²: {max_delta:.4f}")
    print(f"  Verdict: {verdict}")
    
    return results


# ====================================================================
# 5. Paradigm C Details Check
# ====================================================================

def paradigm_c_details(results_dir):
    """Verify Paradigm C reports γ fit, normalised precision, Fisher info.
    
    Pre-reg: "Fit precision = a / n^γ. Report both raw and log-step-normalised precision."
    """
    print(f"\n{'='*70}")
    print(f"5. PARADIGM C DETAILS CHECK")
    print(f"{'='*70}")
    
    for model_key in ["llama_instruct", "mistral_instruct"]:
        for domain in ["numerical", "temporal", "spatial"]:
            rob_path = Path(results_dir) / "paradigm_a" / model_key / domain / "paradigm_c_robustness.json"
            if not rob_path.exists():
                print(f"  {model_key}/{domain}: not found")
                continue
            
            with open(rob_path) as f:
                data = json.load(f)
            
            # Check what's in the file
            keys = list(data.keys()) if isinstance(data, dict) else "list"
            
            has_gamma = False
            has_normalised = False
            has_fisher = False
            
            if isinstance(data, dict):
                # Search for gamma/power-law fit
                for k, v in data.items():
                    k_lower = k.lower()
                    if 'gamma' in k_lower or 'power' in k_lower:
                        has_gamma = True
                    if 'normalise' in k_lower or 'normalize' in k_lower or 'log_step' in k_lower:
                        has_normalised = True
                    if 'fisher' in k_lower:
                        has_fisher = True
            
            status = []
            if has_gamma:
                status.append("γ ✅")
            else:
                status.append("γ ❌")
            if has_normalised:
                status.append("normalised ✅")
            else:
                status.append("normalised ❌")
            if has_fisher:
                status.append("Fisher ✅")
            else:
                status.append("Fisher ❌ (exploratory)")
            
            print(f"  {model_key}/{domain}: {', '.join(status)}")
            print(f"    Keys: {list(data.keys())[:10] if isinstance(data, dict) else 'N/A'}")


# ====================================================================
# Main
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1 compliance gap-fill")
    parser.add_argument("--project-root", type=str, default=r"C:\weber")
    args = parser.parse_args()
    
    results_dir = Path(args.project_root) / "results"
    output_dir = results_dir / "compliance"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. Semantic diagnostic
    all_results["semantic_diagnostic"] = semantic_structure_diagnostic(results_dir)
    
    # 2. H5
    all_results["h5"] = evaluate_h5(results_dir)
    
    # 3. H6
    all_results["h6"] = evaluate_h6(results_dir)
    
    # 4. Single-token control
    all_results["single_token_control"] = single_token_control(results_dir)
    
    # 5. Paradigm C check
    paradigm_c_details(results_dir)
    
    # Save
    out_path = output_dir / "phase1_compliance.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n\nSaved: {out_path}")


if __name__ == "__main__":
    main()
