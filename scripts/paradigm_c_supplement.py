#!/usr/bin/env python3
"""
Paradigm C Supplement: γ Fit, Normalised Precision, Fisher Information
Weber's Law in Transformer Magnitude Representations (Project 4.2)

Fills gaps identified in compliance audit:
- Fit precision = a / n^γ (efficient coding prediction: γ ≈ 1)
- Log-step-normalised precision
- Fisher information analogue (exploratory)

Uses existing Paradigm A hidden states. No GPU needed.

Author: JP Cacioli
Date: March 2026
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

MAGNITUDES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50,
              60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1000]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def compute_precision_supplement(results_dir, model_key, domain):
    """Compute γ fit, normalised precision, and Fisher info for one model × domain."""
    
    hs_path = Path(results_dir) / "paradigm_a" / model_key / domain / "hidden_states.npz"
    data = np.load(str(hs_path), allow_pickle=True)
    centroids = data['centroids']  # (n_mags, n_layers, d_model)
    
    # Also load per-carrier for Fisher info
    per_carrier = data.get('per_carrier', None)  # (n_mags, n_carriers, n_layers, d_model)
    
    n_mags, n_layers, d_model = centroids.shape
    mags = np.array(MAGNITUDES[:n_mags], dtype=np.float64)
    log_mags = np.log(mags)
    
    # Adjacent pairs
    n_pairs = n_mags - 1
    midpoints = (mags[:-1] + mags[1:]) / 2
    log_midpoints = np.log(midpoints)
    
    # Step sizes
    linear_steps = np.diff(mags)
    log_steps = np.diff(log_mags)
    
    results = {"model": model_key, "domain": domain, "layers": []}
    
    # Power-law fit function: precision = a * n^(-γ)
    def power_law(n, a, gamma):
        return a * np.power(n, -gamma)
    
    for layer in range(n_layers):
        # Raw precision: 1 / ||h(n+1) - h(n)||
        diffs = []
        for i in range(n_pairs):
            d = np.linalg.norm(
                centroids[i+1, layer, :].astype(np.float64) - 
                centroids[i, layer, :].astype(np.float64)
            )
            diffs.append(d)
        
        diffs = np.array(diffs)
        raw_precision = 1.0 / (diffs + 1e-10)
        
        # Log-step-normalised precision: precision / log_step
        # Pre-reg: "Under logarithmic geometry, precision normalised by log-step-size 
        # should be approximately constant."
        normalised_precision = raw_precision / (log_steps + 1e-10)
        
        # Linear-step-normalised (for comparison)
        linear_normalised = raw_precision / (linear_steps + 1e-10)
        
        # Spearman ρ of raw precision vs magnitude midpoint (H3 test)
        rho_raw, p_raw = spearmanr(midpoints, raw_precision)
        
        # Spearman ρ of normalised precision vs magnitude (should be ~0 if log geometry)
        rho_norm, p_norm = spearmanr(midpoints, normalised_precision)
        
        # Fit precision = a * n^(-γ)
        try:
            popt, pcov = curve_fit(
                power_law, midpoints, raw_precision,
                p0=[raw_precision[0], 1.0],
                bounds=([0, 0], [np.inf, 5.0]),
                maxfev=5000
            )
            gamma = popt[1]
            a_fit = popt[0]
            
            # R² of the fit
            pred = power_law(midpoints, *popt)
            ss_res = np.sum((raw_precision - pred)**2)
            ss_tot = np.sum((raw_precision - raw_precision.mean())**2)
            r2_gamma = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except Exception:
            gamma = None
            a_fit = None
            r2_gamma = None
        
        # Fisher information analogue (exploratory)
        # FI(n) ≈ (∂h/∂n)² / Var(h)
        # Estimated from finite differences and cross-carrier variance
        fisher_info = None
        if per_carrier is not None:
            fi_values = []
            for i in range(n_pairs):
                # Finite difference: ∂h/∂n ≈ (h(n+1) - h(n)) / (n+1 - n)
                dh = centroids[i+1, layer, :].astype(np.float64) - centroids[i, layer, :].astype(np.float64)
                dn = mags[i+1] - mags[i]
                deriv_sq = np.sum((dh / dn)**2)
                
                # Variance: across carriers at magnitude i
                carrier_states = per_carrier[i, :, layer, :].astype(np.float64)
                var_h = np.mean(np.var(carrier_states, axis=0))
                
                fi = deriv_sq / (var_h + 1e-10)
                fi_values.append(fi)
            
            fi_values = np.array(fi_values)
            fisher_info = fi_values.tolist()
            
            # Check if FI ∝ 1/n² (efficient coding prediction)
            rho_fi, p_fi = spearmanr(midpoints, fi_values)
        else:
            rho_fi = None
            p_fi = None
        
        layer_result = {
            "layer": layer,
            "raw_precision": raw_precision.tolist(),
            "normalised_precision_log": normalised_precision.tolist(),
            "normalised_precision_linear": linear_normalised.tolist(),
            "rho_raw": float(rho_raw),
            "p_raw": float(p_raw),
            "rho_normalised": float(rho_norm),
            "p_normalised": float(p_norm),
            "gamma": float(gamma) if gamma is not None else None,
            "gamma_a": float(a_fit) if a_fit is not None else None,
            "gamma_r2": float(r2_gamma) if r2_gamma is not None else None,
        }
        
        if fisher_info is not None:
            layer_result["fisher_info"] = fisher_info
            layer_result["rho_fisher"] = float(rho_fi) if rho_fi is not None else None
            layer_result["p_fisher"] = float(p_fi) if p_fi is not None else None
        
        results["layers"].append(layer_result)
    
    # Summary across primary layers
    primary = [r for r in results["layers"] if r["layer"] >= 16]
    gammas = [r["gamma"] for r in primary if r["gamma"] is not None]
    
    results["summary"] = {
        "n_primary_layers": len(primary),
        "gamma_mean": float(np.mean(gammas)) if gammas else None,
        "gamma_std": float(np.std(gammas)) if gammas else None,
        "gamma_median": float(np.median(gammas)) if gammas else None,
        "gamma_interpretation": (
            "≈1 (consistent with 1/n prior)" if gammas and 0.8 <= np.mean(gammas) <= 1.2
            else "≠1 (different prior)" if gammas
            else "could not fit"
        ),
        "normalised_precision_flat": (
            float(np.mean([abs(r["rho_normalised"]) for r in primary]))
        ),
        "midpoints": midpoints.tolist(),
    }
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Paradigm C supplement")
    parser.add_argument("--project-root", type=str, default=r"C:\weber")
    args = parser.parse_args()
    
    results_dir = Path(args.project_root) / "results"
    
    print(f"\n{'='*70}")
    print(f"PARADIGM C SUPPLEMENT: γ Fit, Normalised Precision, Fisher Info")
    print(f"{'='*70}")
    
    for model_key in ["llama_instruct", "mistral_instruct"]:
        for domain in ["numerical", "temporal", "spatial"]:
            hs_path = results_dir / "paradigm_a" / model_key / domain / "hidden_states.npz"
            if not hs_path.exists():
                print(f"\n  {model_key}/{domain}: hidden states not found, skipping")
                continue
            
            print(f"\n  --- {model_key} / {domain} ---")
            result = compute_precision_supplement(results_dir, model_key, domain)
            
            s = result["summary"]
            print(f"    γ mean (primary layers): {s['gamma_mean']:.3f}" if s['gamma_mean'] else "    γ: could not fit")
            if s['gamma_mean']:
                print(f"    γ std: {s['gamma_std']:.3f}")
                print(f"    Interpretation: {s['gamma_interpretation']}")
            print(f"    Mean |ρ(normalised precision)| at primary: {s['normalised_precision_flat']:.3f}")
            
            # Save alongside existing robustness file
            out_path = results_dir / "paradigm_a" / model_key / domain / "paradigm_c_supplement.json"
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            print(f"    Saved: {out_path}")


if __name__ == "__main__":
    main()
