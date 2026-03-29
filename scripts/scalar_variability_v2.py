"""
Scalar Variability in Transformer Magnitude Representations
Analysis script v2 — companion to Weber (Cacioli, 2026)

Pre-registered on OSF prior to execution.
Runs on existing Paradigm A hidden states (per_carrier tensor).

Usage:
    cd C:\weber
    .\venv\Scripts\Activate.ps1
    python scripts/scalar_variability.py

Output: results/scalar_variability/

Changes from v1 (post-adversarial review):
  - Sentence-identity control elevated to co-primary
  - V_proj (magnitude-axis variance) elevated to confirmatory robustness
  - Theil-Sen robustness check added
  - Outlier exclusion rule (>3x layerwise median)
  - H2 window tightened to [0.8, 1.2]
  - AIC comparison flagged as descriptive
"""

import numpy as np
import json
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────

WEBER_ROOT = Path(r"C:\weber")
RESULTS_DIR = WEBER_ROOT / "results" / "scalar_variability"
PARADIGM_A_DIR = WEBER_ROOT / "results" / "paradigm_a"

MODELS = {
    "llama_instruct": "Llama-3-8B-Instruct",
    "mistral_instruct": "Mistral-7B-Instruct-v0.3",
    "llama_base": "Llama-3-8B-Base",
}

MAGNITUDES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        15, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                        150, 200, 300, 500, 700, 1000])

SEED = 42
N_BOOTSTRAP = 10_000
N_CARRIERS = 5
BOUNDARY_INDICES = [8, 9, 19]  # magnitudes 9, 10, 100
OUTLIER_FACTOR = 3.0  # exclude V(n) > 3x layerwise median
H2_WINDOW = (0.8, 1.2)  # tightened from (0.5, 1.5)

np.random.seed(SEED)


# ── Variability Measures ───────────────────────────────────────

def compute_variability(per_carrier):
    """
    Compute variability measures from per-carrier hidden states.
    per_carrier: (26, 5, 33, 4096)
    Returns dict of arrays each (26, 33) plus centroid info.
    """
    n_mag, n_car, n_layers, n_dims = per_carrier.shape
    centroids = per_carrier.mean(axis=1)  # (26, 33, 4096)

    # Primary: mean Euclidean distance from centroid
    deviations = per_carrier - centroids[:, np.newaxis, :, :]
    dev_norms = np.sqrt((deviations ** 2).sum(axis=-1))  # (26, 5, 33)
    v_eucl = dev_norms.mean(axis=1)  # (26, 33)

    # Secondary: mean pairwise Euclidean distance
    v_pairwise = np.zeros((n_mag, n_layers))
    for i in range(n_car):
        for j in range(i + 1, n_car):
            diff = per_carrier[:, i, :, :] - per_carrier[:, j, :, :]
            v_pairwise += np.sqrt((diff ** 2).sum(axis=-1))
    v_pairwise /= (n_car * (n_car - 1) / 2)

    # Secondary: trace of covariance
    v_trace = np.zeros((n_mag, n_layers))
    for m in range(n_mag):
        for l in range(n_layers):
            vecs = per_carrier[m, :, l, :]
            vecs_c = vecs - vecs.mean(axis=0, keepdims=True)
            v_trace[m, l] = (vecs_c ** 2).sum() / (n_car - 1)

    # Confirmatory robustness: V_proj (variance along magnitude PC1)
    v_proj = np.zeros((n_mag, n_layers))
    for l in range(n_layers):
        cent_l = centroids[:, l, :]  # (26, 4096)
        cent_c = cent_l - cent_l.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(cent_c, full_matrices=False)
        pc1 = Vt[0]
        for m in range(n_mag):
            projections = per_carrier[m, :, l, :] @ pc1
            v_proj[m, l] = projections.std(ddof=1)

    centroid_norms = np.sqrt((centroids ** 2).sum(axis=-1))  # (26, 33)

    return {
        "v_eucl": v_eucl,
        "v_pairwise": v_pairwise,
        "v_trace": v_trace,
        "v_proj": v_proj,
        "centroid_norms": centroid_norms,
        "centroids": centroids,
    }


def compute_sentence_residual_variability(per_carrier):
    """
    Co-primary: remove systematic sentence effects, then compute V_eucl.
    per_carrier: (26, 5, 33, 4096)
    Returns v_residual: (26, 33)
    """
    n_mag, n_car, n_layers, n_dims = per_carrier.shape
    v_residual = np.zeros((n_mag, n_layers))

    for l in range(n_layers):
        layer_data = per_carrier[:, :, l, :]  # (26, 5, 4096)
        # Sentence means across all magnitudes
        sent_means = layer_data.mean(axis=0)  # (5, 4096)
        grand_mean = sent_means.mean(axis=0)  # (4096,)
        sent_effects = sent_means - grand_mean  # (5, 4096)
        # Subtract sentence effects
        adjusted = layer_data - sent_effects[np.newaxis, :, :]  # (26, 5, 4096)
        # Compute V_eucl on adjusted
        adj_centroids = adjusted.mean(axis=1)  # (26, 4096)
        devs = adjusted - adj_centroids[:, np.newaxis, :]
        dev_norms = np.sqrt((devs ** 2).sum(axis=-1))  # (26, 5)
        v_residual[:, l] = dev_norms.mean(axis=1)

    return v_residual


# ── Scaling Analysis ───────────────────────────────────────────

def theil_sen_slope(x, y):
    """Theil-Sen estimator: median of pairwise slopes."""
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(slopes))


def fit_scaling_exponent(magnitudes, variability, bootstrap=True, mask=None):
    """
    Fit log(V) = log(k) + alpha * log(n) via OLS + Theil-Sen.
    mask: boolean array — True = include. If None, include all.
    """
    if mask is not None:
        magnitudes = magnitudes[mask]
        variability = variability[mask]

    # Remove any zero or negative V
    valid = variability > 0
    if valid.sum() < 5:
        return {"alpha": float('nan'), "alpha_ts": float('nan'),
                "r_squared": float('nan'), "p_value": float('nan')}

    mags = magnitudes[valid]
    var = variability[valid]
    log_n = np.log(mags)
    log_v = np.log(var)

    # OLS
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_v)

    # Theil-Sen
    ts_slope = theil_sen_slope(log_n, log_v)

    result = {
        "alpha": float(slope),
        "alpha_ts": float(ts_slope),
        "alpha_ols_ts_diff": float(abs(slope - ts_slope)),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n_valid": int(valid.sum()),
    }

    if bootstrap:
        rng = np.random.RandomState(SEED)
        boot_alphas = []
        n = len(mags)
        for _ in range(N_BOOTSTRAP):
            idx = rng.randint(0, n, n)
            s, _, _, _, _ = stats.linregress(log_n[idx], log_v[idx])
            boot_alphas.append(s)
        boot_alphas = np.array(boot_alphas)
        result["alpha_ci_lo"] = float(np.percentile(boot_alphas, 2.5))
        result["alpha_ci_hi"] = float(np.percentile(boot_alphas, 97.5))

    return result


def apply_outlier_mask(variability_layer, factor=OUTLIER_FACTOR):
    """
    Returns boolean mask: True = keep, False = outlier.
    Outlier defined as V(n) > factor * median(V) at this layer.
    """
    median_v = np.median(variability_layer)
    if median_v <= 0:
        return np.ones(len(variability_layer), dtype=bool)
    return variability_layer <= factor * median_v


def fit_noise_models(magnitudes, variability):
    """Fit four noise models, return AICc. Descriptive only."""
    valid = variability > 0
    if valid.sum() < 5:
        return {"best": "insufficient_data"}

    mags = magnitudes[valid]
    var = variability[valid]
    n = len(mags)
    log_n = np.log(mags)
    results = {}

    def aicc(ss, k_params):
        ll = n * np.log(ss / n + 1e-30)
        aic = ll + 2 * k_params
        return float(aic + (2 * k_params * (k_params + 1)) / max(n - k_params - 1, 1))

    # 1. Constant
    c = var.mean()
    ss = ((var - c) ** 2).sum()
    results["constant"] = {"aicc": aicc(ss, 1)}

    # 2. Scalar: V = k*n
    k_s = (var / mags).mean()
    ss = ((var - k_s * mags) ** 2).sum()
    results["scalar"] = {"aicc": aicc(ss, 1)}

    # 3. Log-scalar: V = k*log(n) + c
    sl, ic, _, _, _ = stats.linregress(log_n, var)
    ss = ((var - (sl * log_n + ic)) ** 2).sum()
    results["log_scalar"] = {"aicc": aicc(ss, 2)}

    # 4. Power-law: V = k*n^alpha (via log-log OLS)
    log_v = np.log(var)
    sl2, ic2, _, _, _ = stats.linregress(log_n, log_v)
    pred = np.exp(ic2) * mags ** sl2
    ss = ((var - pred) ** 2).sum()
    results["power_law"] = {"aicc": aicc(ss, 2), "alpha": float(sl2)}

    best = min(results.keys(), key=lambda x: results[x]["aicc"])
    for k in results:
        results[k]["delta_aicc"] = results[k]["aicc"] - results[best]["aicc"]
    results["best"] = best
    return results


# ── Plotting ───────────────────────────────────────────────────

def plot_variability_vs_magnitude(magnitudes, v_raw, v_resid, model_name, layer, out_path):
    """Log-log plot: raw V and sentence-corrected V at one layer."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, v, label in [(axes[0], v_raw, "V_eucl (raw)"),
                          (axes[1], v_resid, "V_residual (sentence-corrected)")]:
        valid = v > 0
        ax.loglog(magnitudes[valid], v[valid], 'ko', markersize=5)
        log_n = np.log(magnitudes[valid])
        log_v = np.log(v[valid])
        sl, ic, r, p, _ = stats.linregress(log_n, log_v)
        n_fit = np.logspace(np.log10(magnitudes.min()), np.log10(magnitudes.max()), 100)
        ax.loglog(n_fit, np.exp(ic) * n_fit ** sl, 'r-', lw=1.5,
                  label=f'α = {sl:.3f} (R² = {r**2:.3f})')
        # Scalar reference
        if v[valid].min() > 0:
            ref_k = v[valid][0] / magnitudes[valid][0]
            ax.loglog(n_fit, ref_k * n_fit, ':', color='gray', alpha=0.4, label='α=1 (scalar)')
        ax.set_xlabel('Magnitude', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{model_name} — Layer {layer}', fontsize=11)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_alpha_profile(all_results, out_path):
    """Layerwise alpha for all models: raw, residual, and V_proj."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {'llama_instruct': '#1f77b4', 'mistral_instruct': '#ff7f0e', 'llama_base': '#2ca02c'}
    titles = ['V_eucl (raw)', 'V_residual (sentence-corrected)', 'V_proj (magnitude axis)']
    keys = ['layerwise_alpha_raw', 'layerwise_alpha_residual', 'layerwise_alpha_proj']

    for ax, title, key in zip(axes, titles, keys):
        for mk, ml in MODELS.items():
            if mk not in all_results or key not in all_results[mk]:
                continue
            alphas = all_results[mk][key]
            ax.plot(range(len(alphas)), alphas, 'o-', color=colors[mk],
                    markersize=3, lw=1.2, label=ml)
        ax.axhline(y=1.0, color='gray', ls='--', alpha=0.4, label='Scalar (α=1)')
        ax.axhline(y=0.0, color='gray', ls=':', alpha=0.4)
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('α', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ── Main Analysis ──────────────────────────────────────────────

def analyse_model(model_key, model_label):
    print(f"\n{'='*60}")
    print(f"  {model_label}")
    print(f"{'='*60}")

    data_path = PARADIGM_A_DIR / model_key / "numerical" / "hidden_states.npz"
    if not data_path.exists():
        print(f"  [SKIP] No data at {data_path}")
        return None

    data = np.load(data_path)
    per_carrier = data["per_carrier"]
    print(f"  Loaded: {per_carrier.shape}")
    n_mag, n_car, n_layers, n_dims = per_carrier.shape

    model_out = RESULTS_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    # ── Compute measures ──
    print("  Computing variability measures...")
    var_m = compute_variability(per_carrier)
    v_raw = var_m["v_eucl"]
    v_proj = var_m["v_proj"]
    centroid_norms = var_m["centroid_norms"]

    print("  Computing sentence-corrected variability (co-primary)...")
    v_resid = compute_sentence_residual_variability(per_carrier)

    primary_layers = list(range(16, 32))  # Weber primary layers (transformer layers 16-31)

    # ── Layerwise analysis ──
    print("  Fitting scaling exponents (OLS + Theil-Sen)...")
    alpha_raw, alpha_resid, alpha_proj = [], [], []
    layer_details = []

    for l in range(n_layers):
        do_boot = l in primary_layers

        # Outlier mask
        mask_raw = apply_outlier_mask(v_raw[:, l])
        mask_resid = apply_outlier_mask(v_resid[:, l])
        n_excluded_raw = int((~mask_raw).sum())
        n_excluded_resid = int((~mask_resid).sum())

        # Raw V_eucl
        sc_raw = fit_scaling_exponent(MAGNITUDES, v_raw[:, l],
                                       bootstrap=do_boot, mask=mask_raw)
        # Sentence-corrected (co-primary)
        sc_resid = fit_scaling_exponent(MAGNITUDES, v_resid[:, l],
                                         bootstrap=do_boot, mask=mask_resid)
        # V_proj (confirmatory robustness)
        sc_proj = fit_scaling_exponent(MAGNITUDES, v_proj[:, l],
                                        bootstrap=do_boot)

        # Noise model comparison (descriptive)
        nm = fit_noise_models(MAGNITUDES, v_raw[:, l])

        # CV
        cv_l = v_raw[:, l] / (centroid_norms[:, l] + 1e-30)
        cv_rho, cv_p = stats.spearmanr(MAGNITUDES, cv_l)

        alpha_raw.append(sc_raw["alpha"])
        alpha_resid.append(sc_resid["alpha"])
        alpha_proj.append(sc_proj["alpha"])

        layer_details.append({
            "layer": l,
            "raw": sc_raw,
            "residual": sc_resid,
            "proj": sc_proj,
            "noise_models": nm,
            "cv_rho": float(cv_rho), "cv_p": float(cv_p),
            "n_excluded_raw": n_excluded_raw,
            "n_excluded_resid": n_excluded_resid,
        })

    alpha_raw = np.array(alpha_raw)
    alpha_resid = np.array(alpha_resid)
    alpha_proj = np.array(alpha_proj)

    # ── Hypothesis tests ──
    pl = np.array(primary_layers)
    ar = alpha_raw[pl]
    ares = alpha_resid[pl]
    ap = alpha_proj[pl]

    def test_hypotheses(alphas, label):
        h1_pos = int((alphas > 0).sum())
        h1_n = len(alphas)
        h1_maj = h1_pos > h1_n / 2
        h1_t, h1_p = stats.ttest_1samp(alphas, 0)

        h2_in = int(((alphas >= H2_WINDOW[0]) & (alphas <= H2_WINDOW[1])).sum())
        h2_maj = h2_in > h1_n / 2
        h2_t, h2_p = stats.ttest_1samp(alphas, 1.0)

        h3_sub = int((alphas < 1).sum())
        h3_maj = h3_sub > h1_n / 2

        print(f"\n  [{label}]")
        print(f"    H1: α>0 at {h1_pos}/{h1_n} layers "
              f"({'PASS' if h1_maj else 'FAIL'}), "
              f"mean={alphas.mean():.4f}, t={h1_t:.2f}, p={h1_p:.4g}")
        print(f"    H2: α∈{H2_WINDOW} at {h2_in}/{h1_n} layers "
              f"({'PASS' if h2_maj else 'FAIL'}), "
              f"t(α≠1)={h2_t:.2f}, p={h2_p:.4g}")
        print(f"    H3: α<1 at {h3_sub}/{h1_n} layers "
              f"({'PASS' if h3_maj else 'FAIL'}), "
              f"median={np.median(alphas):.4f}")

        return {
            "H1_positive_layers": h1_pos, "H1_total": h1_n,
            "H1_majority": bool(h1_maj),
            "H1_t": float(h1_t), "H1_p": float(h1_p),
            "H1_mean_alpha": float(alphas.mean()),
            "H2_in_window": h2_in, "H2_majority": bool(h2_maj),
            "H2_t_vs_1": float(h2_t), "H2_p_vs_1": float(h2_p),
            "H3_subscalar_layers": h3_sub, "H3_majority": bool(h3_maj),
            "H3_median_alpha": float(np.median(alphas)),
        }

    hyp_raw = test_hypotheses(ar, "V_eucl (raw)")
    hyp_resid = test_hypotheses(ares, "V_residual (sentence-corrected)")
    hyp_proj = test_hypotheses(ap, "V_proj (magnitude axis)")

    # OLS vs Theil-Sen divergence check
    ols_ts_diffs = [d["raw"]["alpha_ols_ts_diff"] for d in layer_details
                    if not np.isnan(d["raw"].get("alpha_ols_ts_diff", float('nan')))]
    mean_ols_ts = np.mean(ols_ts_diffs) if ols_ts_diffs else float('nan')
    print(f"\n  OLS–Theil-Sen mean |Δα| = {mean_ols_ts:.4f} "
          f"({'OK' if mean_ols_ts < 0.1 else 'CAUTION: >0.1'})")

    # Raw vs residual divergence check
    delta_raw_resid = abs(ar.mean() - ares.mean())
    print(f"  Raw–Residual mean α divergence = {delta_raw_resid:.4f} "
          f"({'OK' if delta_raw_resid < 0.15 else 'CAUTION: sentence effects present'})")

    # Raw vs V_proj divergence check
    delta_raw_proj = abs(ar.mean() - ap.mean())
    print(f"  Raw–V_proj mean α divergence = {delta_raw_proj:.4f} "
          f"({'OK' if delta_raw_proj < 0.3 else 'CAUTION: high-D concentration'})")

    # H4 (descriptive)
    h4_rho, h4_p = stats.spearmanr(range(n_layers), alpha_raw)
    print(f"\n  H4 (descriptive): layer–α ρ = {h4_rho:.3f}")

    # ── Sensitivity E3 ──
    print("  E3: Sensitivity (exclude boundaries)...")
    mask_no_bound = np.ones(26, dtype=bool)
    mask_no_bound[BOUNDARY_INDICES] = False
    e3_alphas = []
    for l in primary_layers:
        e3 = fit_scaling_exponent(MAGNITUDES, v_raw[:, l],
                                   bootstrap=False, mask=mask_no_bound)
        e3_alphas.append(e3["alpha"])
    e3_alphas = np.array(e3_alphas)
    print(f"  E3: Mean α (no boundaries) = {e3_alphas.mean():.4f} vs {ar.mean():.4f}")

    # ── Plots ──
    print("  Generating figures...")
    mid = n_layers // 2
    peak = int(np.nanargmax(alpha_raw[1:])) + 1

    for li in [1, mid, peak, n_layers - 1]:
        plot_variability_vs_magnitude(
            MAGNITUDES, v_raw[:, li], v_resid[:, li],
            model_label, li, model_out / f"v_vs_mag_L{li:02d}.png")

    # ── Save ──
    results = {
        "model": model_label,
        "shape": list(per_carrier.shape),
        "hypotheses_raw": hyp_raw,
        "hypotheses_residual": hyp_resid,
        "hypotheses_proj": hyp_proj,
        "H4_layer_rho": float(h4_rho),
        "ols_ts_mean_diff": float(mean_ols_ts),
        "raw_resid_divergence": float(delta_raw_resid),
        "raw_proj_divergence": float(delta_raw_proj),
        "layerwise_alpha_raw": [float(a) for a in alpha_raw],
        "layerwise_alpha_residual": [float(a) for a in alpha_resid],
        "layerwise_alpha_proj": [float(a) for a in alpha_proj],
        "sensitivity_e3_mean_alpha": float(e3_alphas.mean()),
        "layer_details": layer_details,
    }

    with open(model_out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Saved to {model_out}")
    return results


def main():
    print("=" * 60)
    print("  Scalar Variability Analysis v2")
    print("  Pre-registered companion to Weber (Cacioli, 2026)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for mk, ml in MODELS.items():
        r = analyse_model(mk, ml)
        if r is not None:
            all_results[mk] = r

    if not all_results:
        print("\n[ERROR] No models processed.")
        return

    # Cross-model figure
    print("\n  Generating cross-model figures...")
    plot_alpha_profile(all_results, RESULTS_DIR / "alpha_profiles.png")

    # Programme-level summary
    print("\n" + "=" * 60)
    print("  PROGRAMME-LEVEL SUMMARY")
    print("=" * 60)

    for measure, key in [("V_eucl (raw)", "hypotheses_raw"),
                          ("V_residual", "hypotheses_residual"),
                          ("V_proj", "hypotheses_proj")]:
        h1 = sum(1 for r in all_results.values() if r[key]["H1_majority"])
        h2 = sum(1 for r in all_results.values() if r[key]["H2_majority"])
        h3 = sum(1 for r in all_results.values() if r[key]["H3_majority"])
        n = len(all_results)
        print(f"\n  [{measure}]")
        print(f"    H1 (α>0):     {h1}/{n} → {'SUPPORTED' if h1 >= 2 else 'NOT SUPPORTED'}")
        print(f"    H2 (α≈1):     {h2}/{n} → {'SUPPORTED' if h2 >= 2 else 'NOT SUPPORTED'}")
        print(f"    H3 (α<1):     {h3}/{n} → {'SUPPORTED' if h3 >= 2 else 'NOT SUPPORTED'}")

    # Save summary
    summary = {k: {
        "raw": r["hypotheses_raw"],
        "residual": r["hypotheses_residual"],
        "proj": r["hypotheses_proj"],
    } for k, r in all_results.items()}

    with open(RESULTS_DIR / "programme_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  All results: {RESULTS_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
