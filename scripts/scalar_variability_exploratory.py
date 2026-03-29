"""
Scalar Variability — Exploratory Analyses
Companion to scalar_variability_v2.py

Pre-registered exploratory analyses E4, E5, E6 plus
corpus-frequency scatterplot.

Usage:
    cd C:\weber
    .\venv\Scripts\Activate.ps1
    python scripts/scalar_variability_exploratory.py

Requires: scalar_variability_v2.py results already generated.
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
POWER_ANALYSIS_DIR = WEBER_ROOT / "results" / "power_analysis"

MODELS = {
    "llama_instruct": ("Llama-3-8B-Instruct", list(range(16, 32))),
    "mistral_instruct": ("Mistral-7B-Instruct-v0.3", list(range(16, 32))),
    "llama_base": ("Llama-3-8B-Base", list(range(16, 32))),
}

MAGNITUDES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        15, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                        150, 200, 300, 500, 700, 1000])

SEED = 42
np.random.seed(SEED)


# ── Load Data ──────────────────────────────────────────────────

def load_model_data(model_key):
    """Load per-carrier hidden states for a model."""
    path = PARADIGM_A_DIR / model_key / "numerical" / "hidden_states.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["per_carrier"]  # (26, 5, 33, 4096)


def load_corpus_frequencies():
    """
    Load corpus frequency counts for the 26 probing magnitudes.
    Searches multiple locations for the corpus distribution JSON.
    Returns array of shape (26,) or None.
    """
    search_dirs = [
        WEBER_ROOT / "results" / "appendix_e",
        WEBER_ROOT / "results" / "power_analysis",
        WEBER_ROOT / "results" / "corpus_analysis",
        WEBER_ROOT / "results" / "phase0",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for fpath in sorted(search_dir.iterdir()):
            if fpath.suffix != '.json':
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
                for key in ['frequencies', 'counts', 'integer_counts',
                            'magnitude_counts', 'corpus_counts']:
                    if key in data:
                        freq_dict = data[key]
                        freqs = []
                        for mag in MAGNITUDES:
                            freqs.append(freq_dict.get(str(mag), freq_dict.get(mag, 0)))
                        freqs = np.array(freqs, dtype=float)
                        if freqs.sum() > 0:
                            print(f"  [OK] Loaded frequencies from {fpath.name}['{key}']")
                            return freqs
            except Exception as e:
                continue

    print("  [WARN] Could not find corpus frequency data in any location.")
    return None


# ── E5: Corpus Frequency Correlation ──────────────────────────

def run_e5(per_carrier, model_label, primary_layers, freqs, out_dir):
    """
    E5: Correlation between corpus frequency and per-magnitude variability.
    Tests whether V(n) tracks how often n appears in training data.
    """
    print(f"\n  E5: Corpus frequency correlation — {model_label}")

    n_mag, n_car, n_layers, n_dims = per_carrier.shape
    centroids = per_carrier.mean(axis=1)

    # Compute V_eucl at each layer
    deviations = per_carrier - centroids[:, np.newaxis, :, :]
    dev_norms = np.sqrt((deviations ** 2).sum(axis=-1))
    v_eucl = dev_norms.mean(axis=1)  # (26, 33)

    log_freq = np.log(freqs + 1)  # log(count + 1) to handle zeros

    # Correlation at each primary layer
    rhos = []
    for l in primary_layers:
        rho, p = stats.spearmanr(log_freq, v_eucl[:, l])
        rhos.append({"layer": l, "rho": float(rho), "p": float(p)})

    rho_values = np.array([r["rho"] for r in rhos])
    mean_rho = float(rho_values.mean())
    median_rho = float(np.median(rho_values))

    # How many layers show significant positive correlation?
    sig_pos = sum(1 for r in rhos if r["rho"] > 0 and r["p"] < 0.05)

    print(f"    Mean ρ(log_freq, V) = {mean_rho:.3f}")
    print(f"    Median ρ = {median_rho:.3f}")
    print(f"    Sig positive (p<.05): {sig_pos}/{len(primary_layers)} layers")

    # Scatterplot at peak primary layer (highest |rho|)
    best_layer = max(rhos, key=lambda r: abs(r["rho"]))
    bl = best_layer["layer"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(log_freq, v_eucl[:, bl], c='black', s=30, zorder=3)

    # Label a few points
    for i, mag in enumerate(MAGNITUDES):
        if mag in [1, 5, 10, 100, 1000]:
            ax.annotate(str(mag), (log_freq[i], v_eucl[i, bl]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Regression line
    slope, intercept, r, p, _ = stats.linregress(log_freq, v_eucl[:, bl])
    x_fit = np.linspace(log_freq.min(), log_freq.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=1.5)

    ax.set_xlabel('log(corpus frequency)', fontsize=11)
    ax.set_ylabel('V_eucl (representational variability)', fontsize=11)
    ax.set_title(f'{model_label} — Layer {bl}\n'
                 f'ρ = {best_layer["rho"]:.3f}, p = {best_layer["p"]:.4g}',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "e5_freq_vs_variability.png", dpi=200)
    plt.close()

    return {
        "mean_rho": mean_rho,
        "median_rho": median_rho,
        "sig_positive_layers": sig_pos,
        "best_layer": best_layer,
        "per_layer": rhos,
    }


# ── E4: Magnitude-Axis vs Orthogonal Variance ─────────────────

def run_e4(per_carrier, model_label, primary_layers, out_dir):
    """
    E4: Decompose variance into magnitude-axis (PC1) and orthogonal components.
    Tests whether the anti-scalar pattern is magnitude-specific or generic.
    """
    print(f"\n  E4: On-axis vs off-axis variance — {model_label}")

    n_mag, n_car, n_layers, n_dims = per_carrier.shape
    centroids = per_carrier.mean(axis=1)

    results_per_layer = []

    for l in primary_layers:
        cent_l = centroids[:, l, :]  # (26, 4096)
        cent_c = cent_l - cent_l.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(cent_c, full_matrices=False)
        pc1 = Vt[0]  # magnitude axis

        on_axis_var = np.zeros(n_mag)
        off_axis_var = np.zeros(n_mag)

        for m in range(n_mag):
            vecs = per_carrier[m, :, l, :]  # (5, 4096)
            mu = vecs.mean(axis=0)
            devs = vecs - mu  # (5, 4096)

            # Project deviations onto PC1
            proj_on = devs @ pc1  # (5,)
            on_axis_var[m] = proj_on.var(ddof=1)

            # Residual (off-axis)
            proj_off = devs - np.outer(proj_on, pc1)  # (5, 4096)
            off_axis_var[m] = (proj_off ** 2).sum(axis=-1).mean()

        # Scaling exponents for on-axis and off-axis
        log_n = np.log(MAGNITUDES)

        valid_on = on_axis_var > 0
        valid_off = off_axis_var > 0

        if valid_on.sum() >= 5:
            s_on, _, r_on, p_on, _ = stats.linregress(log_n[valid_on],
                                                        np.log(on_axis_var[valid_on]))
        else:
            s_on, r_on, p_on = float('nan'), float('nan'), float('nan')

        if valid_off.sum() >= 5:
            s_off, _, r_off, p_off, _ = stats.linregress(log_n[valid_off],
                                                          np.log(off_axis_var[valid_off]))
        else:
            s_off, r_off, p_off = float('nan'), float('nan'), float('nan')

        # Ratio of on-axis to total variance
        total_var = on_axis_var + off_axis_var
        ratio = np.where(total_var > 0, on_axis_var / total_var, 0)

        results_per_layer.append({
            "layer": l,
            "alpha_on_axis": float(s_on),
            "alpha_off_axis": float(s_off),
            "r2_on": float(r_on ** 2) if not np.isnan(r_on) else float('nan'),
            "r2_off": float(r_off ** 2) if not np.isnan(r_off) else float('nan'),
            "mean_on_axis_ratio": float(ratio.mean()),
        })

    # Summary
    alphas_on = np.array([r["alpha_on_axis"] for r in results_per_layer])
    alphas_off = np.array([r["alpha_off_axis"] for r in results_per_layer])

    print(f"    On-axis (PC1) mean α = {np.nanmean(alphas_on):.4f}")
    print(f"    Off-axis mean α = {np.nanmean(alphas_off):.4f}")
    print(f"    Difference = {np.nanmean(alphas_on) - np.nanmean(alphas_off):.4f}")

    # Is the anti-scalar pattern stronger on the magnitude axis?
    valid = ~(np.isnan(alphas_on) | np.isnan(alphas_off))
    if valid.sum() >= 3:
        t, p = stats.wilcoxon(alphas_on[valid], alphas_off[valid])
        print(f"    Wilcoxon on vs off: p = {p:.4g}")
    else:
        t, p = float('nan'), float('nan')

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    layers = [r["layer"] for r in results_per_layer]
    ax.plot(layers, alphas_on, 'o-', color='#d62728', markersize=4, lw=1.2,
            label=f'On-axis (PC1), mean={np.nanmean(alphas_on):.3f}')
    ax.plot(layers, alphas_off, 's-', color='#1f77b4', markersize=4, lw=1.2,
            label=f'Off-axis, mean={np.nanmean(alphas_off):.3f}')
    ax.axhline(y=0, color='gray', ls=':', alpha=0.4)
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('Scaling exponent α', fontsize=10)
    ax.set_title(f'{model_label} — On-axis vs Off-axis Variance Scaling', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "e4_on_vs_off_axis.png", dpi=200)
    plt.close()

    return {
        "mean_alpha_on": float(np.nanmean(alphas_on)),
        "mean_alpha_off": float(np.nanmean(alphas_off)),
        "wilcoxon_p": float(p),
        "per_layer": results_per_layer,
    }


# ── E6: Instruct vs Base Comparison ───────────────────────────

def run_e6(all_results, out_dir):
    """
    E6: Compare Llama-Instruct vs Llama-Base alpha profiles.
    Tests whether instruction tuning affects noise structure.
    """
    print("\n  E6: Instruct vs Base comparison")

    if "llama_instruct" not in all_results or "llama_base" not in all_results:
        print("    [SKIP] Need both Llama-Instruct and Llama-Base")
        return None

    alpha_inst = np.array(all_results["llama_instruct"]["alphas_raw"])
    alpha_base = np.array(all_results["llama_base"]["alphas_raw"])

    # These should be the same length (16 primary layers)
    assert len(alpha_inst) == len(alpha_base), \
        f"Length mismatch: {len(alpha_inst)} vs {len(alpha_base)}"

    diff = alpha_inst - alpha_base
    mean_diff = float(diff.mean())

    # Wilcoxon signed-rank
    stat, p = stats.wilcoxon(alpha_inst, alpha_base)

    print(f"    Instruct mean α = {alpha_inst.mean():.4f}")
    print(f"    Base mean α = {alpha_base.mean():.4f}")
    print(f"    Mean difference = {mean_diff:.4f}")
    print(f"    Wilcoxon p = {p:.4g}")
    print(f"    Direction: Instruct is {'more' if mean_diff < 0 else 'less'} anti-scalar")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    layers = list(range(16, 32))

    ax = axes[0]
    ax.plot(layers, alpha_inst, 'o-', color='#1f77b4', markersize=4, lw=1.2,
            label=f'Instruct (mean={alpha_inst.mean():.4f})')
    ax.plot(layers, alpha_base, 's-', color='#2ca02c', markersize=4, lw=1.2,
            label=f'Base (mean={alpha_base.mean():.4f})')
    ax.axhline(y=0, color='gray', ls=':', alpha=0.4)
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('Scaling exponent α', fontsize=10)
    ax.set_title('Instruct vs Base: Layerwise α', fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.bar(layers, diff, color=['#d62728' if d < 0 else '#1f77b4' for d in diff],
           alpha=0.7, width=0.8)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('α(Instruct) − α(Base)', fontsize=10)
    ax.set_title(f'Difference (Wilcoxon p = {p:.4g})', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_dir / "e6_instruct_vs_base.png", dpi=200)
    plt.close()

    return {
        "instruct_mean": float(alpha_inst.mean()),
        "base_mean": float(alpha_base.mean()),
        "mean_diff": mean_diff,
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
    }


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Scalar Variability — Exploratory Analyses")
    print("  Pre-registered: E4, E5, E6")
    print("=" * 60)

    # Load corpus frequencies
    print("\n  Loading corpus frequencies...")
    freqs = load_corpus_frequencies()

    all_results = {}

    for model_key, (model_label, primary_layers) in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {model_label}")
        print(f"{'='*60}")

        per_carrier = load_model_data(model_key)
        if per_carrier is None:
            print(f"  [SKIP] No data")
            continue

        out_dir = RESULTS_DIR / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compute V_eucl at primary layers for E6 storage
        centroids = per_carrier.mean(axis=1)
        deviations = per_carrier - centroids[:, np.newaxis, :, :]
        dev_norms = np.sqrt((deviations ** 2).sum(axis=-1))
        v_eucl = dev_norms.mean(axis=1)

        # Store primary-layer alphas for E6
        alphas_raw = []
        for l in primary_layers:
            log_n = np.log(MAGNITUDES)
            log_v = np.log(v_eucl[:, l] + 1e-30)
            s, _, _, _, _ = stats.linregress(log_n, log_v)
            alphas_raw.append(float(s))

        model_results = {"alphas_raw": alphas_raw}

        # E4: On-axis vs off-axis
        e4 = run_e4(per_carrier, model_label, primary_layers, out_dir)
        model_results["e4"] = e4

        # E5: Corpus frequency correlation
        if freqs is not None:
            e5 = run_e5(per_carrier, model_label, primary_layers, freqs, out_dir)
            model_results["e5"] = e5
        else:
            print(f"\n  E5: [SKIP] No corpus frequency data available")

        all_results[model_key] = model_results

    # E6: Instruct vs Base
    e6 = run_e6(all_results, RESULTS_DIR)

    # Save all exploratory results
    save_results = {}
    for mk, mr in all_results.items():
        save_results[mk] = {
            k: v for k, v in mr.items() if k != "alphas_raw"
        }
    if e6 is not None:
        save_results["e6_instruct_vs_base"] = e6

    with open(RESULTS_DIR / "exploratory_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n  All exploratory results saved to {RESULTS_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
