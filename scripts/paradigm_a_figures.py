"""
Weber's Law Project 4.2 — Paradigm A: Figure Generation
Classical Minds, Modern Machines

Pre-registered figures from v2.7 Section 14:
    F1: Layer × domain heatmap of RSA Spearman correlations
    F4: Precision gradient (H3)
    F7: Frequency-matched noun control comparison
    F8: Stevens exponent β across layers (H5)
    F10: Digit-boundary variance partitioning

Usage:
    python paradigm_a_figures.py --model llama_instruct --domain numerical
    python paradigm_a_figures.py --model llama_instruct --domain all --figure all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS, DOMAINS, N_LAYERS_TOTAL, RESULTS_DIR,
    PRIMARY_LAYER_RANGE, BONFERRONI_ALPHA,
    NUMERICAL_MAGNITUDES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_analysis(model_key: str, domain_key: str, results_dir: Path) -> dict:
    """Load Paradigm A analysis results."""
    path = results_dir / "paradigm_a" / model_key / domain_key / "paradigm_a_analysis.json"
    with open(path) as f:
        return json.load(f)


def load_centroids(model_key: str, domain_key: str, results_dir: Path) -> np.ndarray:
    """Load centroid hidden states."""
    path = results_dir / "paradigm_a" / model_key / domain_key / "hidden_states.npz"
    return np.load(path)["centroids"]


# ── F1: RSA heatmap ──

def figure_f1(
    model_key: str,
    domains: list[str],
    results_dir: Path,
    out_dir: Path,
):
    """
    F1: Layer × domain heatmap of RSA Spearman correlations.

    v2.7: "One panel per model, per distance metric."
    Shows correlation with Weber, Linear, Stevens theoretical RDMs.
    """
    for metric in ["cosine", "euclidean"]:
        fig, axes = plt.subplots(
            1, 3, figsize=(18, 6), sharey=True,
            gridspec_kw={"wspace": 0.05},
        )
        fig.suptitle(
            f"F1: RSA Correlations — {MODELS[model_key]['hf_id']} ({metric})",
            fontsize=14, fontweight="bold",
        )

        for col, theo_model in enumerate(["weber", "linear", "stevens"]):
            ax = axes[col]
            # Build heatmap: rows = layers, columns = domains
            heatmap = np.full((N_LAYERS_TOTAL, len(domains)), np.nan)
            p_mask = np.full((N_LAYERS_TOTAL, len(domains)), False)

            for d_idx, domain_key in enumerate(domains):
                try:
                    analysis = load_analysis(model_key, domain_key, results_dir)
                except FileNotFoundError:
                    continue

                for layer in range(N_LAYERS_TOTAL):
                    layer_key = f"layer_{layer:02d}"
                    rsa = (analysis.get("layers", {})
                           .get(layer_key, {})
                           .get(metric, {})
                           .get("rsa", {}))
                    if theo_model in rsa:
                        heatmap[layer, d_idx] = rsa[theo_model]["rho"]
                        p_mask[layer, d_idx] = rsa[theo_model]["p_value"] < BONFERRONI_ALPHA

            im = ax.imshow(
                heatmap, aspect="auto", cmap="RdBu_r",
                vmin=-1, vmax=1,
                origin="lower",
            )

            # Mark significant layers
            for layer in range(N_LAYERS_TOTAL):
                for d_idx in range(len(domains)):
                    if p_mask[layer, d_idx]:
                        ax.plot(d_idx, layer, "k*", markersize=4)

            ax.set_title(theo_model.capitalize(), fontsize=12)
            ax.set_xticks(range(len(domains)))
            ax.set_xticklabels([d[:3].upper() for d in domains], fontsize=10)

            if col == 0:
                ax.set_ylabel("Layer", fontsize=11)
                ax.set_yticks(range(0, N_LAYERS_TOTAL, 4))

            # Draw primary layer range
            ax.axhspan(
                PRIMARY_LAYER_RANGE[0] - 0.5,
                PRIMARY_LAYER_RANGE[1] - 0.5,
                alpha=0.1, color="green", linewidth=0,
            )
            ax.axhline(PRIMARY_LAYER_RANGE[0] - 0.5, color="green", linewidth=0.5, linestyle="--")
            ax.axhline(PRIMARY_LAYER_RANGE[1] - 0.5, color="green", linewidth=0.5, linestyle="--")

        # Colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="Spearman ρ")

        # Legend
        fig.text(
            0.02, 0.02,
            "★ = Mantel p < 0.017 | Green band = primary layers (16-31)",
            fontsize=9, style="italic",
        )

        fig.savefig(
            out_dir / f"F1_rsa_heatmap_{model_key}_{metric}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)
        log.info(f"  Saved F1 heatmap ({metric})")


# ── F4: Precision gradient (H3) ──

def figure_f4(
    model_key: str,
    domain_key: str,
    results_dir: Path,
    out_dir: Path,
):
    """
    F4: Precision gradient — local representational precision as function of magnitude.

    v2.7 Section 5.5: "precision = 1 / ||h(n+1) - h(n)||"
    v2.7 Section 14: "Raw and log-step-normalised. One panel per domain per model."
    """
    centroids = load_centroids(model_key, domain_key, results_dir)
    analysis = load_analysis(model_key, domain_key, results_dir)
    magnitudes = np.array(analysis["magnitudes"])

    n_mags = len(magnitudes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"F4: Precision Gradient — {MODELS[model_key]['hf_id']} / {domain_key}",
        fontsize=13, fontweight="bold",
    )

    # Select a few representative layers from the primary range
    display_layers = list(range(PRIMARY_LAYER_RANGE[0], PRIMARY_LAYER_RANGE[1], 2))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(display_layers)))

    for ax_idx, (title, normalise) in enumerate([
        ("Raw precision", False),
        ("Log-step normalised", True),
    ]):
        ax = axes[ax_idx]

        for c_idx, layer in enumerate(display_layers):
            vecs = centroids[:, layer, :]  # (n_mags, d_model)

            # Local precision: 1 / ||h(n+1) - h(n)||
            diffs = np.linalg.norm(np.diff(vecs, axis=0), axis=1)  # (n_mags-1,)
            precision = 1.0 / np.where(diffs == 0, 1e-10, diffs)

            # Midpoints for x-axis
            midpoints = (magnitudes[:-1] + magnitudes[1:]) / 2

            if normalise:
                # Normalise by log step size
                log_steps = np.diff(np.log(magnitudes))
                precision = precision * log_steps

            ax.plot(
                midpoints, precision,
                color=cmap[c_idx], alpha=0.7,
                label=f"L{layer}" if c_idx % 3 == 0 else None,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Magnitude (midpoint)", fontsize=11)
        ax.set_ylabel("Precision" + (" × log-step" if normalise else ""), fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.savefig(
        out_dir / f"F4_precision_gradient_{model_key}_{domain_key}.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)
    log.info(f"  Saved F4 precision gradient")


# ── F7: Frequency-matched noun control ──

def figure_f7(
    model_key: str,
    results_dir: Path,
    out_dir: Path,
):
    """
    F7: Frequency-matched noun control — number RDM vs noun RDM.

    v2.7 Section 14: "Side-by-side heatmaps with RSA correlation comparison."
    """
    # Load number centroids
    try:
        num_path = results_dir / "paradigm_a" / model_key / "numerical" / "hidden_states.npz"
        num_centroids = np.load(num_path)["centroids"]
    except FileNotFoundError:
        log.warning("Numerical centroids not found for F7")
        return

    # Load noun centroids
    try:
        noun_path = results_dir / "paradigm_a" / model_key / "freq_nouns" / "hidden_states.npz"
        noun_centroids = np.load(noun_path)["centroids"]
    except FileNotFoundError:
        log.warning("Noun centroids not found for F7. Run extraction with --include-nouns.")
        return

    # Pick a representative layer (middle of primary range)
    layer = (PRIMARY_LAYER_RANGE[0] + PRIMARY_LAYER_RANGE[1]) // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"F7: Frequency-Matched Control — {MODELS[model_key]['hf_id']} (Layer {layer})",
        fontsize=13, fontweight="bold",
    )

    # Number RDM
    num_vecs = num_centroids[:, layer, :]
    num_rdm = squareform(pdist(num_vecs, metric="cosine"))
    im0 = axes[0].imshow(num_rdm, cmap="viridis")
    axes[0].set_title("Number RDM (cosine)", fontsize=11)
    axes[0].set_xlabel("Magnitude index")
    axes[0].set_ylabel("Magnitude index")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Noun RDM
    noun_vecs = noun_centroids[:, layer, :]
    noun_rdm = squareform(pdist(noun_vecs, metric="cosine"))
    im1 = axes[1].imshow(noun_rdm, cmap="viridis")
    axes[1].set_title("Noun RDM (cosine)", fontsize=11)
    axes[1].set_xlabel("Noun index")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # RSA comparison across layers
    layers = range(N_LAYERS_TOTAL)
    rsa_numbers_weber = []
    rsa_nouns_weber = []

    from paradigm_a_analyse import build_theoretical_rdms
    from config import NUMERICAL_MAGNITUDES

    theo_rdms = build_theoretical_rdms(NUMERICAL_MAGNITUDES)
    weber_theo = theo_rdms["weber"]

    for l in layers:
        # Number RSA
        nv = num_centroids[:, l, :]
        if not np.any(np.isnan(nv)):
            nd = pdist(nv, metric="cosine")
            nd_z = (nd - nd.mean()) / (nd.std() + 1e-10)
            rho, _ = spearmanr(nd_z, weber_theo)
            rsa_numbers_weber.append(rho)
        else:
            rsa_numbers_weber.append(np.nan)

        # Noun RSA (against same Weber RDM — should NOT correlate if control works)
        nv2 = noun_centroids[:, l, :]
        if not np.any(np.isnan(nv2)):
            nd2 = pdist(nv2, metric="cosine")
            nd2_z = (nd2 - nd2.mean()) / (nd2.std() + 1e-10)
            rho2, _ = spearmanr(nd2_z, weber_theo)
            rsa_nouns_weber.append(rho2)
        else:
            rsa_nouns_weber.append(np.nan)

    axes[2].plot(list(layers), rsa_numbers_weber, "b-o", markersize=3, label="Numbers")
    axes[2].plot(list(layers), rsa_nouns_weber, "r-s", markersize=3, label="Freq-matched nouns")
    axes[2].axhspan(
        PRIMARY_LAYER_RANGE[0], PRIMARY_LAYER_RANGE[1],
        alpha=0.1, color="green",
    )
    axes[2].set_xlabel("Layer", fontsize=11)
    axes[2].set_ylabel("Spearman ρ with Weber RDM", fontsize=11)
    axes[2].set_title("RSA: Weber correlation", fontsize=11)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.savefig(
        out_dir / f"F7_freq_noun_control_{model_key}.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)
    log.info("  Saved F7 frequency-matched noun control")


# ── F8: Stevens exponent across layers (H5) ──

def figure_f8(
    model_key: str,
    domains: list[str],
    results_dir: Path,
    out_dir: Path,
):
    """
    F8: Stevens exponent β across layers.

    v2.7 Section 14: "One line per domain per model."
    H5: "Spearman ρ between layer index and estimated Stevens exponent β
    is significantly negative."
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"F8: Stevens Exponent β — {MODELS[model_key]['hf_id']}",
        fontsize=13, fontweight="bold",
    )

    domain_colors = {"numerical": "blue", "temporal": "red", "spatial": "green"}

    for domain_key in domains:
        try:
            analysis = load_analysis(model_key, domain_key, results_dir)
        except FileNotFoundError:
            continue

        layers = []
        betas = []
        for layer in range(N_LAYERS_TOTAL):
            layer_key = f"layer_{layer:02d}"
            fits = (analysis.get("layers", {})
                    .get(layer_key, {})
                    .get("cosine", {})
                    .get("model_fits", {}))
            stevens = fits.get("stevens", {})
            if "params" in stevens:
                layers.append(layer)
                betas.append(stevens["params"]["beta"])

        if layers:
            color = domain_colors.get(domain_key, "gray")
            ax.plot(layers, betas, "-o", color=color, markersize=4,
                    label=domain_key.capitalize())

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="β=1 (linear)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="β=0.5 (√n)")
    ax.axhspan(
        ax.get_ylim()[0],
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 2,
        xmin=(PRIMARY_LAYER_RANGE[0]) / N_LAYERS_TOTAL,
        xmax=(PRIMARY_LAYER_RANGE[1]) / N_LAYERS_TOTAL,
        alpha=0.1, color="green",
    )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Stevens exponent β", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(
        out_dir / f"F8_stevens_exponent_{model_key}.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)
    log.info("  Saved F8 Stevens exponent")


# ── F10: Digit-boundary variance partitioning ──

def figure_f10(
    model_key: str,
    results_dir: Path,
    out_dir: Path,
):
    """
    F10: Partial R² for log-magnitude vs digit-count predictors across layers.

    v2.7 Section 5.7.1 (variance-partitioning supplement):
    "Regress the full RDM distances (325 pairs) simultaneously on two predictors:
    (a) |log(ni) - log(nj)|  and  (b) |digits(ni) - digits(nj)|.
    Report partial R² for each predictor at each layer."
    """
    try:
        centroids = load_centroids(model_key, "numerical", results_dir)
    except FileNotFoundError:
        log.warning("Numerical centroids not found for F10")
        return

    magnitudes = np.array(NUMERICAL_MAGNITUDES, dtype=float)
    n = len(magnitudes)
    pairs = list(__import__("itertools").combinations(range(n), 2))

    # Predictors
    log_mag_diffs = np.array([abs(np.log(magnitudes[i]) - np.log(magnitudes[j])) for i, j in pairs])
    digit_diffs = np.array([abs(len(str(int(magnitudes[i]))) - len(str(int(magnitudes[j])))) for i, j in pairs])

    partial_r2_logmag = []
    partial_r2_digits = []

    for layer in range(N_LAYERS_TOTAL):
        vecs = centroids[:, layer, :]
        if np.any(np.isnan(vecs)):
            partial_r2_logmag.append(np.nan)
            partial_r2_digits.append(np.nan)
            continue

        y = pdist(vecs, metric="cosine")

        # Full model: y ~ logmag + digits
        X_full = np.column_stack([np.ones(len(y)), log_mag_diffs, digit_diffs])
        beta_full, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
        ss_res_full = np.sum((y - X_full @ beta_full) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        # Reduced: y ~ digits only
        X_reduced_digits = np.column_stack([np.ones(len(y)), digit_diffs])
        beta_rd, _, _, _ = np.linalg.lstsq(X_reduced_digits, y, rcond=None)
        ss_res_digits_only = np.sum((y - X_reduced_digits @ beta_rd) ** 2)

        # Reduced: y ~ logmag only
        X_reduced_logmag = np.column_stack([np.ones(len(y)), log_mag_diffs])
        beta_rl, _, _, _ = np.linalg.lstsq(X_reduced_logmag, y, rcond=None)
        ss_res_logmag_only = np.sum((y - X_reduced_logmag @ beta_rl) ** 2)

        # Partial R² for logmag = (SS_reduced_without_logmag - SS_full) / SS_reduced_without_logmag
        pr2_lm = (ss_res_digits_only - ss_res_full) / ss_res_digits_only if ss_res_digits_only > 0 else 0
        pr2_dg = (ss_res_logmag_only - ss_res_full) / ss_res_logmag_only if ss_res_logmag_only > 0 else 0

        partial_r2_logmag.append(float(pr2_lm))
        partial_r2_digits.append(float(pr2_dg))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(N_LAYERS_TOTAL), partial_r2_logmag, "b-o", markersize=4,
            label="Log-magnitude (partial R²)")
    ax.plot(range(N_LAYERS_TOTAL), partial_r2_digits, "r-s", markersize=4,
            label="Digit-count (partial R²)")

    ax.axvspan(
        PRIMARY_LAYER_RANGE[0] - 0.5, PRIMARY_LAYER_RANGE[1] - 0.5,
        alpha=0.1, color="green", label="Primary layers",
    )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Partial R²", fontsize=11)
    ax.set_title(
        f"F10: Variance Partitioning — {MODELS[model_key]['hf_id']}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(
        out_dir / f"F10_variance_partitioning_{model_key}.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)
    log.info("  Saved F10 variance partitioning")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Paradigm A: Figure Generation")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--domain", default="all", choices=list(DOMAINS.keys()) + ["all"])
    parser.add_argument("--figure", default="all",
                        choices=["F1", "F4", "F7", "F8", "F10", "all"])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    domains = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]
    out_dir = args.results_dir / "figures" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = [args.figure] if args.figure != "all" else ["F1", "F4", "F7", "F8", "F10"]

    log.info(f"Generating figures: {figs}")

    if "F1" in figs:
        log.info("Generating F1 (RSA heatmap)...")
        figure_f1(args.model, domains, args.results_dir, out_dir)

    if "F4" in figs:
        for d in domains:
            log.info(f"Generating F4 (precision gradient) for {d}...")
            try:
                figure_f4(args.model, d, args.results_dir, out_dir)
            except FileNotFoundError:
                log.warning(f"  Skipping F4 for {d}: data not found")

    if "F7" in figs:
        log.info("Generating F7 (frequency-matched noun control)...")
        figure_f7(args.model, args.results_dir, out_dir)

    if "F8" in figs:
        log.info("Generating F8 (Stevens exponent)...")
        figure_f8(args.model, domains, args.results_dir, out_dir)

    if "F10" in figs:
        log.info("Generating F10 (variance partitioning)...")
        figure_f10(args.model, args.results_dir, out_dir)

    log.info(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
