"""
Weber's Law Project 4.2 — Paradigm B & D Figure Generation
Classical Minds, Modern Machines

Pre-registered figures:
    F2: Psychometric curves (accuracy vs ratio, per baseline)
    F3: Weber fraction constancy plot
    F5: Paradigm D dose-response
    F6: Digit-boundary diagnostic (Cohen's d across layers)

Usage:
    python paradigm_bd_figures.py --model llama_instruct
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
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, RESULTS_DIR, PRIMARY_LAYER_RANGE, N_LAYERS_TOTAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── F2: Psychometric curves ──

def figure_f2(model_key: str, results_dir: Path, out_dir: Path):
    """
    F2 (v2.7 Section 14): "Accuracy as a function of magnitude ratio,
    one curve per baseline, with fitted cumulative Gaussians."
    """
    b_path = results_dir / "paradigm_b" / model_key / "numerical" / "paradigm_b_results.json"
    if not b_path.exists():
        log.warning("Paradigm B results not found for F2")
        return

    with open(b_path) as f:
        b_data = json.load(f)

    cells = b_data.get("cells", {})
    fits = b_data.get("psychometric_fits", {})

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

    for idx, (baseline, ratio_data) in enumerate(sorted(cells.items(), key=lambda x: float(x[0]))):
        ratios = sorted([float(r) for r in ratio_data.keys()])
        accuracies = [ratio_data[str(r) if str(r) in ratio_data else r]["accuracy"]
                      for r in ratios]

        # Plot data points
        ax.plot(ratios, accuracies, "o-", color=colors[idx % len(colors)],
                label=f"Baseline {baseline}", markersize=6, linewidth=1.5)

        # Plot fitted psychometric function
        fit = fits.get(str(baseline), fits.get(float(baseline), {}))
        if "mu" in fit:
            x_fit = np.linspace(min(ratios) * 0.9, max(ratios) * 1.1, 100)
            log_x = np.log(x_fit)
            gamma = 0.5
            y_fit = fit["lapse"] + (1 - fit["lapse"] - gamma) * ndtr(
                (log_x - fit["mu"]) / max(fit["sigma"], 1e-6)
            )
            ax.plot(x_fit, y_fit, "--", color=colors[idx % len(colors)],
                    alpha=0.5, linewidth=1)

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.axhline(0.75, color="gray", linestyle="--", alpha=0.3, label="75% threshold")
    ax.set_xlabel("Magnitude Ratio", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"F2: Psychometric Functions — {MODELS[model_key]['hf_id']}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / f"F2_psychometric_{model_key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved F2 psychometric curves")


# ── F3: Weber fraction constancy ──

def figure_f3(model_key: str, results_dir: Path, out_dir: Path):
    """
    F3 (v2.7 Section 14): "Weber fraction (75% threshold) at each baseline,
    with 95% BCa CIs. Horizontal band shows human literature range (0.10-0.25)."
    """
    b_path = results_dir / "paradigm_b" / model_key / "numerical" / "paradigm_b_results.json"
    if not b_path.exists():
        log.warning("Paradigm B results not found for F3")
        return

    with open(b_path) as f:
        b_data = json.load(f)

    boot_wf = b_data.get("bootstrap_weber_fractions", [])
    if not boot_wf:
        log.warning("No bootstrap Weber fractions available for F3")
        return

    baselines = [w["baseline"] for w in boot_wf]
    wfs = [w["weber_fraction_median"] for w in boot_wf]
    ci_lo = [w["ci_95_lo"] for w in boot_wf]
    ci_hi = [w["ci_95_hi"] for w in boot_wf]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Human literature range (v2.7: 0.10-0.25 for numerical)
    ax.axhspan(0.10, 0.25, alpha=0.15, color="green", label="Human range (0.10-0.25)")

    # Plot Weber fractions with CIs
    yerr = [
        [wfs[i] - ci_lo[i] for i in range(len(wfs))],
        [ci_hi[i] - wfs[i] for i in range(len(wfs))],
    ]
    ax.errorbar(
        baselines, wfs, yerr=yerr,
        fmt="ko-", capsize=5, markersize=8, linewidth=2,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Baseline Magnitude", fontsize=12)
    ax.set_ylabel("Weber Fraction", fontsize=12)
    ax.set_title(
        f"F3: Weber Fraction Constancy — {MODELS[model_key]['hf_id']}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / f"F3_weber_constancy_{model_key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved F3 Weber fraction constancy")


# ── F5: Paradigm D dose-response ──

def figure_f5(model_key: str, results_dir: Path, out_dir: Path):
    """
    F5 (v2.7 Section 14): "Comparison probability shift (∆p) as a function
    of patching dose, with 95% CI band for random-direction control."
    """
    d_path = results_dir / "paradigm_d" / model_key / "paradigm_d_results.json"
    if not d_path.exists():
        # Check if no-go
        gate_path = results_dir / "paradigm_d" / model_key / "paradigm_d_gate.json"
        if gate_path.exists():
            with open(gate_path) as f:
                gate = json.load(f)
            if not gate.get("go"):
                log.info("  F5: Paradigm D was NO-GO. Creating placeholder figure.")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "Paradigm D: NO-GO\n(Probe R² < 0.50, no RSA fallback)",
                        ha="center", va="center", fontsize=14, transform=ax.transAxes)
                ax.set_title(f"F5: Dose-Response — {model_key}", fontsize=13)
                fig.savefig(out_dir / f"F5_dose_response_{model_key}.png", dpi=300,
                            bbox_inches="tight")
                plt.close(fig)
                return
        log.warning("Paradigm D results not found for F5")
        return

    with open(d_path) as f:
        d_data = json.load(f)

    mag_results = d_data.get("magnitude_patching", [])
    if not mag_results:
        return

    # Aggregate ∆p by dose
    doses = [0.25, 0.50, 0.75, 1.00]
    dose_deltas = {d: [] for d in doses}

    for r in mag_results:
        for d_info in r.get("doses", []):
            dose = d_info["dose"]
            if dose in dose_deltas:
                dose_deltas[dose].append(abs(d_info["delta_p"]))

    # Load random control
    rand_path = results_dir / "paradigm_d" / model_key / "paradigm_d_random_control.json"
    random_delta_ps = []
    if rand_path.exists():
        with open(rand_path) as f:
            rand_data = json.load(f)
        random_delta_ps = [abs(r["delta_p"]) for r in rand_data]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Magnitude direction
    means = [np.mean(dose_deltas[d]) for d in doses]
    sems = [np.std(dose_deltas[d]) / np.sqrt(len(dose_deltas[d]))
            if dose_deltas[d] else 0 for d in doses]
    ax.errorbar(doses, means, yerr=sems, fmt="bo-", capsize=5,
                markersize=8, linewidth=2, label="Magnitude direction")

    # Random direction threshold
    if random_delta_ps:
        threshold = np.percentile(random_delta_ps, 97.5)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
                    label=f"Random 97.5th pctile ({threshold:.4f})")
        ax.axhspan(0, np.percentile(random_delta_ps, 95),
                    alpha=0.1, color="red", label="Random 95% band")

    ax.set_xlabel("Patching Dose", fontsize=12)
    ax.set_ylabel("|∆p(correct)|", fontsize=12)
    ax.set_title(
        f"F5: Activation Patching Dose-Response — {MODELS[model_key]['hf_id']}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / f"F5_dose_response_{model_key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved F5 dose-response")


# ── F6: Digit-boundary diagnostic ──

def figure_f6(model_key: str, results_dir: Path, out_dir: Path):
    """
    F6 (v2.7 Section 14): "Cohen's d (cross-digit vs within-digit distances)
    across layers. One panel per model."
    """
    pc_path = results_dir / "paradigm_a" / model_key / "numerical" / "paradigm_c_robustness.json"
    if not pc_path.exists():
        log.warning("Paradigm C results not found for F6")
        return

    with open(pc_path) as f:
        pc_data = json.load(f)

    digit_diag = pc_data.get("robustness", {}).get("digit_boundary", {})
    per_layer = digit_diag.get("per_layer", {})

    layers = []
    cohens_ds = []
    for l in range(N_LAYERS_TOTAL):
        layer_data = per_layer.get(str(l), {})
        d = layer_data.get("cohens_d")
        if d is not None:
            layers.append(l)
            cohens_ds.append(d)

    if not layers:
        log.warning("No digit-boundary data for F6")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(layers, cohens_ds, color="steelblue", alpha=0.7, width=0.8)

    # Threshold lines (v2.7: ">0.5 tempered, <0.2 negligible")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="d=0.5 (tempered)")
    ax.axhline(0.2, color="orange", linestyle="--", alpha=0.7, label="d=0.2 (negligible)")
    ax.axhline(0, color="black", linewidth=0.5)

    # Primary layer range
    ax.axvspan(PRIMARY_LAYER_RANGE[0] - 0.5, PRIMARY_LAYER_RANGE[1] - 0.5,
               alpha=0.1, color="green", label="Primary layers")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cohen's d (cross-digit - within-digit)", fontsize=12)
    ax.set_title(
        f"F6: Digit-Boundary Diagnostic — {MODELS[model_key]['hf_id']}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(out_dir / f"F6_digit_boundary_{model_key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved F6 digit-boundary diagnostic")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Paradigm B & D Figure Generation")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    out_dir = args.results_dir / "figures" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Generating Paradigm B & D figures for {args.model}...")

    figure_f2(args.model, args.results_dir, out_dir)
    figure_f3(args.model, args.results_dir, out_dir)
    figure_f5(args.model, args.results_dir, out_dir)
    figure_f6(args.model, args.results_dir, out_dir)

    log.info(f"\nFigures saved to {out_dir}")


if __name__ == "__main__":
    main()
