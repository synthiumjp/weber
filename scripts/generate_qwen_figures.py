"""
generate_qwen_figures.py — Qwen exploratory model figure panels
Generates:
  1. RSA heatmap (matching Figure 1 style) for Qwen
  2. B1 accuracy-by-ratio curve (matching Figure 2 style) for Qwen
  3. Cross-model comparison bar chart (new — all 3 models)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

OUT_DIR = r"C:\weber\results\exploratory\qwen25_7b"

# ── Load Qwen Paradigm A results ──
with open(f"{OUT_DIR}\\paradigm_a_analysis.json") as f:
    a_data = json.load(f)

# ── Load Qwen B1 results ──
with open(f"{OUT_DIR}\\paradigm_b1_crossformat.json") as f:
    b1_data = json.load(f)

# ═══════════════════════════════════════════════════════════════
# FIGURE: Qwen RSA Heatmap (matches Figure 1 style)
# ═══════════════════════════════════════════════════════════════

def plot_rsa_heatmap():
    layers = sorted(a_data["layers"].keys(), key=lambda x: int(x.split("_")[1]))
    
    # Extract Weber RSA rho at each layer (cosine)
    weber_rhos = []
    linear_rhos = []
    weber_r2s = []
    linear_r2s = []
    layer_nums = []
    
    for lk in layers:
        layer_num = int(lk.split("_")[1])
        layer_nums.append(layer_num)
        
        cos_data = a_data["layers"][lk].get("cosine", {})
        rsa = cos_data.get("rsa", {})
        fits = cos_data.get("model_fits", {})
        
        weber_rhos.append(rsa.get("weber", {}).get("rho", 0))
        linear_rhos.append(rsa.get("linear_rho", 0))
        weber_r2s.append(fits.get("weber", {}).get("r2", 0))
        linear_r2s.append(fits.get("linear", {}).get("r2", 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    
    # Left: RSA rho across layers
    ax = axes[0]
    ax.plot(layer_nums, weber_rhos, "o-", color="#2166ac", label="Weber (log)", linewidth=2, markersize=4)
    ax.plot(layer_nums, linear_rhos, "s--", color="#b2182b", label="Linear", linewidth=1.5, markersize=3, alpha=0.7)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("RSA Spearman ρ", fontsize=11)
    ax.set_title("Qwen-2.5-7B-Instruct: RSA (cosine)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    
    # Shade primary layer range
    summary = a_data.get("summary", {})
    p_start = summary.get("primary_layer_range", [14, 27])[0]
    p_end = summary.get("primary_layer_range", [14, 27])[1]
    ax.axvspan(p_start, p_end, alpha=0.1, color="blue", label="Primary range")
    
    # Right: R² comparison
    ax = axes[1]
    x = np.arange(len(layer_nums))
    width = 0.35
    ax.bar(x - width/2, weber_r2s, width, color="#2166ac", label="Weber R²", alpha=0.8)
    ax.bar(x + width/2, linear_r2s, width, color="#b2182b", label="Linear R²", alpha=0.8)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("Qwen-2.5-7B-Instruct: Model Fit", fontsize=12, fontweight="bold")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([str(l) for l in layer_nums[::4]])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}\\Qwen_RSA_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}\\Qwen_RSA_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: Qwen_RSA_heatmap.png/pdf")


# ═══════════════════════════════════════════════════════════════
# FIGURE: Qwen B1 Accuracy by Ratio (matches Figure 2 style)
# ═══════════════════════════════════════════════════════════════

def plot_b1_accuracy():
    ratio_data = b1_data["accuracy_by_ratio"]
    
    ratios = sorted([float(k) for k in ratio_data.keys()])
    accs = [ratio_data[str(r)]["accuracy"] for r in ratios]
    ns = [ratio_data[str(r)]["total"] for r in ratios]
    
    # Compute 95% CI (Wilson interval)
    cis = []
    for r in ratios:
        d = ratio_data[str(r)]
        p = d["accuracy"]
        n = d["total"]
        z = 1.96
        denominator = 1 + z**2/n
        centre = (p + z**2/(2*n)) / denominator
        spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
        cis.append(spread)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left: Accuracy by ratio
    ax = axes[0]
    ax.errorbar(ratios, accs, yerr=cis, fmt="o-", color="#1b7837", linewidth=2,
                markersize=6, capsize=4, label="Qwen-2.5-7B")
    ax.set_xlabel("Magnitude Ratio", fontsize=11)
    ax.set_ylabel("B1 Accuracy", fontsize=11)
    ax.set_title("Qwen B1 Cross-Format: Accuracy by Ratio", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--", label="Chance")
    ax.set_xscale("log")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r:.2f}" for r in ratios], fontsize=9)
    ax.legend(fontsize=9)
    
    # Right: Cross-model comparison
    ax = axes[1]
    models = ["Llama-3\n8B-Inst", "Mistral\n7B-Inst", "Qwen-2.5\n7B-Inst"]
    
    # Llama ratios (from session log)
    llama_ratios = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
    llama_accs = [0.752, 0.724, 0.748, 0.768, 0.828, 0.892]
    
    # Mistral ratios (from session log)
    mistral_accs = [0.732, 0.724, 0.712, 0.720, 0.776, 0.824]
    
    # Qwen ratios
    qwen_accs = accs
    
    ax.plot(ratios, llama_accs, "D-", color="#2166ac", linewidth=2, markersize=5, label="Llama")
    ax.plot(ratios, mistral_accs, "s--", color="#b2182b", linewidth=1.5, markersize=5, label="Mistral")
    ax.plot(ratios, qwen_accs, "o-", color="#1b7837", linewidth=2, markersize=5, label="Qwen")
    
    ax.set_xlabel("Magnitude Ratio", fontsize=11)
    ax.set_ylabel("B1 Accuracy", fontsize=11)
    ax.set_title("Cross-Model B1 Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r:.2f}" for r in ratios], fontsize=9)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}\\Qwen_B1_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}\\Qwen_B1_accuracy.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: Qwen_B1_accuracy.png/pdf")


# ═══════════════════════════════════════════════════════════════
# FIGURE: Cross-Model Summary (new composite)
# ═══════════════════════════════════════════════════════════════

def plot_cross_model_summary():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    models = ["Llama-3-8B\nInstruct", "Mistral-7B\nInstruct", "Qwen-2.5-7B\nInstruct"]
    colors = ["#2166ac", "#b2182b", "#1b7837"]
    
    # Panel A: Stevens β
    betas = [0.01, 0.01, 0.01]
    ax = axes[0]
    bars = ax.bar(models, betas, color=colors, alpha=0.8, width=0.6)
    ax.set_ylabel("Stevens β", fontsize=11)
    ax.set_title("(a) Compression Exponent", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.05)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.text(0.5, 0.85, "β → 0 = logarithmic", transform=ax.transAxes,
            fontsize=9, ha="center", style="italic", color="gray")
    for bar, b in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{b:.3f}", ha="center", fontsize=9)
    
    # Panel B: B1 Overall Accuracy
    b1_accs = [0.785, 0.748, 0.839]
    ax = axes[1]
    bars = ax.bar(models, b1_accs, color=colors, alpha=0.8, width=0.6)
    ax.set_ylabel("B1 Accuracy", fontsize=11)
    ax.set_title("(b) Cross-Format Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
    for bar, a in zip(bars, b1_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{a:.1%}", ha="center", fontsize=9)
    
    # Panel C: Symbolic Control
    sym_accs = [0.999, 0.500, 0.999]
    ax = axes[2]
    bars = ax.bar(models, sym_accs, color=colors, alpha=0.8, width=0.6)
    ax.set_ylabel("Symbolic Accuracy", fontsize=11)
    ax.set_title("(c) Symbolic Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.1)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
    for bar, a in zip(bars, sym_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{a:.1%}", ha="center", fontsize=9)
    
    plt.tight_layout()
    out_base = r"C:\weber\results\paper_figures\Figure7_cross_model"
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: Figure7_cross_model.png/pdf")


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Qwen exploratory figures...\n")
    plot_rsa_heatmap()
    plot_b1_accuracy()
    plot_cross_model_summary()
    print("\nAll figures generated.")
