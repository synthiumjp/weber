"""
M3 Pilot Analysis — Categorical Perception Go/No-Go
====================================================
Paper M3, "Classical Minds, Modern Machines" programme.
Author: JP Cacioli
Research assistant: Claude (Anthropic)

Go/no-go analysis for M3 pilot:
  1. Compute empirical RDM (cosine + Euclidean) from centroids
  2. Construct theoretical RDMs: Continuous, CP-Additive, Categorical, Linear
  3. RSA: Spearman correlation + Mantel permutation test (10K) per theoretical RDM per layer
  4. Precision gradient: local distance between adjacent values, look for dip at boundary
  5. Identification: extract logit probabilities for category labels, fit sigmoid
  6. Visualisation: RDM heatmap, precision gradient plot, identification function,
     RSA rho comparison (Continuous vs CP-Additive across layers)

Go/no-go question: Does the RDM show any warping at the decade boundary (10)
that is absent at the control position (15)?

Seed = 42. All analyses follow m3_project_outline.md v0.4.
"""

import json
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional

SEED = 42
rng = np.random.default_rng(SEED)

# =============================================================================
# 1. Configuration
# =============================================================================

class AnalysisConfig:
    extraction_dir: Path = Path("extractions")
    stimulus_dir: Path = Path("stimuli")
    output_dir: Path = Path("results")
    model_short: str = "llama3-8b-instruct"
    conditions: Tuple[str, ...] = ("decade_10", "control_15")
    n_permutations: int = 10_000  # Mantel test
    seed: int = SEED
    # Primary layers for focused analysis (following Weber: middle-to-late layers)
    # For Llama-3-8B: layers 8-24 are typically primary
    primary_layer_range: Tuple[int, int] = (8, 25)


# =============================================================================
# 2. Load Data
# =============================================================================

def load_centroids(config: AnalysisConfig, condition: str):
    """Load RSA centroids and metadata for a condition."""
    npz_path = config.extraction_dir / f"m3_centroids_{condition}_{config.model_short}.npz"
    meta_path = config.extraction_dir / f"m3_meta_{condition}_{config.model_short}.json"
    
    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)
    
    return {
        "rsa_centroids": data["rsa_centroids"],  # (n_values, n_layers, d_model)
        "b0_centroids": data["b0_centroids"],
        "values": data["values"],
        "meta": meta,
    }


def load_stimuli(config: AnalysisConfig, condition: str):
    """Load stimulus file for theoretical RDMs and pair info."""
    stim_path = config.stimulus_dir / f"m3_stimuli_{condition}.json"
    with open(stim_path) as f:
        return json.load(f)


# =============================================================================
# 3. Empirical RDM Computation
# =============================================================================

def compute_empirical_rdm(
    centroids: np.ndarray,  # (n_values, d_model)
    metric: str = "cosine",
) -> np.ndarray:
    """Compute the empirical representational dissimilarity matrix.
    
    Following Weber: cosine and Euclidean as co-primary metrics.
    Returns symmetric n_values × n_values matrix.
    """
    if metric == "cosine":
        # scipy pdist returns condensed form; squareform expands
        dists = pdist(centroids, metric="cosine")
    elif metric == "euclidean":
        dists = pdist(centroids, metric="euclidean")
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return squareform(dists)


def compute_rdms_all_layers(
    rsa_centroids: np.ndarray,  # (n_values, n_layers, d_model)
    metric: str = "cosine",
) -> np.ndarray:
    """Compute empirical RDMs at all layers.
    
    Returns: (n_layers, n_values, n_values)
    """
    n_values, n_layers, d_model = rsa_centroids.shape
    rdms = np.zeros((n_layers, n_values, n_values))
    
    for layer in range(n_layers):
        centroids_at_layer = rsa_centroids[:, layer, :]
        rdms[layer] = compute_empirical_rdm(centroids_at_layer, metric)
    
    return rdms


# =============================================================================
# 4. Theoretical RDMs
# =============================================================================

def build_theoretical_rdms(
    values: np.ndarray,
    boundary: int,
) -> Dict[str, np.ndarray]:
    """Build all theoretical RDMs for RSA model comparison.
    
    Following Section 5.1 of v0.4:
      1. Continuous (Weber/Log): d_ij = |log(x_i) - log(x_j)|
      2. CP-Additive: d_ij = |log(x_i) - log(x_j)| + lambda * 1[diff category]
      3. Categorical: d_ij = 0 if same, 1 if different
      4. Linear (null): d_ij = |x_i - x_j|
    
    Lambda = 1.0 for template (relative weighting tested via regression).
    """
    n = len(values)
    log_vals = np.log(values.astype(float))
    
    rdms = {}
    
    # Continuous
    cont = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cont[i, j] = abs(log_vals[i] - log_vals[j])
    rdms["continuous"] = cont
    
    # Categorical
    cat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ci = 0 if values[i] < boundary else 1
            cj = 0 if values[j] < boundary else 1
            cat[i, j] = 0.0 if ci == cj else 1.0
    rdms["categorical"] = cat
    
    # CP-Additive (lambda = 1.0)
    cp_add = cont.copy() + 1.0 * cat
    rdms["cp_additive"] = cp_add
    
    # Linear
    lin = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lin[i, j] = abs(float(values[i]) - float(values[j]))
    rdms["linear"] = lin
    
    return rdms


# =============================================================================
# 5. RSA: Spearman Correlation + Mantel Permutation Test
# =============================================================================

def rdm_to_condensed(rdm: np.ndarray) -> np.ndarray:
    """Extract upper triangle of symmetric RDM as condensed vector."""
    return squareform(rdm, checks=False)


def mantel_test(
    rdm_empirical: np.ndarray,
    rdm_theoretical: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = SEED,
) -> Tuple[float, float]:
    """Mantel permutation test for RSA significance.
    
    Following Weber / Kriegeskorte et al. (2008):
      - Compute Spearman rho between condensed RDMs
      - Permute rows/columns of empirical RDM
      - Count how often permuted rho >= observed rho
    
    Returns: (rho, p_value)
    """
    rng_local = np.random.default_rng(seed)
    
    emp_condensed = rdm_to_condensed(rdm_empirical)
    theo_condensed = rdm_to_condensed(rdm_theoretical)
    
    # Observed Spearman rho
    rho_observed, _ = stats.spearmanr(emp_condensed, theo_condensed)
    
    # Permutation distribution
    n = rdm_empirical.shape[0]
    count_ge = 0
    
    for _ in range(n_permutations):
        perm = rng_local.permutation(n)
        rdm_perm = rdm_empirical[np.ix_(perm, perm)]
        perm_condensed = rdm_to_condensed(rdm_perm)
        rho_perm, _ = stats.spearmanr(perm_condensed, theo_condensed)
        if rho_perm >= rho_observed:
            count_ge += 1
    
    p_value = (count_ge + 1) / (n_permutations + 1)  # +1 for observed
    
    return float(rho_observed), float(p_value)


def rsa_all_layers(
    empirical_rdms: np.ndarray,   # (n_layers, n_values, n_values)
    theoretical_rdms: Dict[str, np.ndarray],
    n_permutations: int = 10_000,
    seed: int = SEED,
) -> Dict[str, Dict[str, List]]:
    """Run RSA (Spearman + Mantel) for each theoretical RDM at each layer.
    
    Returns nested dict: {theo_name: {"rho": [...], "p": [...], "layers": [...]}}
    """
    n_layers = empirical_rdms.shape[0]
    results = {}
    
    for theo_name, theo_rdm in theoretical_rdms.items():
        rhos = []
        ps = []
        print(f"  RSA: {theo_name}", end="", flush=True)
        
        for layer in range(n_layers):
            rho, p = mantel_test(
                empirical_rdms[layer], theo_rdm,
                n_permutations=n_permutations,
                seed=seed + layer,  # Different seed per layer for independence
            )
            rhos.append(rho)
            ps.append(p)
            
            if (layer + 1) % 10 == 0:
                print(".", end="", flush=True)
        
        print(f" done (max rho={max(rhos):.3f})")
        results[theo_name] = {
            "rho": rhos,
            "p": ps,
            "layers": list(range(n_layers)),
        }
    
    return results


# =============================================================================
# 6. Precision Gradient (Paradigm C)
# =============================================================================

def compute_precision_gradient(
    rsa_centroids: np.ndarray,  # (n_values, n_layers, d_model)
    values: np.ndarray,
    metric: str = "cosine",
) -> Dict[str, np.ndarray]:
    """Compute local precision gradient along the continuum.
    
    Local distance between adjacent values at each layer.
    Precision = 1 / distance (higher = more compressed = less discriminable).
    
    CP prediction: precision should DECREASE at boundary
    (= distance INCREASES = representations more spread out).
    
    Returns:
      "distances": (n_layers, n_values-1) — distance between adjacent values
      "precision": (n_layers, n_values-1) — 1/distance
      "midpoints": (n_values-1,) — midpoint values for x-axis
    """
    n_values, n_layers, d_model = rsa_centroids.shape
    
    distances = np.zeros((n_layers, n_values - 1))
    
    for layer in range(n_layers):
        for i in range(n_values - 1):
            v1 = rsa_centroids[i, layer, :]
            v2 = rsa_centroids[i + 1, layer, :]
            
            if metric == "cosine":
                # Cosine distance
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                distances[layer, i] = 1.0 - cos_sim
            else:
                distances[layer, i] = np.linalg.norm(v1 - v2)
    
    # Precision = 1/distance (with floor to avoid inf)
    precision = 1.0 / np.maximum(distances, 1e-10)
    
    # Midpoint values for plotting
    midpoints = (values[:-1] + values[1:]) / 2.0
    
    return {
        "distances": distances,
        "precision": precision,
        "midpoints": midpoints,
    }


# =============================================================================
# 7. Identification Analysis (Paradigm B0)
# =============================================================================

def sigmoid(x, a, b, c, d):
    """Sigmoid function: d + (c-d) / (1 + exp(-a * (x - b)))
    
    a = slope, b = crossover point, c = upper asymptote, d = lower asymptote
    """
    return d + (c - d) / (1.0 + np.exp(-a * (x - b)))


def fit_identification_sigmoid(
    values: np.ndarray,
    prob_category_b: np.ndarray,
) -> Dict[str, float]:
    """Fit a sigmoid to the identification function.
    
    Following Section 5.0 of v0.4:
      - Crossover point (P(category_b) = 0.50) defines the empirical boundary
      - Slope at crossover defines boundary sharpness
    
    Returns dict with fitted parameters and derived measures.
    """
    try:
        popt, pcov = curve_fit(
            sigmoid,
            values.astype(float),
            prob_category_b,
            p0=[1.0, np.median(values), 1.0, 0.0],  # initial guess
            bounds=(
                [-np.inf, min(values), 0.0, -0.5],
                [np.inf, max(values), 1.5, 0.5]
            ),
            maxfev=10000,
        )
        a, b, c, d = popt
        
        # Crossover: where sigmoid = 0.5
        # For ideal sigmoid (c=1, d=0), crossover = b
        crossover = b
        
        # Slope at crossover
        slope_at_crossover = a * (c - d) / 4.0  # derivative of sigmoid at midpoint
        
        # Goodness of fit
        y_pred = sigmoid(values.astype(float), *popt)
        ss_res = np.sum((prob_category_b - y_pred) ** 2)
        ss_tot = np.sum((prob_category_b - np.mean(prob_category_b)) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-10)
        
        return {
            "a_slope": float(a),
            "b_crossover": float(b),
            "c_upper": float(c),
            "d_lower": float(d),
            "crossover": float(crossover),
            "slope_at_crossover": float(slope_at_crossover),
            "r_squared": float(r_squared),
            "fitted": True,
        }
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Sigmoid fit failed: {e}")
        return {
            "fitted": False,
            "error": str(e),
        }


def analyse_identification(
    meta: dict,
) -> Dict[str, dict]:
    """Analyse identification results from Paradigm B0.
    
    Extracts identification functions per framing and fits sigmoids.
    """
    id_results = meta.get("identification_results", [])
    if not id_results:
        return {"error": "No identification results found"}
    
    results = {}
    
    for framing in ["small_large", "single_multi"]:
        framing_data = [r for r in id_results if r["framing"] == framing]
        if not framing_data:
            continue
        
        values = np.array([r["value"] for r in framing_data])
        prob_b = np.array([r["prob_category_b"] for r in framing_data])
        
        # Sort by value
        sort_idx = np.argsort(values)
        values = values[sort_idx]
        prob_b = prob_b[sort_idx]
        
        # Fit sigmoid
        fit = fit_identification_sigmoid(values, prob_b)
        
        results[framing] = {
            "values": values.tolist(),
            "prob_category_b": prob_b.tolist(),
            "sigmoid_fit": fit,
        }
    
    return results


# =============================================================================
# 8. Go/No-Go Decision Logic
# =============================================================================

def go_nogo_analysis(
    rsa_results_decade: Dict,
    rsa_results_control: Dict,
    precision_decade: Dict,
    precision_control: Dict,
    values_decade: np.ndarray,
    values_control: np.ndarray,
    boundary_decade: int = 10,
    boundary_control: int = 15,
    primary_layers: Tuple[int, int] = (8, 25),
) -> Dict:
    """Evaluate the go/no-go question.
    
    Question: Does the RDM show any warping at the decade boundary (10)
    that is ABSENT at the control position (15)?
    
    Criteria:
      1. CP-Additive > Continuous in RSA at decade_10 (at primary layers)
      2. CP-Additive ≤ Continuous at control_15 (or weaker advantage)
      3. Precision gradient shows dip at boundary for decade_10 but not control_15
    """
    decision = {"criteria": {}, "go": False, "summary": ""}
    
    # Criterion 1: CP-Additive beats Continuous at decade boundary
    l_start, l_end = primary_layers
    decade_cp_rhos = rsa_results_decade["cp_additive"]["rho"][l_start:l_end]
    decade_cont_rhos = rsa_results_decade["continuous"]["rho"][l_start:l_end]
    
    cp_advantage_decade = np.array(decade_cp_rhos) - np.array(decade_cont_rhos)
    mean_advantage_decade = float(np.mean(cp_advantage_decade))
    max_advantage_decade = float(np.max(cp_advantage_decade))
    n_layers_cp_wins = int(np.sum(cp_advantage_decade > 0))
    n_primary_layers = l_end - l_start
    
    decision["criteria"]["cp_advantage_decade"] = {
        "mean": mean_advantage_decade,
        "max": max_advantage_decade,
        "n_layers_cp_wins": n_layers_cp_wins,
        "n_primary_layers": n_primary_layers,
        "fraction_cp_wins": n_layers_cp_wins / n_primary_layers,
    }
    
    # Criterion 2: CP-Additive does NOT beat Continuous at control
    control_cp_rhos = rsa_results_control["cp_additive"]["rho"][l_start:l_end]
    control_cont_rhos = rsa_results_control["continuous"]["rho"][l_start:l_end]
    
    cp_advantage_control = np.array(control_cp_rhos) - np.array(control_cont_rhos)
    mean_advantage_control = float(np.mean(cp_advantage_control))
    
    decision["criteria"]["cp_advantage_control"] = {
        "mean": mean_advantage_control,
        "max": float(np.max(cp_advantage_control)),
        "n_layers_cp_wins": int(np.sum(cp_advantage_control > 0)),
        "n_primary_layers": n_primary_layers,
    }
    
    # Criterion 3: Precision gradient — is there a dip at the boundary?
    # For decade_10: look at distance between value 9 and value 10
    # Dip in precision = spike in distance at boundary
    midpoints_decade = precision_decade["midpoints"]
    
    # Find the midpoint closest to the boundary
    boundary_idx_decade = np.argmin(np.abs(midpoints_decade - boundary_decade))
    
    # Average distance at boundary vs average distance elsewhere (primary layers)
    distances_decade = precision_decade["distances"]
    boundary_dist = np.mean(distances_decade[l_start:l_end, boundary_idx_decade])
    nonboundary_dists = np.delete(
        distances_decade[l_start:l_end], boundary_idx_decade, axis=1
    )
    avg_nonboundary = np.mean(nonboundary_dists)
    
    distance_ratio_decade = boundary_dist / (avg_nonboundary + 1e-10)
    
    # Same for control_15
    midpoints_control = precision_control["midpoints"]
    boundary_idx_control = np.argmin(np.abs(midpoints_control - boundary_control))
    
    distances_control = precision_control["distances"]
    boundary_dist_ctrl = np.mean(distances_control[l_start:l_end, boundary_idx_control])
    nonboundary_dists_ctrl = np.delete(
        distances_control[l_start:l_end], boundary_idx_control, axis=1
    )
    avg_nonboundary_ctrl = np.mean(nonboundary_dists_ctrl)
    
    distance_ratio_control = boundary_dist_ctrl / (avg_nonboundary_ctrl + 1e-10)
    
    decision["criteria"]["precision_gradient"] = {
        "decade_10_boundary_distance_ratio": float(distance_ratio_decade),
        "control_15_boundary_distance_ratio": float(distance_ratio_control),
        "decade_exceeds_control": distance_ratio_decade > distance_ratio_control,
    }
    
    # Go/No-Go decision
    # GO if:
    #   (a) CP-Additive beats Continuous at majority of primary layers for decade_10
    #   (b) This advantage is STRONGER than at control_15
    #   (c) OR precision gradient shows boundary-specific dip
    
    cp_wins_majority = n_layers_cp_wins > n_primary_layers / 2
    cp_stronger_at_decade = mean_advantage_decade > mean_advantage_control
    precision_boundary_specific = distance_ratio_decade > distance_ratio_control * 1.2
    
    go = (cp_wins_majority and cp_stronger_at_decade) or precision_boundary_specific
    
    decision["go"] = go
    decision["summary"] = (
        f"GO" if go else "NO-GO"
    ) + (
        f"\n  CP-Additive > Continuous at {n_layers_cp_wins}/{n_primary_layers} "
        f"primary layers (decade_10)\n"
        f"  Mean CP advantage (decade): {mean_advantage_decade:.4f}\n"
        f"  Mean CP advantage (control): {mean_advantage_control:.4f}\n"
        f"  Precision boundary ratio (decade): {distance_ratio_decade:.3f}\n"
        f"  Precision boundary ratio (control): {distance_ratio_control:.3f}"
    )
    
    return decision


# =============================================================================
# 9. Visualisation
# =============================================================================

def plot_rdm_heatmap(
    rdm: np.ndarray,
    values: np.ndarray,
    title: str,
    filepath: Path,
    cmap: str = "viridis",
):
    """Plot RDM as heatmap with value labels."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    im = ax.imshow(rdm, cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(values, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(values, fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Probing Value")
    ax.set_ylabel("Probing Value")
    plt.colorbar(im, ax=ax, label="Dissimilarity")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_rsa_comparison(
    rsa_results: Dict,
    condition: str,
    filepath: Path,
    primary_layers: Tuple[int, int] = (8, 25),
):
    """Plot RSA rho for Continuous vs CP-Additive across layers.
    
    The key go/no-go comparison: does CP-Additive ever beat Continuous?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: All theoretical models
    ax = axes[0]
    colors = {
        "continuous": "#2196F3",
        "cp_additive": "#F44336",
        "categorical": "#4CAF50",
        "linear": "#9E9E9E",
    }
    labels = {
        "continuous": "Continuous (Weber/Log)",
        "cp_additive": "CP-Additive",
        "categorical": "Categorical",
        "linear": "Linear",
    }
    
    for name in ["continuous", "cp_additive", "categorical", "linear"]:
        if name in rsa_results:
            rhos = rsa_results[name]["rho"]
            ax.plot(range(len(rhos)), rhos, color=colors[name],
                    label=labels[name], linewidth=1.5)
    
    l_start, l_end = primary_layers
    ax.axvspan(l_start, l_end, alpha=0.1, color="blue", label="Primary layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"RSA: All Theoretical Models ({condition})")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Right panel: CP advantage (CP-Additive - Continuous)
    ax = axes[1]
    if "cp_additive" in rsa_results and "continuous" in rsa_results:
        cp_rhos = np.array(rsa_results["cp_additive"]["rho"])
        cont_rhos = np.array(rsa_results["continuous"]["rho"])
        advantage = cp_rhos - cont_rhos
        
        ax.bar(range(len(advantage)), advantage,
               color=["#F44336" if a > 0 else "#2196F3" for a in advantage],
               alpha=0.7, width=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvspan(l_start, l_end, alpha=0.1, color="blue")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δρ (CP-Additive − Continuous)")
        ax.set_title(f"CP Advantage ({condition})")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_precision_gradient(
    precision_data: Dict,
    values: np.ndarray,
    boundary: int,
    condition: str,
    filepath: Path,
    primary_layers: Tuple[int, int] = (8, 25),
):
    """Plot precision gradient with boundary marker.
    
    Shows local distance between adjacent values.
    CP prediction: distance should SPIKE at boundary.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    midpoints = precision_data["midpoints"]
    l_start, l_end = primary_layers
    
    # Left: Average distance across primary layers
    ax = axes[0]
    avg_dist = np.mean(precision_data["distances"][l_start:l_end], axis=0)
    se_dist = np.std(precision_data["distances"][l_start:l_end], axis=0) / np.sqrt(l_end - l_start)
    
    ax.plot(midpoints, avg_dist, "o-", color="#2196F3", linewidth=1.5, markersize=4)
    ax.fill_between(midpoints, avg_dist - se_dist, avg_dist + se_dist,
                     alpha=0.2, color="#2196F3")
    ax.axvline(boundary, color="red", linestyle="--", alpha=0.7,
               label=f"Boundary ({boundary})")
    ax.set_xlabel("Magnitude (midpoint between adjacent values)")
    ax.set_ylabel("Local Distance (cosine)")
    ax.set_title(f"Precision Gradient — {condition}\n"
                 f"(avg over primary layers {l_start}–{l_end-1})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Right: Layerwise heatmap of distances
    ax = axes[1]
    im = ax.imshow(
        precision_data["distances"],
        aspect="auto",
        cmap="hot_r",
        interpolation="nearest",
    )
    ax.set_xlabel("Adjacent pair index")
    ax.set_ylabel("Layer")
    ax.set_title(f"Distance Heatmap — {condition}")
    
    # Mark boundary position
    boundary_idx = np.argmin(np.abs(midpoints - boundary))
    ax.axvline(boundary_idx, color="cyan", linestyle="--", alpha=0.8, linewidth=1.5)
    
    plt.colorbar(im, ax=ax, label="Cosine distance")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_identification_function(
    id_analysis: Dict,
    condition: str,
    boundary: int,
    filepath: Path,
):
    """Plot identification function with sigmoid fit."""
    n_framings = len([k for k in id_analysis if k != "error"])
    if n_framings == 0:
        return
    
    fig, axes = plt.subplots(1, n_framings, figsize=(6 * n_framings, 5))
    if n_framings == 1:
        axes = [axes]
    
    framing_labels = {
        "small_large": "Small vs Large",
        "single_multi": "Single-digit vs Multi-digit",
        "digit_count": "One Digit vs Two Digits",
    }
    
    available_framings = [k for k in ["small_large", "single_multi", "digit_count"]
                          if k in id_analysis]
    for ax, framing in zip(axes, available_framings):
        if framing not in id_analysis:
            continue
        
        data = id_analysis[framing]
        values = np.array(data["values"])
        prob_b = np.array(data["prob_category_b"])
        fit = data["sigmoid_fit"]
        
        # Plot data points
        ax.plot(values, prob_b, "ko", markersize=6, label="Data")
        
        # Plot sigmoid fit
        if fit.get("fitted", False):
            x_smooth = np.linspace(min(values), max(values), 200)
            y_smooth = sigmoid(x_smooth, fit["a_slope"], fit["b_crossover"],
                              fit["c_upper"], fit["d_lower"])
            ax.plot(x_smooth, y_smooth, "r-", linewidth=1.5,
                    label=f"Sigmoid (R²={fit['r_squared']:.3f})")
            
            # Mark crossover
            ax.axvline(fit["crossover"], color="green", linestyle="--", alpha=0.7,
                       label=f"Crossover = {fit['crossover']:.1f}")
            ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        
        ax.axvline(boundary, color="blue", linestyle=":", alpha=0.5,
                   label=f"Expected boundary ({boundary})")
        ax.set_xlabel("Probing Value")
        ax.set_ylabel("P(Category B)")
        ax.set_title(f"Identification: {framing_labels.get(framing, framing)}")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Paradigm B0 — {condition}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_go_nogo_summary(
    rsa_decade: Dict,
    rsa_control: Dict,
    precision_decade: Dict,
    precision_control: Dict,
    values_decade: np.ndarray,
    values_control: np.ndarray,
    decision: Dict,
    filepath: Path,
    primary_layers: Tuple[int, int] = (8, 25),
):
    """Summary figure for go/no-go decision.
    
    2×2 grid: RSA advantage (decade vs control) × Precision gradient (decade vs control)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    l_start, l_end = primary_layers
    
    # Top-left: CP advantage — decade_10
    ax = axes[0, 0]
    cp = np.array(rsa_decade["cp_additive"]["rho"])
    cont = np.array(rsa_decade["continuous"]["rho"])
    adv = cp - cont
    ax.bar(range(len(adv)), adv,
           color=["#F44336" if a > 0 else "#2196F3" for a in adv],
           alpha=0.7, width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvspan(l_start, l_end, alpha=0.1, color="blue")
    ax.set_title("CP Advantage — decade_10 (boundary at 10)")
    ax.set_ylabel("Δρ (CP-Additive − Continuous)")
    ax.grid(True, alpha=0.3)
    
    # Top-right: CP advantage — control_15
    ax = axes[0, 1]
    cp = np.array(rsa_control["cp_additive"]["rho"])
    cont = np.array(rsa_control["continuous"]["rho"])
    adv = cp - cont
    ax.bar(range(len(adv)), adv,
           color=["#F44336" if a > 0 else "#2196F3" for a in adv],
           alpha=0.7, width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvspan(l_start, l_end, alpha=0.1, color="blue")
    ax.set_title("CP Advantage — control_15 (no boundary)")
    ax.set_ylabel("Δρ")
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Precision gradient — decade_10
    ax = axes[1, 0]
    midpoints = precision_decade["midpoints"]
    avg_dist = np.mean(precision_decade["distances"][l_start:l_end], axis=0)
    ax.plot(midpoints, avg_dist, "o-", color="#2196F3", linewidth=1.5, markersize=4)
    ax.axvline(10, color="red", linestyle="--", alpha=0.7, label="Boundary (10)")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Local Distance")
    ax.set_title("Precision Gradient — decade_10")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Precision gradient — control_15
    ax = axes[1, 1]
    midpoints = precision_control["midpoints"]
    avg_dist = np.mean(precision_control["distances"][l_start:l_end], axis=0)
    ax.plot(midpoints, avg_dist, "o-", color="#2196F3", linewidth=1.5, markersize=4)
    ax.axvline(15, color="red", linestyle="--", alpha=0.7, label="Pseudo-boundary (15)")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Local Distance")
    ax.set_title("Precision Gradient — control_15")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add decision text
    verdict = "GO ✓" if decision["go"] else "NO-GO ✗"
    fig.suptitle(
        f"M3 Pilot Go/No-Go: {verdict}",
        fontsize=16, fontweight="bold",
        color="green" if decision["go"] else "red",
        y=1.02,
    )
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


# =============================================================================
# 10. Main Pipeline
# =============================================================================

def main():
    config = AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("M3 Pilot Analysis — Go/No-Go Decision")
    print(f"Model: {config.model_short}")
    print(f"Conditions: {config.conditions}")
    print(f"Permutations: {config.n_permutations}")
    print(f"Seed: {config.seed}")
    print("=" * 70)
    
    all_results = {}
    
    for condition in config.conditions:
        print(f"\n{'='*60}")
        print(f"Analysing condition: {condition}")
        print(f"{'='*60}")
        
        # Load data
        try:
            centroid_data = load_centroids(config, condition)
            stim_data = load_stimuli(config, condition)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print(f"  Run m3_extract.py first.")
            continue
        
        rsa_centroids = centroid_data["rsa_centroids"]
        values = centroid_data["values"]
        boundary = stim_data["metadata"]["boundary"]
        meta = centroid_data["meta"]
        
        print(f"  Values: {values}")
        print(f"  Boundary: {boundary}")
        print(f"  Centroid shape: {rsa_centroids.shape}")
        
        # Step 1: Compute empirical RDMs (cosine — primary metric)
        print("\n  Step 1: Computing empirical RDMs (cosine)")
        empirical_rdms = compute_rdms_all_layers(rsa_centroids, metric="cosine")
        print(f"  RDM shape: {empirical_rdms.shape}")
        
        # Plot RDM at a representative primary layer (layer 16)
        rep_layer = 16
        if rep_layer < empirical_rdms.shape[0]:
            plot_rdm_heatmap(
                empirical_rdms[rep_layer], values,
                f"Empirical RDM — {condition} (Layer {rep_layer}, Cosine)",
                config.output_dir / f"rdm_heatmap_{condition}_layer{rep_layer}.png",
            )
            print(f"  RDM heatmap saved (layer {rep_layer})")
        
        # Step 2: Build theoretical RDMs
        print("\n  Step 2: Building theoretical RDMs")
        theoretical_rdms = build_theoretical_rdms(values, boundary)
        for name, rdm in theoretical_rdms.items():
            print(f"    {name}: shape={rdm.shape}, "
                  f"range=[{rdm.min():.3f}, {rdm.max():.3f}]")
        
        # Step 3: RSA with Mantel tests
        print(f"\n  Step 3: RSA with Mantel tests ({config.n_permutations} permutations)")
        rsa_results = rsa_all_layers(
            empirical_rdms, theoretical_rdms,
            n_permutations=config.n_permutations,
            seed=config.seed,
        )
        
        # Print summary at primary layers
        l_start, l_end = config.primary_layer_range
        print(f"\n  RSA Summary (primary layers {l_start}-{l_end-1}):")
        for name in ["continuous", "cp_additive", "categorical", "linear"]:
            if name in rsa_results:
                primary_rhos = rsa_results[name]["rho"][l_start:l_end]
                primary_ps = rsa_results[name]["p"][l_start:l_end]
                print(f"    {name:15s}: mean ρ = {np.mean(primary_rhos):.4f}, "
                      f"max ρ = {np.max(primary_rhos):.4f}, "
                      f"min p = {np.min(primary_ps):.4f}")
        
        # Plot RSA comparison
        plot_rsa_comparison(
            rsa_results, condition,
            config.output_dir / f"rsa_comparison_{condition}.png",
            config.primary_layer_range,
        )
        print("  RSA comparison plot saved")
        
        # Step 4: Precision gradient
        print("\n  Step 4: Precision gradient")
        precision_data = compute_precision_gradient(rsa_centroids, values, metric="cosine")
        
        plot_precision_gradient(
            precision_data, values, boundary, condition,
            config.output_dir / f"precision_gradient_{condition}.png",
            config.primary_layer_range,
        )
        print("  Precision gradient plot saved")
        
        # Step 5: Identification analysis (Paradigm B0)
        print("\n  Step 5: Identification analysis (Paradigm B0)")
        id_analysis = analyse_identification(meta)
        
        if "error" not in id_analysis:
            for framing, data in id_analysis.items():
                fit = data.get("sigmoid_fit", {})
                if fit.get("fitted"):
                    print(f"    {framing}: crossover={fit['crossover']:.2f}, "
                          f"slope={fit['slope_at_crossover']:.3f}, "
                          f"R²={fit['r_squared']:.3f}")
                else:
                    print(f"    {framing}: sigmoid fit failed")
            
            plot_identification_function(
                id_analysis, condition, boundary,
                config.output_dir / f"identification_{condition}.png",
            )
            print("  Identification plot saved")
        else:
            print(f"  {id_analysis['error']}")
        
        # Store results
        all_results[condition] = {
            "rsa_results": rsa_results,
            "precision_data": {
                "distances": precision_data["distances"].tolist(),
                "midpoints": precision_data["midpoints"].tolist(),
            },
            "identification": id_analysis,
            "values": values.tolist(),
            "boundary": boundary,
        }
    
    # Step 6: Go/No-Go decision
    if "decade_10" in all_results and "control_15" in all_results:
        print(f"\n{'='*60}")
        print("GO/NO-GO DECISION")
        print(f"{'='*60}")
        
        decision = go_nogo_analysis(
            rsa_results_decade=all_results["decade_10"]["rsa_results"],
            rsa_results_control=all_results["control_15"]["rsa_results"],
            precision_decade={
                "distances": np.array(all_results["decade_10"]["precision_data"]["distances"]),
                "midpoints": np.array(all_results["decade_10"]["precision_data"]["midpoints"]),
            },
            precision_control={
                "distances": np.array(all_results["control_15"]["precision_data"]["distances"]),
                "midpoints": np.array(all_results["control_15"]["precision_data"]["midpoints"]),
            },
            values_decade=np.array(all_results["decade_10"]["values"]),
            values_control=np.array(all_results["control_15"]["values"]),
            primary_layers=config.primary_layer_range,
        )
        
        print(f"\n{decision['summary']}")
        
        # Save go/no-go summary figure
        plot_go_nogo_summary(
            all_results["decade_10"]["rsa_results"],
            all_results["control_15"]["rsa_results"],
            {
                "distances": np.array(all_results["decade_10"]["precision_data"]["distances"]),
                "midpoints": np.array(all_results["decade_10"]["precision_data"]["midpoints"]),
            },
            {
                "distances": np.array(all_results["control_15"]["precision_data"]["distances"]),
                "midpoints": np.array(all_results["control_15"]["precision_data"]["midpoints"]),
            },
            np.array(all_results["decade_10"]["values"]),
            np.array(all_results["control_15"]["values"]),
            decision,
            config.output_dir / "go_nogo_summary.png",
            config.primary_layer_range,
        )
        print("  Go/no-go summary figure saved")
        
        all_results["go_nogo_decision"] = decision
    
    # Save all results as JSON
    results_path = config.output_dir / f"m3_pilot_results_{config.model_short}.json"
    
    # Convert non-serialisable types
    def make_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serialisable(v) for v in obj]
        return obj
    
    with open(results_path, "w") as f:
        json.dump(make_serialisable(all_results), f, indent=2)
    print(f"\n  Full results saved: {results_path}")
    
    print("\n" + "=" * 70)
    print("Pilot analysis complete.")
    print(f"Results: {config.output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
