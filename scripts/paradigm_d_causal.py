"""
Weber's Law Project 4.2 — Paradigm D: Causal Intervention (Activation Patching)
Classical Minds, Modern Machines

Step 1: Identify magnitude direction (linear probe + PCA fallback).
Step 2: Activation patching along magnitude direction.
Step 3: Specificity control (random directions).

Pre-registration ref: v2.7 Section 5.6, Appendix B (go/no-go), Section 4.2 (H7).

Usage:
    python paradigm_d_causal.py --model llama_instruct
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS, RESULTS_DIR, STIMULI_DIR,
    N_LAYERS_TOTAL, PRIMARY_LAYER_RANGE,
    SEED_PARADIGM_D, SEED_RANDOM_DIRECTIONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Go/No-Go Gate (v2.7 Appendix B) ──

def check_go_nogo(model_key: str, results_dir: Path) -> dict:
    """
    Paradigm D go/no-go (v2.7 Appendix B):
    Primary: linear probe R² > 0.50 at any layer.
    Secondary (v2.5): RSA Mantel significant at any layer 16-32 → proceed with PCA.
    """
    # Load Paradigm A results (numerical domain)
    analysis_path = results_dir / "paradigm_a" / model_key / "numerical" / "paradigm_a_analysis.json"
    if not analysis_path.exists():
        return {"go": False, "reason": "paradigm_a_not_run"}

    with open(analysis_path) as f:
        analysis = json.load(f)

    # Check probe R² (we compute this from the centroids)
    hs_path = results_dir / "paradigm_a" / model_key / "numerical" / "hidden_states.npz"
    data = np.load(hs_path)
    centroids = data["centroids"]
    magnitudes = np.array(analysis["magnitudes"])
    log_mags = np.log(magnitudes)

    probe_results = {}
    best_r2 = 0
    best_layer = None

    for layer in range(N_LAYERS_TOTAL):
        vecs = centroids[:, layer, :]
        if np.any(np.isnan(vecs)):
            continue

        # Ridge regression: log(magnitude) ~ hidden_state
        # v2.7: "Train a linear probe (ridge regression)"
        from sklearn.linear_model import Ridge
        model_ridge = Ridge(alpha=1.0)
        model_ridge.fit(vecs, log_mags)
        y_pred = model_ridge.predict(vecs)
        ss_res = np.sum((log_mags - y_pred) ** 2)
        ss_tot = np.sum((log_mags - log_mags.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        probe_results[layer] = {
            "r2": float(r2),
            "weight_norm": float(np.linalg.norm(model_ridge.coef_)),
        }

        if r2 > best_r2:
            best_r2 = r2
            best_layer = layer

    # Primary gate: R² > 0.50
    primary_go = best_r2 > 0.50

    # Secondary gate: RSA significant at any primary layer
    rsa_significant = False
    rsa_layer = None
    for layer in range(PRIMARY_LAYER_RANGE[0], PRIMARY_LAYER_RANGE[1]):
        layer_key = f"layer_{layer:02d}"
        rsa = (analysis.get("layers", {})
               .get(layer_key, {})
               .get("cosine", {})
               .get("rsa", {}))
        for theo in ["weber", "linear", "stevens"]:
            if rsa.get(theo, {}).get("p_value", 1) < 0.05:
                rsa_significant = True
                rsa_layer = layer
                break
        if rsa_significant:
            break

    secondary_go = rsa_significant and not primary_go

    # Determine direction method
    if primary_go:
        direction_method = "probe"
        go_layer = best_layer
    elif secondary_go:
        direction_method = "pca"
        go_layer = rsa_layer
    else:
        direction_method = None
        go_layer = None

    return {
        "go": primary_go or secondary_go,
        "primary_go": primary_go,
        "secondary_go": secondary_go,
        "direction_method": direction_method,
        "best_probe_r2": float(best_r2),
        "best_probe_layer": best_layer,
        "rsa_significant": rsa_significant,
        "rsa_layer": rsa_layer,
        "go_layer": go_layer,
        "probe_results": {str(k): v for k, v in probe_results.items()},
    }


# ── Step 1: Identify magnitude direction ──

def get_magnitude_direction(
    centroids: np.ndarray,
    magnitudes: np.ndarray,
    layer: int,
    method: str = "probe",
) -> dict:
    """
    v2.7 Section 5.6 Step 1:
    Probe method: "The probe's weight vector vmag, normalised to unit length,
    defines the magnitude direction."
    PCA method: "Take PC1. If PC1 correlates with log(magnitude) at r > 0.80,
    it defines a second magnitude direction."
    """
    vecs = centroids[:, layer, :]
    log_mags = np.log(magnitudes)
    d_model = vecs.shape[1]

    result = {}

    if method in ["probe", "both"]:
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(vecs, log_mags)
        v_probe = ridge.coef_.copy()
        v_probe /= np.linalg.norm(v_probe)
        result["probe_direction"] = v_probe
        result["probe_r2"] = float(1 - np.sum((log_mags - ridge.predict(vecs))**2) /
                                   np.sum((log_mags - log_mags.mean())**2))

    if method in ["pca", "both"]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        scores = pca.fit_transform(vecs).ravel()
        v_pca = pca.components_[0]
        v_pca /= np.linalg.norm(v_pca)

        # Check correlation with log(magnitude)
        rho, _ = spearmanr(scores, log_mags)
        if rho < 0:
            v_pca = -v_pca  # Flip to positive correlation
            rho = -rho

        result["pca_direction"] = v_pca
        result["pca_logmag_corr"] = float(rho)
        result["pca_valid"] = rho > 0.80  # v2.7 threshold

    return result


# ── Step 2: Activation patching ──

def run_activation_patching(
    model,
    tokenizer,
    prompts: list[dict],
    magnitude_direction: np.ndarray,
    anchor_projection: float,
    layer: int,
    doses: list[float] = [0.25, 0.50, 0.75, 1.00],
) -> list[dict]:
    """
    v2.7 Section 5.6 Step 2:
    "For each dose d: replace h with h + d × (proj_anchor - proj_h) × v_mag.
    This is additive patching along the magnitude direction only,
    preserving all orthogonal components."
    """
    import torch

    results = []
    v_mag = torch.tensor(magnitude_direction, dtype=torch.float16).to(next(model.parameters()).device)

    for idx, prompt_info in enumerate(prompts):
        prompt = prompt_info["prompt"]

        # Find magnitude token position (first magnitude in the prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        # Baseline (unpatched) forward pass
        with torch.no_grad():
            base_outputs = model(**inputs)
        base_logits = base_outputs.logits[0, -1, :].float().cpu()

        # Get A/B token probabilities
        a_ids = tokenizer.encode("A", add_special_tokens=False)
        b_ids = tokenizer.encode("B", add_special_tokens=False)
        base_probs = torch.softmax(base_logits, dim=0)
        base_p_correct = base_probs[a_ids[0]].item() if prompt_info["correct_answer"] == "A" else base_probs[b_ids[0]].item()

        # Patched forward passes at each dose
        dose_results = []
        for dose in doses:
            # Register forward hook at the target layer
            hook_handle = None
            patched = False

            def make_hook(d, v, anchor_proj, mag_token_pos):
                def hook_fn(module, input, output):
                    # output is a tuple; hidden states are output[0]
                    h = output[0]
                    # Project onto magnitude direction
                    proj_h = torch.dot(h[0, mag_token_pos, :], v)
                    # Patch: h + d * (anchor - proj_h) * v
                    shift = d * (anchor_proj - proj_h.item())
                    h[0, mag_token_pos, :] += shift * v
                    return (h,) + output[1:]
                return hook_fn

            # Magnitude token is near the start of the prompt
            # Use position 1 (after BOS) as approximation
            # TODO: More precise position finding from prompt structure
            mag_token_pos = min(5, seq_len - 2)  # Approximate

            # Get the appropriate layer module
            if hasattr(model, "model"):  # Llama/Mistral structure
                layer_module = model.model.layers[layer]
            else:
                layer_module = model.transformer.h[layer]

            hook_handle = layer_module.register_forward_hook(
                make_hook(dose, v_mag, anchor_projection, mag_token_pos)
            )

            with torch.no_grad():
                patched_outputs = model(**inputs)

            hook_handle.remove()

            patched_logits = patched_outputs.logits[0, -1, :].float().cpu()
            patched_probs = torch.softmax(patched_logits, dim=0)
            patched_p_correct = (patched_probs[a_ids[0]].item()
                                 if prompt_info["correct_answer"] == "A"
                                 else patched_probs[b_ids[0]].item())

            delta_p = patched_p_correct - base_p_correct

            dose_results.append({
                "dose": float(dose),
                "base_p_correct": float(base_p_correct),
                "patched_p_correct": float(patched_p_correct),
                "delta_p": float(delta_p),
            })

        results.append({
            "prompt_idx": idx,
            "baseline": prompt_info.get("baseline"),
            "ratio": prompt_info.get("ratio"),
            "base_p_correct": float(base_p_correct),
            "doses": dose_results,
        })

        if (idx + 1) % 50 == 0:
            log.info(f"  Patched {idx+1}/{len(prompts)} prompts")

    return results


# ── Step 3: Specificity control ──

def run_random_direction_control(
    model,
    tokenizer,
    prompts: list[dict],
    d_model: int,
    layer: int,
    anchor_projection: float,
    n_random: int = 10,
    seed: int = SEED_RANDOM_DIRECTIONS,
) -> list[dict]:
    """
    v2.7 Section 5.6 Step 3:
    "10 random unit vectors in d_model-dimensional space (seed 42).
    The 97.5th percentile of the 10 × 200 = 2,000 random-direction shifts
    serves as the null threshold."
    """
    rng = np.random.default_rng(seed)
    all_random_results = []

    for r_idx in range(n_random):
        # Random unit vector
        v_random = rng.standard_normal(d_model)
        v_random /= np.linalg.norm(v_random)

        log.info(f"  Running random direction {r_idx+1}/{n_random}...")
        results = run_activation_patching(
            model, tokenizer, prompts,
            magnitude_direction=v_random,
            anchor_projection=anchor_projection,
            layer=layer,
            doses=[1.00],  # Full dose only for random control
        )

        for r in results:
            for d in r["doses"]:
                all_random_results.append({
                    "random_idx": r_idx,
                    "prompt_idx": r["prompt_idx"],
                    "delta_p": d["delta_p"],
                })

    return all_random_results


# ── H7 evaluation ──

def evaluate_h7(
    magnitude_results: list[dict],
    random_results: list[dict],
) -> dict:
    """
    v2.7 Section 4.2 (H7):
    "Mean absolute shift under magnitude-direction patching exceeds the
    97.5th percentile of the random-direction distribution, for at least
    75% of the 200 comparison prompts."
    """
    # Extract |Δp| at full dose for magnitude direction
    mag_delta_ps = []
    for r in magnitude_results:
        full_dose = [d for d in r["doses"] if d["dose"] == 1.00]
        if full_dose:
            mag_delta_ps.append(abs(full_dose[0]["delta_p"]))

    # Random direction |Δp| distribution
    random_delta_ps = np.array([abs(r["delta_p"]) for r in random_results])
    threshold_975 = np.percentile(random_delta_ps, 97.5)

    # Count prompts where magnitude effect > threshold
    n_exceeds = sum(1 for dp in mag_delta_ps if dp > threshold_975)
    fraction_exceeds = n_exceeds / len(mag_delta_ps) if mag_delta_ps else 0

    return {
        "n_prompts": len(mag_delta_ps),
        "n_exceeds_threshold": n_exceeds,
        "fraction_exceeds": float(fraction_exceeds),
        "threshold_975": float(threshold_975),
        "mean_mag_delta_p": float(np.mean(mag_delta_ps)) if mag_delta_ps else None,
        "mean_random_delta_p": float(np.mean(random_delta_ps)),
        "h7_passes": fraction_exceeds >= 0.75,
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Paradigm D: Causal Intervention")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--stimuli-dir", type=Path, default=STIMULI_DIR)
    parser.add_argument("--skip-gate", action="store_true",
                        help="Skip go/no-go check (for debugging)")
    args = parser.parse_args()

    log.info(f"\n{'='*60}")
    log.info(f"Paradigm D: Causal Intervention — {args.model}")
    log.info(f"{'='*60}")

    # Step 0: Go/No-Go
    if not args.skip_gate:
        log.info("Checking go/no-go gate...")
        gate = check_go_nogo(args.model, args.results_dir)
        log.info(f"  Probe best R²: {gate['best_probe_r2']:.4f} at layer {gate['best_probe_layer']}")
        log.info(f"  Primary go: {gate['primary_go']} | Secondary go: {gate['secondary_go']}")

        if not gate["go"]:
            log.warning("GO/NO-GO: NO-GO. Paradigm D deferred.")
            # Save the gate result
            out_dir = args.results_dir / "paradigm_d" / args.model
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "paradigm_d_gate.json", "w") as f:
                json.dump(gate, f, indent=2)
            return

        log.info(f"  GO via {gate['direction_method']} at layer {gate['go_layer']}")
    else:
        # Determine layer from probe results
        gate = check_go_nogo(args.model, args.results_dir)

    # Load centroids for direction computation
    hs_path = args.results_dir / "paradigm_a" / args.model / "numerical" / "hidden_states.npz"
    data = np.load(hs_path)
    centroids = data["centroids"]

    with open(args.results_dir / "paradigm_a" / args.model / "numerical" / "extraction_metadata.json") as f:
        meta = json.load(f)
    magnitudes = np.array(meta["magnitudes_numeric"])

    target_layer = gate["go_layer"]
    d_model = MODELS[args.model]["d_model"]

    # Step 1: Get magnitude direction
    log.info(f"\nComputing magnitude direction at layer {target_layer}...")
    direction_info = get_magnitude_direction(
        centroids, magnitudes, target_layer,
        method="both",
    )

    if gate["direction_method"] == "probe":
        v_mag = direction_info["probe_direction"]
        log.info(f"  Using probe direction (R²={direction_info['probe_r2']:.4f})")
    else:
        v_mag = direction_info["pca_direction"]
        log.info(f"  Using PCA direction (log-mag corr={direction_info['pca_logmag_corr']:.4f})")

    # Compute anchor projection (v2.7: "centroid of magnitude 1000")
    # Find index of magnitude 1000
    idx_1000 = np.where(magnitudes == 1000)[0]
    if len(idx_1000) == 0:
        idx_1000 = len(magnitudes) - 1  # largest magnitude
    else:
        idx_1000 = idx_1000[0]

    anchor_vec = centroids[idx_1000, target_layer, :]
    anchor_proj = float(np.dot(anchor_vec, v_mag))
    log.info(f"  Anchor projection (mag=1000): {anchor_proj:.4f}")

    # Load comparison prompts for patching
    # v2.7: "200 comparison prompts (stratified: 40 per baseline)"
    stim_path = args.stimuli_dir / "paradigm_d_prompts.json"
    if stim_path.exists():
        with open(stim_path) as f:
            prompts = json.load(f)
    else:
        log.warning("Paradigm D prompts not found. Using first 200 from Paradigm B stimuli.")
        b_path = args.stimuli_dir / "comparison_pairs_numerical.json"
        if b_path.exists():
            with open(b_path) as f:
                all_stim = json.load(f)
            prompts = all_stim[:200]
        else:
            log.error("No prompts available for Paradigm D.")
            return

    log.info(f"  Loaded {len(prompts)} comparison prompts")

    # Load model
    from paradigm_b_behaviour import load_model as load_model_b
    model, tokenizer = load_model_b(args.model)

    # Step 2: Activation patching
    log.info("\nRunning activation patching (magnitude direction)...")
    mag_results = run_activation_patching(
        model, tokenizer, prompts,
        magnitude_direction=v_mag,
        anchor_projection=anchor_proj,
        layer=target_layer,
    )

    # Step 3: Random direction control
    log.info("\nRunning random direction control...")
    random_results = run_random_direction_control(
        model, tokenizer, prompts,
        d_model=d_model,
        layer=target_layer,
        anchor_projection=anchor_proj,
    )

    # H7 evaluation
    h7 = evaluate_h7(mag_results, random_results)
    log.info(
        f"\nH7: {h7['n_exceeds_threshold']}/{h7['n_prompts']} prompts exceed "
        f"97.5th percentile threshold ({h7['threshold_975']:.4f}). "
        f"Fraction: {h7['fraction_exceeds']:.3f}. "
        f"{'PASS' if h7['h7_passes'] else 'FAIL'}"
    )

    # Check probe vs PCA consistency (v2.7)
    if "probe_direction" in direction_info and "pca_direction" in direction_info:
        cos_sim = float(np.dot(direction_info["probe_direction"],
                               direction_info["pca_direction"]))
        log.info(f"  Probe-PCA direction cosine similarity: {cos_sim:.4f}")

    # Save results
    out_dir = args.results_dir / "paradigm_d" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "model_key": args.model,
        "gate": gate,
        "target_layer": target_layer,
        "direction_method": gate["direction_method"],
        "direction_info": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in direction_info.items()
        },
        "anchor_projection": anchor_proj,
        "n_prompts": len(prompts),
        "magnitude_patching": mag_results,
        "h7_evaluation": h7,
    }

    with open(out_dir / "paradigm_d_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save random results separately (large)
    with open(out_dir / "paradigm_d_random_control.json", "w") as f:
        json.dump(random_results, f, indent=2, default=str)

    log.info(f"\nResults saved to {out_dir}")
    log.info("\n=== Paradigm D complete ===")


if __name__ == "__main__":
    main()
