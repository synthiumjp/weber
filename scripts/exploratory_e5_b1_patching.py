#!/usr/bin/env python3
"""
Exploratory E5: Causal Intervention on Approximate-Processing Prompts
Weber's Law in Transformer Magnitude Representations (Project 4.2)

NOT PRE-REGISTERED. This is a post-hoc exploratory analysis motivated by
the H7 null result. The pre-registered Paradigm D used symbolic comparison
prompts where both models were at ceiling (96-100% accuracy). This exploratory
tests whether the magnitude subspace is causal for approximate magnitude
judgements (B1 cross-format), where Llama shows genuine Weber-like behaviour
(75-89% accuracy by ratio).

Design:
  - 200 B1 prompts subsampled from the 1,500 B1 stimulus set (stratified by
    baseline, seed 42)
  - Patching at two layers: (a) the pre-reg selected layer, (b) peak RSA layer
  - Same procedure as primary Paradigm D: probe direction, PCA direction,
    4 dose levels, 10 random directions
  - Patching at last token of A-expression (multi-token magnitude analogue)

Author: JP Cacioli
Date: March 2026
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAGNITUDES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50,
              60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1000]

DOSE_LEVELS = [0.25, 0.50, 0.75, 1.00]
N_RANDOM_DIRECTIONS = 10
RANDOM_SEED = 42
N_SUBSAMPLE = 200  # Match pre-registered Paradigm D size

# H7 criterion (applied exploratorily — same threshold for comparability)
H7_PERCENTILE = 97.5
H7_PROMPT_THRESHOLD = 0.75

MODELS = {
    "llama": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "n_layers": 33,
        "prereg_layer": 5,     # From primary Paradigm D gate
        "peak_rsa_layer": 23,  # Peak Weber RSA rho (cosine, layers 16-32)
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_layers": 33,
        "prereg_layer": 13,    # From primary Paradigm D gate
        "peak_rsa_layer": None,  # Will be looked up
    },
}

LLAMA_CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

MISTRAL_CHAT_TEMPLATE = (
    "<s>[INST] {prompt} [/INST]"
)


def get_chat_template(model_key):
    if model_key == "llama":
        return LLAMA_CHAT_TEMPLATE
    elif model_key == "mistral":
        return MISTRAL_CHAT_TEMPLATE
    else:
        raise ValueError(f"Unknown model key: {model_key}")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_paradigm_a_centroids(results_dir, model_key, domain="numerical"):
    """Load centroid hidden states from Paradigm A."""
    hs_path = Path(results_dir) / "paradigm_a" / f"{model_key}_instruct" / domain / "hidden_states.npz"
    if not hs_path.exists():
        raise FileNotFoundError(f"Not found: {hs_path}")
    data = np.load(str(hs_path), allow_pickle=True)
    centroids = data['centroids']
    print(f"  Loaded centroids: {centroids.shape}")
    return centroids


def load_b1_prompts(stimuli_dir, n_subsample=N_SUBSAMPLE, seed=RANDOM_SEED):
    """Load and subsample B1 cross-format prompts.
    
    Stratified by baseline (40 per baseline level) to match Paradigm D design.
    """
    path = Path(stimuli_dir) / "prompts_b1.json"
    with open(path) as f:
        all_prompts = json.load(f)
    
    print(f"  Total B1 prompts: {len(all_prompts)}")
    
    # Stratified subsample: 40 per baseline
    rng = np.random.RandomState(seed)
    baselines = sorted(set(p['nominal_baseline'] for p in all_prompts))
    per_baseline = n_subsample // len(baselines)
    
    subsampled = []
    for bl in baselines:
        bl_prompts = [p for p in all_prompts if p['nominal_baseline'] == bl]
        indices = rng.choice(len(bl_prompts), size=min(per_baseline, len(bl_prompts)), replace=False)
        for idx in indices:
            subsampled.append(bl_prompts[idx])
    
    print(f"  Subsampled: {len(subsampled)} ({per_baseline} per baseline × {len(baselines)} baselines)")
    return subsampled


def find_peak_rsa_layer(results_dir, model_key, domain="numerical"):
    """Find the layer with peak Weber RSA rho (cosine) at layers 16-32."""
    analysis_path = (Path(results_dir) / "paradigm_a" / f"{model_key}_instruct" 
                     / domain / "paradigm_a_analysis.json")
    with open(analysis_path) as f:
        data = json.load(f)
    
    best_rho = -1
    best_layer = -1
    for lname, ldata in data['layers'].items():
        layer_num = int(lname.split('_')[1])
        if layer_num < 16:
            continue
        rho = ldata.get('cosine', {}).get('rsa', {}).get('weber', {}).get('rho', -1)
        if rho > best_rho:
            best_rho = rho
            best_layer = layer_num
    
    print(f"  Peak RSA layer: {best_layer} (Weber ρ = {best_rho:.4f})")
    return best_layer


# ---------------------------------------------------------------------------
# Probe and Direction (reused from primary Paradigm D)
# ---------------------------------------------------------------------------

def train_probe(centroids, layer, alpha=1.0):
    """Train ridge probe at a single layer."""
    log_mags = np.log(np.array(MAGNITUDES, dtype=np.float64))
    X = centroids[:, layer, :]
    probe = Ridge(alpha=alpha)
    probe.fit(X, log_mags)
    r2 = probe.score(X, log_mags)
    v_mag = probe.coef_.copy()
    v_mag /= np.linalg.norm(v_mag)
    return v_mag, r2


def get_pca_direction(centroids, layer):
    """PC1 as robustness direction."""
    log_mags = np.log(np.array(MAGNITUDES, dtype=np.float64))
    X = centroids[:, layer, :]
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X).ravel()
    direction = pca.components_[0].copy()
    direction /= np.linalg.norm(direction)
    corr, _ = pearsonr(scores, log_mags)
    if corr < 0:
        direction = -direction
        corr = -corr
    return direction, float(corr)


def get_anchor_projection(centroids, layer, v_mag, anchor_magnitude=1000):
    """Projection of anchor onto magnitude direction."""
    idx = MAGNITUDES.index(anchor_magnitude)
    return float(np.dot(centroids[idx, layer, :], v_mag))


# ---------------------------------------------------------------------------
# Model and Patching
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.config.output_hidden_states = True
    model = model.to(device)
    model.eval()
    return model, tokenizer


def find_a_expression_last_token(tokenizer, full_text, a_expression):
    """Find the last token position of the A-expression in the prompt.
    
    For B1 prompts like "Which represents a larger quantity: A) two times five or B) 11?"
    we patch at the last token of "two times five" — where the model should have
    consolidated the magnitude representation.
    """
    encoding = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"][0].tolist()
    
    # Find A-expression in the text
    search = f"A) {a_expression}"
    marker_pos = full_text.find(search)
    if marker_pos < 0:
        raise ValueError(f"A-expression '{a_expression}' not found in prompt")
    
    # Character span of the A-expression (after "A) ")
    char_start = marker_pos + 3  # skip "A) "
    char_end = char_start + len(a_expression)
    
    # Find tokens covering this span
    expr_tokens = []
    for idx, (s, e) in enumerate(offsets):
        if s < char_end and e > char_start:
            expr_tokens.append(idx)
    
    if not expr_tokens:
        raise ValueError(f"No tokens found for A-expression at chars {char_start}-{char_end}")
    
    return expr_tokens[-1]  # Last token of expression


def format_b1_prompt(prompt_data, model_key):
    """Format a B1 prompt with A/B labels and chat template.
    
    B1 prompts already have A/B structure in the stimulus file.
    Just need to add explicit A)/B) labels and wrap in chat template.
    """
    a_expr = prompt_data["a_expression"]
    b_expr = prompt_data["b_expression"]
    correct = prompt_data["correct_answer"]  # "A" or "B"
    
    raw_prompt = f"Which represents a larger quantity: A) {a_expr} or B) {b_expr}? Answer with only A or B."
    
    template = get_chat_template(model_key)
    formatted = template.format(prompt=raw_prompt)
    
    return formatted, correct, a_expr


def run_patched_forward_pass(model, tokenizer, prompt_text, mag_position,
                              patch_layer, patch_vector, device="cuda"):
    """Forward pass with activation patching at specified layer and position."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    patch_tensor = torch.tensor(patch_vector, dtype=torch.float16, device=device)
    
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        hs[0, mag_position, :] += patch_tensor
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs
    
    layer_module = model.model.layers[patch_layer]
    handle = layer_module.register_forward_hook(patch_hook)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        handle.remove()
    
    return outputs


def get_comparison_probability(logits, tokenizer, correct_answer):
    """P(correct) from softmax over A/B tokens at answer position."""
    answer_logits = logits[0, -1, :]
    
    id_a = tokenizer.encode("A", add_special_tokens=False)[0]
    id_b = tokenizer.encode("B", add_special_tokens=False)[0]
    
    logits_ab = torch.stack([answer_logits[id_a].float(), answer_logits[id_b].float()])
    probs = torch.softmax(logits_ab, dim=0)
    
    return probs[0].item() if correct_answer == "A" else probs[1].item()


# ---------------------------------------------------------------------------
# Patching at a Single Layer
# ---------------------------------------------------------------------------

def run_patching_at_layer(model, tokenizer, prompts, centroids, layer,
                           v_mag, random_dirs, pca_dir, pca_corr,
                           model_key, device="cuda"):
    """Run full patching experiment at one layer.
    
    Returns list of per-prompt results.
    """
    proj_anchor = get_anchor_projection(centroids, layer, v_mag)
    
    pca_valid = abs(pca_corr) > 0.80
    if pca_valid:
        proj_anchor_pca = get_anchor_projection(centroids, layer, pca_dir)
    
    prompt_results = []
    t_start = time.time()
    
    for i, prompt_data in enumerate(prompts):
        formatted, correct, a_expr = format_b1_prompt(prompt_data, model_key)
        
        try:
            mag_pos = find_a_expression_last_token(tokenizer, formatted, a_expr)
        except ValueError as e:
            prompt_results.append({"index": i, "error": str(e)})
            continue
        
        # Unpatched baseline
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs_gpu)
        
        p_baseline = get_comparison_probability(outputs.logits, tokenizer, correct)
        
        # Get hidden state at magnitude position for projection
        h_mag = outputs.hidden_states[layer + 1][0, mag_pos, :].cpu().numpy().astype(np.float64)
        proj_mag = np.dot(h_mag, v_mag)
        
        result = {
            "index": i,
            "pair_id": prompt_data.get("pair_id", ""),
            "nominal_baseline": prompt_data.get("nominal_baseline"),
            "nominal_ratio": prompt_data.get("nominal_ratio"),
            "correct": correct,
            "a_expression": a_expr,
            "mag_position": mag_pos,
            "p_correct_baseline": p_baseline,
            "proj_first_mag": float(proj_mag),
            "magnitude_direction": {},
            "random_directions": {},
        }
        
        # Magnitude direction patching
        for dose in DOSE_LEVELS:
            patch_mag = dose * (proj_anchor - proj_mag)
            patch_vec = patch_mag * v_mag
            
            patched_out = run_patched_forward_pass(
                model, tokenizer, formatted, mag_pos, layer, patch_vec, device
            )
            p_patched = get_comparison_probability(patched_out.logits, tokenizer, correct)
            
            result["magnitude_direction"][str(dose)] = {
                "p_correct": p_patched,
                "delta_p": p_patched - p_baseline,
            }
        
        # Random directions
        full_patch_mag = proj_anchor - proj_mag
        for r_idx in range(N_RANDOM_DIRECTIONS):
            patch_vec = full_patch_mag * random_dirs[r_idx]
            
            patched_out = run_patched_forward_pass(
                model, tokenizer, formatted, mag_pos, layer, patch_vec, device
            )
            p_patched = get_comparison_probability(patched_out.logits, tokenizer, correct)
            
            result["random_directions"][str(r_idx)] = {
                "p_correct": p_patched,
                "delta_p": p_patched - p_baseline,
            }
        
        # PCA direction
        if pca_valid:
            proj_mag_pca = np.dot(h_mag, pca_dir)
            pca_results = {}
            for dose in DOSE_LEVELS:
                patch_mag = dose * (proj_anchor_pca - proj_mag_pca)
                patch_vec = patch_mag * pca_dir
                
                patched_out = run_patched_forward_pass(
                    model, tokenizer, formatted, mag_pos, layer, patch_vec, device
                )
                p_patched = get_comparison_probability(patched_out.logits, tokenizer, correct)
                
                pca_results[str(dose)] = {
                    "p_correct": p_patched,
                    "delta_p": p_patched - p_baseline,
                }
            result["pca_direction"] = pca_results
        
        prompt_results.append(result)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(prompts) - i - 1) / rate
            print(f"    Processed {i+1}/{len(prompts)} prompts "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
    
    return prompt_results


# ---------------------------------------------------------------------------
# H7-style Evaluation
# ---------------------------------------------------------------------------

def evaluate_h7_style(prompt_results):
    """Evaluate using same criterion as H7 (for comparability)."""
    valid = [p for p in prompt_results if "error" not in p]
    if not valid:
        return {"pass": False, "error": "No valid prompts"}
    
    exceeding = 0
    details = []
    
    for p in valid:
        mag_delta = abs(p["magnitude_direction"]["1.0"]["delta_p"])
        random_deltas = [abs(p["random_directions"][str(r)]["delta_p"])
                        for r in range(N_RANDOM_DIRECTIONS)]
        threshold = np.percentile(random_deltas, H7_PERCENTILE)
        exc = mag_delta > threshold
        if exc:
            exceeding += 1
        details.append({
            "index": p["index"],
            "mag_abs_delta_p": mag_delta,
            "random_97_5_pct": threshold,
            "exceeds": exc,
        })
    
    proportion = exceeding / len(valid)
    
    all_mag = [abs(p["magnitude_direction"]["1.0"]["delta_p"]) for p in valid]
    all_random = [abs(p["random_directions"][str(r)]["delta_p"])
                  for p in valid for r in range(N_RANDOM_DIRECTIONS)]
    
    dose_response = {}
    for dose in DOSE_LEVELS:
        ds = str(dose)
        deltas = [p["magnitude_direction"][ds]["delta_p"] for p in valid]
        dose_response[ds] = {
            "mean_delta_p": float(np.mean(deltas)),
            "mean_abs_delta_p": float(np.mean(np.abs(deltas))),
        }
    
    return {
        "pass": proportion >= H7_PROMPT_THRESHOLD,
        "n_valid": len(valid),
        "n_exceeding": exceeding,
        "proportion": proportion,
        "mag_mean_abs_delta": float(np.mean(all_mag)),
        "mag_median_abs_delta": float(np.median(all_mag)),
        "random_mean_abs_delta": float(np.mean(all_random)),
        "mag_to_random_ratio": float(np.mean(all_mag) / max(np.mean(all_random), 1e-10)),
        "dose_response": dose_response,
        "details": details,
        "mean_p_baseline": float(np.mean([p["p_correct_baseline"] for p in valid])),
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_exploratory_e5(model_key, project_root, device="cuda"):
    """Run E5: B1-format patching at pre-reg layer and peak RSA layer."""
    
    results_dir = Path(project_root) / "results"
    stimuli_dir = Path(project_root) / "stimuli"
    output_dir = results_dir / "paradigm_d"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = MODELS[model_key]
    
    print(f"\n{'='*70}")
    print(f"EXPLORATORY E5: B1-Format Causal Intervention — {model_key}")
    print(f"{'='*70}")
    
    # Load centroids
    print(f"\n--- Load Paradigm A centroids ---")
    centroids = load_paradigm_a_centroids(results_dir, model_key)
    
    # Find peak RSA layer if not set
    if config["peak_rsa_layer"] is None:
        config["peak_rsa_layer"] = find_peak_rsa_layer(results_dir, model_key)
    
    layers_to_test = {
        "prereg": config["prereg_layer"],
        "peak_rsa": config["peak_rsa_layer"],
    }
    
    # Train probes and get directions at both layers
    print(f"\n--- Probe and Direction Setup ---")
    layer_info = {}
    for label, layer in layers_to_test.items():
        v_mag, r2 = train_probe(centroids, layer)
        pca_dir, pca_corr = get_pca_direction(centroids, layer)
        alignment = float(np.dot(v_mag, pca_dir))
        
        layer_info[label] = {
            "layer": layer,
            "probe_r2": r2,
            "v_mag": v_mag,
            "pca_dir": pca_dir,
            "pca_corr": pca_corr,
            "probe_pca_alignment": alignment,
        }
        print(f"  {label} (layer {layer}): probe R²={r2:.4f}, "
              f"PCA corr={pca_corr:.4f}, alignment={alignment:.4f}")
    
    # Load B1 prompts
    print(f"\n--- Load B1 Prompts ---")
    prompts = load_b1_prompts(stimuli_dir)
    
    # Load model
    print(f"\n--- Load Model ---")
    model, tokenizer = load_model_and_tokenizer(config["model_id"], device)
    d_model = model.config.hidden_size
    
    # Generate random directions
    random_dirs = generate_random_directions(d_model)
    
    # Run patching at each layer
    all_results = {
        "analysis": "Exploratory E5: B1-format causal intervention",
        "note": "NOT PRE-REGISTERED. Motivated by H7 null due to ceiling accuracy on symbolic prompts.",
        "model": model_key,
        "n_prompts": len(prompts),
        "layers": {},
    }
    
    for label, info in layer_info.items():
        layer = info["layer"]
        print(f"\n--- Patching at {label} layer {layer} ---")
        
        prompt_results = run_patching_at_layer(
            model, tokenizer, prompts, centroids, layer,
            info["v_mag"], random_dirs, info["pca_dir"], info["pca_corr"],
            model_key, device
        )
        
        h7_eval = evaluate_h7_style(prompt_results)
        
        all_results["layers"][label] = {
            "layer": layer,
            "probe_r2": float(info["probe_r2"]),
            "pca_corr": info["pca_corr"],
            "probe_pca_alignment": info["probe_pca_alignment"],
            "h7_style_eval": h7_eval,
            "prompts": prompt_results,
        }
        
        print(f"\n  {label} layer {layer} results:")
        print(f"    Mean baseline accuracy: {h7_eval['mean_p_baseline']:.3f}")
        print(f"    H7-style: {'PASS' if h7_eval['pass'] else 'FAIL'} "
              f"({h7_eval['n_exceeding']}/{h7_eval['n_valid']} = {h7_eval['proportion']:.1%})")
        print(f"    Mag mean |Δp|: {h7_eval['mag_mean_abs_delta']:.4f}")
        print(f"    Random mean |Δp|: {h7_eval['random_mean_abs_delta']:.4f}")
        print(f"    Ratio: {h7_eval['mag_to_random_ratio']:.2f}x")
        print(f"    Dose-response:")
        for ds, stats in h7_eval['dose_response'].items():
            print(f"      dose {ds}: Δp = {stats['mean_delta_p']:+.4f} "
                  f"(|Δp| = {stats['mean_abs_delta_p']:.4f})")
    
    # Save
    out_path = output_dir / f"{model_key}_e5_b1_patching_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved: {out_path}")
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"E5 SUMMARY: {model_key}")
    print(f"{'='*70}")
    print(f"  {'':12s} | {'Pre-reg (L'+str(layers_to_test['prereg'])+')':>20s} | "
          f"{'Peak RSA (L'+str(layers_to_test['peak_rsa'])+')':>20s}")
    print(f"  {'-'*12}-+-{'-'*20}-+-{'-'*20}")
    for metric in ['mean_p_baseline', 'mag_mean_abs_delta', 'random_mean_abs_delta', 
                    'mag_to_random_ratio', 'proportion']:
        v1 = all_results['layers']['prereg']['h7_style_eval'][metric]
        v2 = all_results['layers']['peak_rsa']['h7_style_eval'][metric]
        fmt = '.3f' if metric == 'mean_p_baseline' else '.4f' if 'delta' in metric else '.2f' if 'ratio' in metric else '.1%'
        print(f"  {metric:12s} | {v1:>20{fmt}} | {v2:>20{fmt}}")
    print(f"{'='*70}")
    
    return all_results


def generate_random_directions(d_model, n=N_RANDOM_DIRECTIONS, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n, d_model)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Exploratory E5: B1-format patching")
    parser.add_argument("--model", choices=["llama", "mistral"], required=True)
    parser.add_argument("--project-root", type=str, default=r"C:\weber")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    run_exploratory_e5(args.model, args.project_root, args.device)


if __name__ == "__main__":
    main()
