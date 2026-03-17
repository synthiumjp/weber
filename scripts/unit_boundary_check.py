#!/usr/bin/env python3
"""
Unit-Boundary Manipulation Check (Section 5.7.3)
Weber's Law in Transformer Magnitude Representations (Project 4.2)

Tests whether equivalent magnitudes expressed in different units have
similar representations. E.g., "60 seconds" vs "1 minute" should be
close in activation space if the model encodes magnitude rather than
surface form.

Author: JP Cacioli
Date: March 2026
"""

import json
import logging
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, DOMAINS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def load_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODELS[model_key]
    hf_id = cfg["hf_id"]
    log.info(f"Loading {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16).to("cuda")
    model.config.output_hidden_states = True
    model.eval()
    return model, tokenizer


def find_magnitude_position(tokenizer, text, mag_str):
    """Find last token of magnitude expression. +1 for BOS."""
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    char_start = text.find(mag_str)
    if char_start == -1:
        raise ValueError(f"'{mag_str}' not found in '{text}'")
    char_end = char_start + len(mag_str)
    tokens = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    if not tokens:
        raise ValueError(f"No tokens for '{mag_str}' at {char_start}-{char_end}")
    return tokens[-1] + 1  # +1 for BOS


def extract_centroid(model, tokenizer, magnitude_str, carriers, device="cuda"):
    """Extract centroid hidden state across carrier sentences for a magnitude expression.
    
    Returns: (n_layers, d_model) array
    """
    all_hs = []
    n_layers = None
    
    for carrier in carriers:
        text = carrier.replace("{N}", magnitude_str).replace("{D}", magnitude_str)
        pos = find_magnitude_position(tokenizer, text, magnitude_str)
        
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        
        if n_layers is None:
            n_layers = len(out.hidden_states) - 1
        
        hs = torch.stack([out.hidden_states[l+1][0, pos, :] for l in range(n_layers)])
        all_hs.append(hs.cpu().numpy())
    
    centroid = np.mean(all_hs, axis=0)  # (n_layers, d_model)
    return centroid, n_layers


def cosine_similarity(a, b):
    """Cosine similarity between two vectors. Casts to float64 to avoid FP16 overflow."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(dot / (na * nb))


def run_unit_boundary_check(model_key, project_root):
    """Run the unit-boundary manipulation check."""
    
    stimuli_dir = Path(project_root) / "stimuli"
    output_dir = Path(project_root) / "results" / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load stimuli
    with open(stimuli_dir / "unit_boundary_check.json") as f:
        stimuli = json.load(f)
    
    log.info(f"\n{'='*70}")
    log.info(f"UNIT-BOUNDARY MANIPULATION CHECK — {model_key}")
    log.info(f"{'='*70}")
    
    # Load model
    model, tokenizer = load_model(model_key)
    
    all_results = {"model": model_key, "domains": {}}
    
    for domain in stimuli:
        pairs = stimuli[domain]
        carriers = DOMAINS[domain]["carriers"]
        
        log.info(f"\n--- {domain} ({len(pairs)} pairs) ---")
        
        pair_results = []
        
        for pair in pairs:
            expr_a = pair[0]  # e.g., "60 seconds"
            expr_b = pair[1]  # e.g., "1 minute"
            
            log.info(f"  Pair: '{expr_a}' vs '{expr_b}'")
            
            centroid_a, n_layers = extract_centroid(model, tokenizer, expr_a, carriers)
            centroid_b, _ = extract_centroid(model, tokenizer, expr_b, carriers)
            
            # Compute cosine similarity at each layer
            layer_sims = []
            for layer in range(n_layers):
                sim = cosine_similarity(centroid_a[layer], centroid_b[layer])
                layer_sims.append(sim)
            
            # Also compute a "different magnitude" baseline:
            # How similar are representations of genuinely different magnitudes?
            # We'll report the mean similarity for this pair across primary layers
            primary_sims = [layer_sims[l] for l in range(n_layers) if l >= 16]
            mean_primary = float(np.mean(primary_sims))
            
            log.info(f"    Mean cosine sim (layers 16+): {mean_primary:.4f}")
            log.info(f"    Layer 5: {layer_sims[5]:.4f}, Layer 16: {layer_sims[16]:.4f}, "
                     f"Layer {n_layers-1}: {layer_sims[n_layers-1]:.4f}")
            
            pair_results.append({
                "expression_a": expr_a,
                "expression_b": expr_b,
                "cosine_similarity_per_layer": layer_sims,
                "mean_primary_similarity": mean_primary,
            })
        
        # Compute baseline: mean cosine sim between NON-equivalent expressions
        # (all pairwise sims between expression_a values)
        all_centroids_a = []
        for pair in pairs:
            c, _ = extract_centroid(model, tokenizer, pair[0], carriers)
            all_centroids_a.append(c)
        
        baseline_sims = []
        for i in range(len(all_centroids_a)):
            for j in range(i+1, len(all_centroids_a)):
                for layer in range(16, n_layers):
                    sim = cosine_similarity(all_centroids_a[i][layer], all_centroids_a[j][layer])
                    baseline_sims.append(sim)
        
        mean_equivalent = float(np.mean([p["mean_primary_similarity"] for p in pair_results]))
        mean_different = float(np.mean(baseline_sims)) if baseline_sims else 0.0
        
        log.info(f"\n  SUMMARY ({domain}):")
        log.info(f"    Mean equiv-pair similarity (layers 16+): {mean_equivalent:.4f}")
        log.info(f"    Mean different-magnitude similarity:      {mean_different:.4f}")
        log.info(f"    Difference: {mean_equivalent - mean_different:+.4f}")
        
        if mean_equivalent > mean_different:
            log.info(f"    Equivalent magnitudes are MORE similar than different magnitudes.")
        else:
            log.info(f"    Equivalent magnitudes are NOT more similar than different magnitudes.")
        
        all_results["domains"][domain] = {
            "n_pairs": len(pairs),
            "pairs": pair_results,
            "mean_equivalent_similarity": mean_equivalent,
            "mean_different_magnitude_similarity": mean_different,
            "equivalent_minus_different": float(mean_equivalent - mean_different),
        }
    
    # Save
    out_path = output_dir / f"{model_key}_unit_boundary.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    log.info(f"\nSaved: {out_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Unit-boundary manipulation check")
    parser.add_argument("--model", choices=["llama_instruct", "mistral_instruct"], required=True)
    parser.add_argument("--project-root", type=str, default=r"C:\weber")
    args = parser.parse_args()
    
    import os
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    run_unit_boundary_check(args.model, args.project_root)


if __name__ == "__main__":
    main()
