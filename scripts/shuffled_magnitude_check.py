#!/usr/bin/env python3
"""
Shuffled-Magnitude Sanity Check (Section 5.7.6)
Weber's Law in Transformer Magnitude Representations (Project 4.2)

Tests whether the log-compressed geometry is a property of contextualised
representations (dynamic) or static token embeddings. Uses the same carrier
sentences but with randomly reassigned magnitudes.

If the resulting geometry tracks the SHUFFLED magnitudes (the actual numbers
present in the text), the geometry is token-driven. If it tracks NEITHER
the original nor shuffled order, the carrier sentence context dominates.

Pre-reg: "Same carrier sentences with randomly reassigned magnitudes."

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
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, NUMERICAL_MAGNITUDES, RESULTS_DIR, DOMAINS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def get_carriers(domain="numerical"):
    """Get carrier sentences from config."""
    return DOMAINS[domain]["carriers"]


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
    """Find last token position of magnitude string using offset_mapping.
    
    CRITICAL: offset_mapping is computed WITHOUT special tokens, but the
    forward pass includes a BOS token. We add 1 to account for this offset.
    This matches the approach validated in sanity_check_token_positions.py.
    """
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    char_start = text.find(mag_str)
    char_end = char_start + len(mag_str)
    tokens = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    if not tokens:
        raise ValueError(f"'{mag_str}' not found in '{text}'")
    # +1 for BOS token added during forward pass
    return tokens[-1] + 1


def extract_hidden_states(model, tokenizer, sentences, mag_strs, device="cuda"):
    """Extract hidden states at magnitude positions for a set of sentences.
    
    Returns: (n_sentences, n_layers, d_model) array
    """
    all_hs = []
    n_layers = None
    
    for i, (sent, mag_str) in enumerate(zip(sentences, mag_strs)):
        pos = find_magnitude_position(tokenizer, sent, mag_str)
        inputs = tokenizer(sent, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        
        # hidden_states is tuple of (n_layers+1) tensors: embedding + each layer
        if n_layers is None:
            n_layers = len(out.hidden_states) - 1  # subtract embedding layer
            log.info(f"  Model has {n_layers} layers ({len(out.hidden_states)} hidden states)")
        
        hs = torch.stack([out.hidden_states[l+1][0, pos, :] for l in range(n_layers)])
        all_hs.append(hs.cpu().numpy())
        
        if (i + 1) % 26 == 0:
            log.info(f"  Extracted {i+1}/{len(sentences)} sentences")
    
    return np.array(all_hs), n_layers  # (n_sentences, n_layers, d_model)


def run_shuffled_check(model_key, project_root):
    """Run the shuffled-magnitude sanity check."""
    
    stimuli_dir = Path(project_root) / "stimuli"
    results_dir = Path(project_root) / "results"
    output_dir = results_dir / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load shuffled mapping
    with open(stimuli_dir / "shuffled_magnitudes.json") as f:
        shuffled = json.load(f)
    
    mapping = shuffled["numerical"]["mapping"]  # {original_str: shuffled_str}
    original_order = shuffled["numerical"]["original_order"]  # [1, 2, 3, ...]
    shuffled_order = shuffled["numerical"]["shuffled_order"]  # [70, 30, 10, ...]
    
    log.info(f"Shuffled mapping: {len(mapping)} magnitudes")
    log.info(f"  Example: original 1 → shuffled {mapping['1']}")
    
    # Build shuffled sentences: same carriers, different numbers
    # For each original magnitude slot, insert the shuffled magnitude
    n_mags = len(original_order)
    carriers = get_carriers("numerical")
    n_carriers = len(carriers)
    
    shuffled_sentences = []
    shuffled_mag_strs = []
    
    for orig_mag in original_order:
        shuf_mag = int(mapping[str(orig_mag)])
        for carrier in carriers:
            sent = carrier.format(N=shuf_mag)
            shuffled_sentences.append(sent)
            shuffled_mag_strs.append(str(shuf_mag))
    
    log.info(f"Generated {len(shuffled_sentences)} shuffled sentences "
             f"({n_mags} magnitudes × {n_carriers} carriers)")
    
    # Load model and extract
    model, tokenizer = load_model(model_key)
    
    log.info("Extracting hidden states for shuffled stimuli...")
    t0 = time.time()
    hs, n_layers = extract_hidden_states(model, tokenizer, shuffled_sentences, shuffled_mag_strs)
    # hs shape: (n_mags * n_carriers, n_layers, d_model)
    log.info(f"Extraction done in {time.time()-t0:.1f}s")
    
    # Reshape to (n_mags, n_carriers, n_layers, d_model) and compute centroids
    hs = hs.reshape(n_mags, n_carriers, n_layers, -1)
    centroids = hs.mean(axis=1)  # (n_mags, n_layers, d_model)
    
    # Now compute pairwise cosine distances and correlate with
    # (a) log(shuffled magnitude) RDM — the numbers actually in the text
    # (b) log(original magnitude) RDM — the carrier sentence slot
    
    log_original = np.log(np.array(original_order, dtype=np.float64))
    log_shuffled = np.log(np.array(shuffled_order, dtype=np.float64))
    
    # Theoretical RDMs
    rdm_original = squareform(pdist(log_original.reshape(-1, 1), metric='euclidean'))
    rdm_shuffled = squareform(pdist(log_shuffled.reshape(-1, 1), metric='euclidean'))
    
    # Flatten upper triangle for correlation
    triu_idx = np.triu_indices(n_mags, k=1)
    rdm_orig_flat = rdm_original[triu_idx]
    rdm_shuf_flat = rdm_shuffled[triu_idx]
    
    results_per_layer = []
    
    log.info("\nLayer | ρ(shuffled) | ρ(original) | Winner")
    for layer in range(n_layers):
        X = centroids[:, layer, :]
        
        # Cosine distance RDM
        dists = pdist(X, metric='cosine')
        
        rho_shuf, p_shuf = spearmanr(dists, rdm_shuf_flat)
        rho_orig, p_orig = spearmanr(dists, rdm_orig_flat)
        
        winner = "shuffled" if rho_shuf > rho_orig else "original" if rho_orig > rho_shuf else "tie"
        
        results_per_layer.append({
            "layer": layer,
            "rho_shuffled": float(rho_shuf),
            "p_shuffled": float(p_shuf),
            "rho_original": float(rho_orig),
            "p_original": float(p_orig),
            "winner": winner,
        })
        
        if layer >= 16 or layer in [0, 5, 10]:
            log.info(f"  {layer:2d}  | {rho_shuf:+.4f}      | {rho_orig:+.4f}      | {winner}")
    
    # Summary
    primary_layers = [r for r in results_per_layer if r["layer"] >= 16]
    n_shuffled_wins = sum(1 for r in primary_layers if r["winner"] == "shuffled")
    n_original_wins = sum(1 for r in primary_layers if r["winner"] == "original")
    
    mean_rho_shuf = np.mean([r["rho_shuffled"] for r in primary_layers])
    mean_rho_orig = np.mean([r["rho_original"] for r in primary_layers])
    
    log.info(f"\nPrimary layers (16-32):")
    log.info(f"  Shuffled wins: {n_shuffled_wins}/17, Original wins: {n_original_wins}/17")
    log.info(f"  Mean ρ(shuffled): {mean_rho_shuf:.4f}, Mean ρ(original): {mean_rho_orig:.4f}")
    
    if n_shuffled_wins > n_original_wins:
        log.info("  INTERPRETATION: Geometry tracks the token (number actually present).")
        log.info("  The representation is token-driven, not carrier-driven.")
    else:
        log.info("  INTERPRETATION: Geometry does NOT track the shuffled token.")
        log.info("  Context or carrier sentences dominate the representation.")
    
    # Save
    output = {
        "model": model_key,
        "check": "shuffled_magnitude",
        "mapping": mapping,
        "original_order": original_order,
        "shuffled_order": shuffled_order,
        "layers": results_per_layer,
        "summary": {
            "primary_shuffled_wins": n_shuffled_wins,
            "primary_original_wins": n_original_wins,
            "primary_mean_rho_shuffled": float(mean_rho_shuf),
            "primary_mean_rho_original": float(mean_rho_orig),
        }
    }
    
    out_path = output_dir / f"{model_key}_shuffled_magnitude.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    log.info(f"\nSaved: {out_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Shuffled-magnitude sanity check")
    parser.add_argument("--model", choices=["llama_instruct", "mistral_instruct"], required=True)
    parser.add_argument("--project-root", type=str, default=r"C:\weber")
    args = parser.parse_args()
    
    import os
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    run_shuffled_check(args.model, args.project_root)


if __name__ == "__main__":
    main()
