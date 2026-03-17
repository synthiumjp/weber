#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — Pre-Registration Step 1: Quantisation Sanity Check

BLOCKING GATE: Must pass before main experiment proceeds.

Extracts hidden states for 10 magnitudes from Q5_K_M and full-precision models,
computes Pearson r, Spearman rho, and Procrustes alignment of 45 pairwise distances.

Criterion: worst of Pearson r and Spearman rho > 0.95 at layers 16-32.
If worst < 0.90: flagged as serious limitation.

Requirements:
  pip install llama-cpp-python torch transformers scipy numpy
  
Run on AMD Radeon 7900 GRE (16GB VRAM).

Author: JP Cacioli
"""

import numpy as np
import json
import sys
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes

# ── Configuration ──
MAGNITUDES = [1, 5, 10, 20, 50, 100, 200, 500, 700, 1000]
CARRIER = "The number {N} is a quantity."
N_PAIRS = 45  # C(10,2)

# Model paths — UPDATE THESE to your local paths
GGUF_PATH_LLAMA = "models/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf"
GGUF_PATH_MISTRAL = "models/Mistral-7B-Instruct-v0.3-Q5_K_M.gguf"

# For full-precision comparison, use HuggingFace model IDs
HF_MODEL_LLAMA = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_MODEL_MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"

OUTPUT_DIR = Path("results/quantisation_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_hidden_states_gguf(gguf_path, sentences, magnitude_token_positions):
    """
    Extract hidden states from a GGUF model using llama-cpp-python.
    
    NOTE: llama-cpp-python hidden state extraction requires the model 
    to be loaded with embedding=True. The exact API depends on your 
    llama-cpp-python version. Adjust as needed.
    """
    from llama_cpp import Llama
    
    print(f"Loading GGUF model: {gguf_path}")
    model = Llama(
        model_path=gguf_path,
        n_ctx=512,
        n_gpu_layers=-1,  # all layers on GPU
        embedding=True,
        verbose=False,
    )
    
    all_hidden_states = []  # shape: [n_sentences, n_layers, d_model]
    
    for i, (sentence, tok_pos) in enumerate(zip(sentences, magnitude_token_positions)):
        # Tokenise
        tokens = model.tokenize(sentence.encode(), add_bos=True)
        
        # Forward pass — extract hidden states
        # NOTE: The API for hidden state extraction varies by version.
        # You may need to use model.eval() with output_hidden_states=True
        # or access via the internal context. Adjust to your version.
        model.reset()
        model.eval(tokens)
        
        # Extract hidden states at the magnitude token position
        # This is version-dependent — placeholder for the actual extraction
        # In practice, you'll need to use the low-level API:
        #   model._ctx  or similar to get per-layer activations
        
        # PLACEHOLDER: Replace with actual extraction code
        # hidden_states[layer] = model.get_hidden_state(layer, tok_pos)
        print(f"  Sentence {i+1}/{len(sentences)}: '{sentence[:50]}...' (tok_pos={tok_pos})")
    
    print("  NOTE: Hidden state extraction requires low-level llama-cpp API.")
    print("  See comments in code for implementation guidance.")
    return all_hidden_states


def extract_hidden_states_hf(model_id, sentences, magnitude_token_positions):
    """
    Extract hidden states from a HuggingFace model (full precision).
    Uses CPU offloading or 4-bit loading if VRAM is insufficient.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading HuggingFace model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Try loading with device_map="auto" for automatic CPU/GPU split
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
        )
    except Exception as e:
        print(f"  Full FP16 failed ({e}), trying 4-bit quantisation...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            output_hidden_states=True,
        )
    
    model.eval()
    all_hidden_states = {}  # {magnitude: {layer: vector}}
    
    with torch.no_grad():
        for i, (sentence, tok_pos, mag) in enumerate(
            zip(sentences, magnitude_token_positions, MAGNITUDES)
        ):
            inputs = tokenizer(sentence, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # outputs.hidden_states is a tuple of (n_layers+1,) tensors
            # Each tensor is [batch, seq_len, d_model]
            hidden_states = outputs.hidden_states
            
            layer_vectors = {}
            for layer_idx in range(len(hidden_states)):
                vec = hidden_states[layer_idx][0, tok_pos, :].cpu().numpy()
                layer_vectors[layer_idx] = vec
            
            all_hidden_states[mag] = layer_vectors
            print(f"  [{i+1}/{len(sentences)}] mag={mag}, layers={len(hidden_states)}")
    
    del model
    torch.cuda.empty_cache()
    return all_hidden_states


def compute_pairwise_distances(hidden_states, layer):
    """Compute 45 pairwise Euclidean distances for 10 magnitudes at a given layer."""
    vectors = []
    for mag in MAGNITUDES:
        vectors.append(hidden_states[mag][layer])
    vectors = np.array(vectors)
    distances = pdist(vectors, metric='euclidean')
    return distances


def procrustes_alignment(dist_a, dist_b):
    """
    Compute Procrustes alignment error between two distance matrices.
    Convert distances to coordinate matrices via MDS, then align.
    """
    from sklearn.manifold import MDS
    
    # Convert to square distance matrices
    sq_a = squareform(dist_a)
    sq_b = squareform(dist_b)
    
    # MDS to get coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords_a = mds.fit_transform(sq_a)
    coords_b = mds.fit_transform(sq_b)
    
    # Centre
    coords_a -= coords_a.mean(axis=0)
    coords_b -= coords_b.mean(axis=0)
    
    # Scale
    coords_a /= np.linalg.norm(coords_a)
    coords_b /= np.linalg.norm(coords_b)
    
    # Procrustes rotation
    R, scale = orthogonal_procrustes(coords_a, coords_b)
    coords_a_aligned = coords_a @ R
    
    # Procrustes distance (sum of squared differences)
    error = np.sum((coords_a_aligned - coords_b) ** 2)
    return error


def run_sanity_check(model_name, gguf_path, hf_model_id):
    """Run the full quantisation sanity check for one model."""
    print(f"\n{'='*60}")
    print(f"Quantisation Sanity Check: {model_name}")
    print(f"{'='*60}")
    
    # Generate sentences
    sentences = [CARRIER.format(N=mag) for mag in MAGNITUDES]
    
    # For now, assume magnitude token is at a fixed position
    # This needs to be verified per-tokeniser (Step 2)
    # Placeholder: position after "number "
    tok_positions = list(range(len(MAGNITUDES)))  # PLACEHOLDER
    
    print("\nStep 1: Extract from full-precision model")
    fp_states = extract_hidden_states_hf(hf_model_id, sentences, tok_positions)
    
    print("\nStep 2: Extract from Q5_K_M model")
    # NOTE: GGUF extraction needs custom implementation
    # For now, this is a scaffold
    print("  [SCAFFOLD] GGUF extraction requires llama-cpp low-level API")
    print("  Implement extract_hidden_states_gguf() for your setup")
    
    # PLACEHOLDER: Once both extractions work, compute metrics
    print("\nStep 3: Compare pairwise distance matrices")
    print("  Will compute Pearson r, Spearman rho, and Procrustes at layers 16-32")
    
    results = {
        "model": model_name,
        "magnitudes": MAGNITUDES,
        "n_pairs": N_PAIRS,
        "layers_checked": list(range(16, 33)),
        "status": "SCAFFOLD — implement GGUF extraction",
    }
    
    return results


def run_with_hf_comparison():
    """
    Alternative approach: compare HF FP16 vs HF 4-bit (BnB) 
    as a proxy for quantisation effects if GGUF extraction is difficult.
    """
    print("\n" + "="*60)
    print("Alternative: HF FP16 vs HF 4-bit comparison")
    print("="*60)
    print("This tests whether quantisation distorts geometry,")
    print("using HuggingFace models instead of GGUF.")
    print("If FP16 vs 4-bit correlation > 0.95, Q5_K_M is likely safe.")
    
    # This is implementable right now with HF
    # TODO: implement when ready to run


if __name__ == "__main__":
    print("Weber's Law Project 4.2 — Quantisation Sanity Check")
    print("BLOCKING GATE: Must pass before main experiment")
    print()
    print("NOTE: This script is a scaffold. You need to:")
    print("  1. Update GGUF_PATH_LLAMA and GGUF_PATH_MISTRAL")
    print("  2. Implement GGUF hidden state extraction for your llama-cpp version")
    print("  3. Or use the HF comparison approach (run_with_hf_comparison)")
    print()
    
    # Uncomment when ready:
    # run_sanity_check("Llama-3-8B-Instruct", GGUF_PATH_LLAMA, HF_MODEL_LLAMA)
    # run_sanity_check("Mistral-7B-Instruct-v0.3", GGUF_PATH_MISTRAL, HF_MODEL_MISTRAL)
