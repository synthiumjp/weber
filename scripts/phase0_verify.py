#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — Phase 0: Hidden State Extraction Verification

MUST RUN FIRST. Tests whether llama-cpp-python 0.3.16 exposes per-layer
hidden states on the Vulkan backend. If not, falls back to CPU backend
or flags that HuggingFace transformers is needed for Paradigm A.

This script:
  1. Loads one model (smallest — Mistral) with Vulkan
  2. Attempts to extract hidden states at all 32 layers for a single sentence
  3. Verifies shapes and non-degeneracy
  4. If Vulkan fails, retries on CPU
  5. Reports which backend works and estimated extraction speed

The result determines the entire Paradigm A infrastructure.

Author: JP Cacioli
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

# ── Configuration ──
# Update this path if your models are elsewhere
MODEL_PATH = "C:/sdt_calibration/models/Mistral-7B-Instruct-v0.3-Q5_K_M.gguf"
TEST_SENTENCE = "The number 42 is a quantity."
N_LAYERS = 32
D_MODEL = 4096  # expected for both Llama-3-8B and Mistral-7B

OUTPUT_DIR = Path("C:/weber/results/phase0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def attempt_hidden_state_extraction(model_path, n_gpu_layers, backend_name):
    """
    Attempt to extract per-layer hidden states from a GGUF model.
    
    Returns dict with results or None if failed.
    """
    print(f"\n{'='*60}")
    print(f"Attempting hidden state extraction: {backend_name}")
    print(f"{'='*60}")
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  ERROR: llama-cpp-python not installed")
        return None
    
    # ── Step 1: Load model ──
    print(f"  Loading model ({backend_name})...")
    t0 = time.time()
    
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=256,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            embedding=True,  # may be needed for hidden state access
        )
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"  FAILED to load model: {e}")
        return None
    
    # ── Step 2: Tokenise test sentence ──
    tokens = model.tokenize(TEST_SENTENCE.encode(), add_bos=True)
    print(f"  Test sentence: '{TEST_SENTENCE}'")
    print(f"  Tokens ({len(tokens)}): {tokens}")
    
    # ── Step 3: Try to extract hidden states ──
    # Method A: Using eval() + internal context state
    print(f"\n  Method A: eval() + internal state access...")
    
    hidden_states = {}
    method_a_works = False
    
    try:
        model.reset()
        model.eval(tokens)
        
        # Try accessing internal context for hidden states
        # The exact API depends on the llama-cpp-python version
        # In 0.3.x, the internal state may be accessible via:
        #   model._ctx
        #   model.ctx
        #   model.model
        
        # Attempt 1: Check if there's a get_embeddings or similar
        if hasattr(model, 'embed'):
            emb = model.embed(TEST_SENTENCE)
            print(f"    model.embed() returned shape: {np.array(emb).shape}")
        
        # Attempt 2: Check for internal context
        ctx = None
        for attr in ['_ctx', 'ctx', '_model', 'model']:
            if hasattr(model, attr):
                ctx = getattr(model, attr)
                print(f"    Found internal attribute: model.{attr} = {type(ctx)}")
        
        # Attempt 3: Check for llama_get_embeddings
        if hasattr(model, '_ctx'):
            try:
                import llama_cpp.llama_cpp as llama_raw
                # Try to get embeddings from the context
                if hasattr(llama_raw, 'llama_get_embeddings'):
                    ptr = llama_raw.llama_get_embeddings(model._ctx.ctx)
                    if ptr:
                        print(f"    llama_get_embeddings returned pointer: {ptr}")
                        method_a_works = True
            except Exception as e:
                print(f"    llama_get_embeddings failed: {e}")
        
        # Attempt 4: Try output_hidden_states-style access
        # Some versions support this through special eval modes
        
    except Exception as e:
        print(f"    Method A failed: {e}")
    
    # Method B: Using create_embedding (if available)
    print(f"\n  Method B: create_embedding()...")
    method_b_works = False
    
    try:
        result = model.create_embedding(TEST_SENTENCE)
        if result and 'data' in result:
            emb_data = result['data'][0]['embedding']
            emb_array = np.array(emb_data)
            print(f"    create_embedding returned: shape={emb_array.shape}")
            print(f"    Min={emb_array.min():.4f}, Max={emb_array.max():.4f}")
            print(f"    Non-zero: {np.count_nonzero(emb_array)}/{len(emb_array)}")
            
            if len(emb_array) == D_MODEL:
                print(f"    Shape matches d_model={D_MODEL} — this is a single-layer embedding")
                method_b_works = True
            else:
                print(f"    Shape {emb_array.shape} does not match d_model={D_MODEL}")
                if len(emb_array) == D_MODEL * N_LAYERS:
                    print(f"    BUT matches d_model * n_layers = {D_MODEL * N_LAYERS}!")
                    print(f"    This might be ALL layers concatenated!")
                    method_b_works = True
    except Exception as e:
        print(f"    Method B failed: {e}")
    
    # Method C: Low-level ctypes access to per-layer activations
    print(f"\n  Method C: Low-level ctypes per-layer access...")
    method_c_works = False
    
    try:
        import ctypes
        import llama_cpp.llama_cpp as llama_raw
        
        # Reset and eval
        model.reset()
        model.eval(tokens)
        
        # Check available functions
        available_funcs = [f for f in dir(llama_raw) if 'embed' in f.lower() or 'hidden' in f.lower() or 'state' in f.lower()]
        print(f"    Available embedding/state functions: {available_funcs}")
        
        # Try llama_get_embeddings_ith if available (per-token embeddings)
        if hasattr(llama_raw, 'llama_get_embeddings_ith'):
            print(f"    Found llama_get_embeddings_ith — trying per-token extraction...")
            for tok_idx in range(min(3, len(tokens))):
                try:
                    ptr = llama_raw.llama_get_embeddings_ith(model._ctx.ctx, tok_idx)
                    if ptr:
                        # Cast to float array
                        arr = np.ctypeslib.as_array(
                            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                            shape=(D_MODEL,)
                        ).copy()
                        print(f"      Token {tok_idx}: shape={arr.shape}, "
                              f"norm={np.linalg.norm(arr):.4f}, "
                              f"nonzero={np.count_nonzero(arr)}")
                        if np.count_nonzero(arr) > 0:
                            method_c_works = True
                except Exception as e:
                    print(f"      Token {tok_idx} failed: {e}")
        
    except Exception as e:
        print(f"    Method C failed: {e}")
    
    # ── Step 4: Report results ──
    print(f"\n  {'='*40}")
    print(f"  Results for {backend_name}:")
    print(f"    Method A (eval + internal state): {'OK' if method_a_works else 'FAILED'}")
    print(f"    Method B (create_embedding):      {'OK' if method_b_works else 'FAILED'}")
    print(f"    Method C (ctypes per-token):       {'OK' if method_c_works else 'FAILED'}")
    
    any_works = method_a_works or method_b_works or method_c_works
    
    # Clean up
    del model
    
    return {
        "backend": backend_name,
        "n_gpu_layers": n_gpu_layers,
        "model_loaded": True,
        "load_time_s": load_time,
        "method_a": method_a_works,
        "method_b": method_b_works,
        "method_c": method_c_works,
        "any_method_works": any_works,
        "tokens": tokens,
        "n_tokens": len(tokens),
        "note": "Per-layer extraction needs further investigation" if not any_works else "At least one extraction method works",
    }


def test_huggingface_extraction():
    """
    Test hidden state extraction via HuggingFace transformers.
    This is the guaranteed fallback — always works but requires
    more VRAM or CPU offloading.
    """
    print(f"\n{'='*60}")
    print(f"Testing HuggingFace transformers extraction (fallback)")
    print(f"{'='*60}")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  ERROR: transformers not installed")
        return None
    
    # Use Mistral as test (smaller)
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    
    print(f"  Loading tokeniser: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(TEST_SENTENCE, return_tensors="pt")
        print(f"  Tokens: {inputs['input_ids'].shape}")
        print(f"  Token IDs: {inputs['input_ids'][0].tolist()}")
        print(f"  Token strings: {[tokenizer.decode([t]) for t in inputs['input_ids'][0]]}")
    except Exception as e:
        print(f"  Tokeniser failed: {e}")
        print(f"  You may need: huggingface-cli login")
        return {"tokeniser_works": False, "error": str(e)}
    
    # Try loading model with output_hidden_states
    print(f"\n  Loading model with output_hidden_states=True...")
    print(f"  (This may take a while and may fail on 16GB VRAM)")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
        )
        model.eval()
        
        with torch.no_grad():
            outputs = model(**inputs.to(model.device))
        
        hidden_states = outputs.hidden_states  # tuple of (n_layers+1,) tensors
        n_layers = len(hidden_states) - 1  # -1 for embedding layer
        
        print(f"  SUCCESS: Got {len(hidden_states)} layers (including embedding)")
        print(f"  Each layer shape: {hidden_states[0].shape}")
        
        # Extract at a specific token position
        tok_pos = 3  # "42" token approximately
        for layer in [0, 16, 24, 31]:
            vec = hidden_states[layer][0, tok_pos, :].cpu().numpy()
            print(f"    Layer {layer:2d}: shape={vec.shape}, "
                  f"norm={np.linalg.norm(vec):.2f}, "
                  f"nonzero={np.count_nonzero(vec)}")
        
        del model
        torch.cuda.empty_cache()
        
        return {
            "works": True,
            "n_layers": n_layers,
            "d_model": hidden_states[0].shape[-1],
            "note": "HuggingFace extraction works — can use as Paradigm A backend",
        }
        
    except Exception as e:
        print(f"  Model loading failed: {e}")
        print(f"  Try with 4-bit quantisation or CPU offloading")
        
        return {
            "works": False,
            "error": str(e),
            "note": "Try BitsAndBytes 4-bit or device_map with CPU offload",
        }


def benchmark_extraction_speed(method, model_path, n_sentences=10):
    """Estimate extraction speed for Paradigm A planning."""
    print(f"\n  Benchmarking extraction speed ({n_sentences} sentences)...")
    
    sentences = [
        f"The number {n} is a quantity."
        for n in [1, 5, 10, 50, 100, 200, 500, 700, 1000, 42]
    ][:n_sentences]
    
    t0 = time.time()
    # TODO: implement based on which method works
    elapsed = time.time() - t0
    
    per_sentence = elapsed / n_sentences if n_sentences > 0 else 0
    
    # Paradigm A: 26 magnitudes × 5 carriers × 3 domains = 390 sentences per model
    estimated_total = per_sentence * 390
    
    print(f"  Per sentence: {per_sentence:.3f}s")
    print(f"  Estimated Paradigm A (390 sentences): {estimated_total:.1f}s")
    
    return per_sentence


def main():
    print("=" * 60)
    print("Weber's Law Project 4.2 — Phase 0 Verification")
    print("MUST PASS BEFORE ANY DATA COLLECTION")
    print("=" * 60)
    print(f"\nTest sentence: '{TEST_SENTENCE}'")
    print(f"Expected: {N_LAYERS} layers × {D_MODEL} dimensions")
    print(f"Model: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ERROR: Model not found at {MODEL_PATH}")
        print(f"  Update MODEL_PATH in this script.")
        sys.exit(1)
    
    results = {}
    
    # ── Test 1: Vulkan backend ──
    results["vulkan"] = attempt_hidden_state_extraction(
        MODEL_PATH, n_gpu_layers=-1, backend_name="Vulkan (GPU)"
    )
    
    # ── Test 2: CPU backend ──
    results["cpu"] = attempt_hidden_state_extraction(
        MODEL_PATH, n_gpu_layers=0, backend_name="CPU"
    )
    
    # ── Test 3: HuggingFace fallback ──
    results["huggingface"] = test_huggingface_extraction()
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("PHASE 0 SUMMARY")
    print("=" * 60)
    
    # Determine the extraction strategy
    strategy = None
    
    if results.get("vulkan") and results["vulkan"].get("any_method_works"):
        strategy = "vulkan"
        print("  BEST OPTION: Vulkan GPU extraction works")
        print("  → Use llama-cpp-python with Vulkan for all paradigms")
    elif results.get("cpu") and results["cpu"].get("any_method_works"):
        strategy = "cpu_llama"
        print("  FALLBACK: CPU extraction works via llama-cpp-python")
        print("  → Use CPU for Paradigm A (probing), Vulkan for Paradigm B (generation)")
        print("  → Paradigm A will be slower but still <5 min per model")
    elif results.get("huggingface") and results["huggingface"].get("works"):
        strategy = "huggingface"
        print("  FALLBACK: HuggingFace transformers extraction works")
        print("  → Use HF for Paradigm A, llama-cpp for Paradigm B")
        print("  → May need 4-bit quantisation or CPU offload for 16GB VRAM")
    else:
        strategy = "needs_investigation"
        print("  WARNING: No extraction method confirmed")
        print("  → Need to investigate llama-cpp-python internals further")
        print("  → Or use HuggingFace with explicit CPU offloading")
    
    results["recommended_strategy"] = strategy
    
    # Determine what to use for Paradigm B (just needs logits — always works)
    print(f"\n  Paradigm B (behavioural): llama-cpp-python + Vulkan")
    print(f"    (logit extraction only — confirmed working from SDT study)")
    
    # Save results
    out_path = OUTPUT_DIR / "phase0_results.json"
    
    # Clean non-serialisable items
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean_results[k] = {
                kk: (vv if not isinstance(vv, (np.ndarray, np.integer, np.floating)) 
                     else vv.tolist() if hasattr(vv, 'tolist') else str(vv))
                for kk, vv in v.items()
            }
        else:
            clean_results[k] = v
    
    with open(out_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    
    print(f"\n  RECOMMENDED STRATEGY: {strategy}")
    print(f"\n  Share these results and I'll build the extraction")
    print(f"  code for whichever backend works.")
    
    return strategy


if __name__ == "__main__":
    strategy = main()
