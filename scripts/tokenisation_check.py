#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — Pre-Registration Step 2: Tokenisation Verification

Verifies which magnitude values are single-token for both primary models.
Records tokenisation for all stimuli. Required before building the stimulus archive.

Author: JP Cacioli
"""

import json
from pathlib import Path

# ── Magnitudes ──
NUMERICAL_MAGNITUDES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    150, 200, 300, 500, 700, 1000
]

CARRIERS = [
    "The number {N} is a quantity.",
    "There are {N} items.",
    "{N} was the value.",
    "The count reached {N}.",
    "Exactly {N} were measured.",
]

OUTPUT_DIR = Path("results/tokenisation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_tokenisation_llama_cpp(gguf_path, model_name):
    """
    Check tokenisation using llama-cpp-python.
    """
    from llama_cpp import Llama
    
    print(f"\nLoading {model_name}: {gguf_path}")
    model = Llama(model_path=gguf_path, n_ctx=128, n_gpu_layers=0, verbose=False)
    
    results = []
    single_token_mags = []
    
    for mag in NUMERICAL_MAGNITUDES:
        # Tokenise the bare magnitude
        mag_str = str(mag)
        tokens = model.tokenize(mag_str.encode(), add_bos=False)
        is_single = len(tokens) == 1
        
        if is_single:
            single_token_mags.append(mag)
        
        # Also check in carrier context
        carrier_tokens = {}
        for carrier in CARRIERS:
            sentence = carrier.format(N=mag)
            full_tokens = model.tokenize(sentence.encode(), add_bos=True)
            
            # Find the magnitude token(s) position
            # Strategy: tokenise prefix up to magnitude, count tokens
            prefix = carrier.split("{N}")[0].format()
            prefix_tokens = model.tokenize(prefix.encode(), add_bos=True)
            
            mag_start = len(prefix_tokens)
            
            # Tokenise the suffix to find where magnitude ends
            suffix = carrier.split("{N}")[1]
            suffix_tokens = model.tokenize(suffix.encode(), add_bos=False)
            
            mag_end = len(full_tokens) - len(suffix_tokens)
            mag_token_count = mag_end - mag_start
            
            carrier_tokens[carrier] = {
                "full_tokens": len(full_tokens),
                "mag_start": mag_start,
                "mag_end": mag_end,
                "mag_token_count": mag_token_count,
            }
        
        results.append({
            "magnitude": mag,
            "magnitude_string": mag_str,
            "bare_tokens": len(tokens),
            "bare_token_ids": tokens,
            "is_single_token": is_single,
            "carrier_analysis": carrier_tokens,
        })
        
        status = "SINGLE" if is_single else f"MULTI ({len(tokens)})"
        print(f"  {mag:>5}: {status} token(s) = {tokens}")
    
    del model
    
    summary = {
        "model": model_name,
        "total_magnitudes": len(NUMERICAL_MAGNITUDES),
        "single_token_count": len(single_token_mags),
        "single_token_magnitudes": single_token_mags,
        "multi_token_magnitudes": [m for m in NUMERICAL_MAGNITUDES if m not in single_token_mags],
        "details": results,
    }
    
    print(f"\n  Single-token: {len(single_token_mags)}/{len(NUMERICAL_MAGNITUDES)}")
    print(f"  Single-token magnitudes: {single_token_mags}")
    
    return summary


def check_tokenisation_hf(model_id, model_name):
    """
    Alternative: check tokenisation using HuggingFace tokeniser.
    Doesn't require loading the full model — just the tokeniser.
    """
    from transformers import AutoTokenizer
    
    print(f"\nLoading tokeniser: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    results = []
    single_token_mags = []
    
    for mag in NUMERICAL_MAGNITUDES:
        mag_str = str(mag)
        tokens = tokenizer.encode(mag_str, add_special_tokens=False)
        token_strs = [tokenizer.decode([t]) for t in tokens]
        is_single = len(tokens) == 1
        
        if is_single:
            single_token_mags.append(mag)
        
        # Check magnitude position in each carrier
        carrier_positions = {}
        for carrier in CARRIERS:
            sentence = carrier.format(N=mag)
            full_tokens = tokenizer.encode(sentence, add_special_tokens=True)
            full_strs = [tokenizer.decode([t]) for t in full_tokens]
            
            # Find magnitude token position by looking for the token(s)
            # that correspond to the magnitude string
            prefix = carrier.split("{N}")[0]
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
            mag_start_pos = len(prefix_tokens)
            
            carrier_positions[carrier] = {
                "mag_start_pos": mag_start_pos,
                "mag_n_tokens": len(tokens),
                "full_seq_len": len(full_tokens),
            }
        
        results.append({
            "magnitude": mag,
            "magnitude_string": mag_str,
            "token_ids": tokens,
            "token_strings": token_strs,
            "n_tokens": len(tokens),
            "is_single_token": is_single,
            "carrier_positions": carrier_positions,
        })
        
        status = "SINGLE" if is_single else f"MULTI ({len(tokens)})"
        print(f"  {mag:>5}: {status}  tokens={token_strs}")
    
    summary = {
        "model": model_name,
        "model_id": model_id,
        "total_magnitudes": len(NUMERICAL_MAGNITUDES),
        "single_token_count": len(single_token_mags),
        "single_token_magnitudes": single_token_mags,
        "multi_token_magnitudes": [m for m in NUMERICAL_MAGNITUDES if m not in single_token_mags],
        "details": results,
    }
    
    print(f"\n  Single-token: {len(single_token_mags)}/{len(NUMERICAL_MAGNITUDES)}")
    print(f"  Single-token magnitudes: {single_token_mags}")
    
    return summary


if __name__ == "__main__":
    print("Weber's Law Project 4.2 — Tokenisation Verification")
    print("=" * 60)
    print()
    print("Choose approach:")
    print("  1. HuggingFace tokeniser (no model weights needed)")
    print("  2. llama-cpp-python (needs GGUF files)")
    print()
    
    # HuggingFace approach (recommended — just downloads tokeniser)
    try:
        llama_result = check_tokenisation_hf(
            "meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct"
        )
        out_path = OUTPUT_DIR / "tokenisation_llama.json"
        with open(out_path, "w") as f:
            json.dump(llama_result, f, indent=2)
        print(f"\nSaved: {out_path}")
    except Exception as e:
        print(f"Llama tokeniser failed: {e}")
        print("You may need to authenticate with HuggingFace: huggingface-cli login")
    
    try:
        mistral_result = check_tokenisation_hf(
            "mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"
        )
        out_path = OUTPUT_DIR / "tokenisation_mistral.json"
        with open(out_path, "w") as f:
            json.dump(mistral_result, f, indent=2)
        print(f"\nSaved: {out_path}")
    except Exception as e:
        print(f"Mistral tokeniser failed: {e}")
