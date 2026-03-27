"""
M3 Hidden-State Extraction — Categorical Perception in LLM Hidden States
=========================================================================
Paper M3, "Classical Minds, Modern Machines" programme.
Author: JP Cacioli
Research assistant: Claude (Anthropic)

Extracts hidden states from LLM forward passes for M3 pilot:
  - Load model via HF Transformers (FP16, output_hidden_states=True)
  - Forward pass each probing sentence
  - Extract hidden states at all layers at the magnitude token position
  - Character-to-token offset verification (zero mismatches target)
  - Average across carrier sentences (4 for RSA) to get centroid per value per layer
  - Save centroids as .npz

Following Weber Paradigm A methodology exactly.
Pilot model: meta-llama/Meta-Llama-3-8B-Instruct (FP16).

Technical environment:
  - Python 3.12, PyTorch 2.8.0a0 with ROCm 6.4
  - AMD RX 7900 GRE (16GB VRAM)
  - Critical: $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for hidden-state extraction."""
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_short: str = "llama3-8b-instruct"
    precision: str = "fp16"  # FP16 via HuggingFace
    device: str = "cuda"
    stimulus_dir: Path = Path("stimuli")
    output_dir: Path = Path("extractions")
    conditions: Tuple[str, ...] = ("decade_10", "control_15")
    rsa_sentence_indices: Tuple[int, ...] = (4, 5, 6, 7)  # RSA centroid sentences
    seed: int = SEED
    
    # Following Weber: extract at ALL layers (embedding + transformer layers)
    # Llama-3-8B: 32 transformer layers + 1 embedding = 33 total
    # output_hidden_states=True returns all 33


# =============================================================================
# 2. Model Loading
# =============================================================================

def load_model_and_tokenizer(config: ExtractionConfig):
    """Load model and tokenizer following Weber methodology.
    
    FP16 precision via torch_dtype=torch.float16.
    output_hidden_states=True for all-layer extraction.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {config.model_name}")
    print(f"Precision: {config.precision}")
    print(f"Device: {config.device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    
    # Ensure pad token exists (Llama-3 uses eos as pad)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map=config.device,
        trust_remote_code=True,
    )
    model.eval()
    
    # Report model structure
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"Model loaded: {n_layers} transformer layers, d_model={d_model}")
    print(f"Total hidden-state layers (incl. embedding): {n_layers + 1}")
    
    # VRAM check
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
    
    return model, tokenizer


# =============================================================================
# 3. Token Position Identification
# =============================================================================

def find_magnitude_token_position(
    tokenizer,
    text: str,
    magnitude_str: str,
) -> Tuple[int, str, bool]:
    """Find the token position of the magnitude value in a probing sentence.
    
    Following Weber methodology:
      - For single-token magnitudes: return the token position
      - For multi-token magnitudes: return the LAST token position
        (final token of the magnitude expression)
      - Verify via character-to-token offset mapping
    
    Returns:
        (token_position, token_text, verified)
    """
    # Tokenise with offset mapping for verification
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    
    input_ids = encoding["input_ids"][0]
    offset_mapping = encoding["offset_mapping"][0]
    
    # Find the character position of the magnitude string in the text
    char_start = text.find(magnitude_str)
    if char_start == -1:
        raise ValueError(
            f"Magnitude string '{magnitude_str}' not found in text: '{text}'"
        )
    char_end = char_start + len(magnitude_str)
    
    # Find which tokens cover this character span
    magnitude_token_positions = []
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        tok_start, tok_end = tok_start.item(), tok_end.item()
        if tok_end == 0 and tok_start == 0:
            continue  # Skip special tokens with (0,0) offset
        # Check overlap between token span and magnitude span
        if tok_start < char_end and tok_end > char_start:
            magnitude_token_positions.append(tok_idx)
    
    if not magnitude_token_positions:
        raise ValueError(
            f"No tokens found covering magnitude '{magnitude_str}' in text: '{text}'"
        )
    
    # Following Weber: use the LAST token for multi-token magnitudes
    target_pos = magnitude_token_positions[-1]
    
    # Decode the target token for verification
    target_token_id = input_ids[target_pos].item()
    target_token_text = tokenizer.decode([target_token_id])
    
    # Verification: check that the token text contains (part of) the magnitude
    # For single-token: token should contain the full magnitude
    # For multi-token: last token should contain the final digit(s)
    verified = True
    if len(magnitude_token_positions) == 1:
        # Single-token: the decoded token should contain the magnitude
        if magnitude_str not in target_token_text.strip():
            verified = False
    else:
        # Multi-token: the last digit should appear in the last token
        if magnitude_str[-1] not in target_token_text:
            verified = False
    
    return target_pos, target_token_text.strip(), verified


def verify_all_positions(
    tokenizer,
    stimuli: List[dict],
) -> Tuple[List[dict], int]:
    """Verify token positions for all probing sentences.
    
    Following Weber: target is ZERO mismatches.
    
    Returns:
        (position_records, n_mismatches)
    """
    records = []
    n_mismatches = 0
    
    for stim in stimuli:
        text = stim["text"]
        mag_str = stim["magnitude_token"]
        
        try:
            pos, tok_text, verified = find_magnitude_token_position(
                tokenizer, text, mag_str
            )
        except ValueError as e:
            print(f"  ERROR: {e}")
            pos, tok_text, verified = -1, "ERROR", False
            n_mismatches += 1
        
        if not verified:
            n_mismatches += 1
        
        records.append({
            "value": stim["value"],
            "sentence_idx": stim["sentence_idx"],
            "text": text,
            "magnitude_token": mag_str,
            "token_position": pos,
            "decoded_token": tok_text,
            "verified": verified,
        })
    
    return records, n_mismatches


# =============================================================================
# 4. Hidden-State Extraction
# =============================================================================

@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    text: str,
    target_position: int,
    device: str = "cuda",
) -> np.ndarray:
    """Extract hidden states at all layers for a specific token position.
    
    Following Weber Paradigm A:
      - Forward pass with output_hidden_states=True
      - Extract vector at target_position from each layer
      - Returns array of shape (n_layers, d_model) in FP32
    
    output_hidden_states returns a tuple of length (n_layers + 1):
      - Index 0: embedding layer output
      - Index 1..n_layers: transformer layer outputs
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d_model)
    
    # Extract vector at target_position from each layer
    # Convert to FP32 for numerical stability in downstream analysis
    vectors = []
    for layer_hs in hidden_states:
        vec = layer_hs[0, target_position, :].float().cpu().numpy()
        vectors.append(vec)
    
    return np.stack(vectors)  # (n_layers+1, d_model)


def extract_condition(
    model,
    tokenizer,
    stimuli: List[dict],
    position_records: List[dict],
    config: ExtractionConfig,
) -> Dict[str, np.ndarray]:
    """Extract hidden states for all stimuli in a condition.
    
    Returns dict mapping:
      "all_states": (n_stimuli, n_layers, d_model) — all individual extractions
      "rsa_centroids": (n_values, n_layers, d_model) — averaged over RSA sentences
      "b0_centroids": (n_values, n_layers, d_model) — averaged over B0 sentences
      "values": array of probing values
    """
    # Build lookup from (value, sentence_idx) -> position
    pos_lookup = {}
    for rec in position_records:
        key = (rec["value"], rec["sentence_idx"])
        pos_lookup[key] = rec["token_position"]
    
    # Get unique values (sorted)
    values = sorted(set(s["value"] for s in stimuli))
    
    # Extract all hidden states
    all_extractions = {}  # (value, sentence_idx) -> (n_layers, d_model)
    
    total = len(stimuli)
    t0 = time.time()
    
    for i, stim in enumerate(stimuli):
        key = (stim["value"], stim["sentence_idx"])
        pos = pos_lookup[key]
        
        if pos < 0:
            print(f"  SKIP: {stim['text']} (invalid position)")
            continue
        
        hs = extract_hidden_states(
            model, tokenizer, stim["text"], pos, config.device
        )
        all_extractions[key] = hs
        
        if (i + 1) % 20 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"  Extracted {i+1}/{total} "
                  f"({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")
    
    # Compute centroids: average over carrier sentences per value per layer
    n_layers = next(iter(all_extractions.values())).shape[0]
    d_model = next(iter(all_extractions.values())).shape[1]
    
    rsa_centroids = np.zeros((len(values), n_layers, d_model))
    b0_centroids = np.zeros((len(values), n_layers, d_model))
    
    for vi, v in enumerate(values):
        # RSA centroids: average over RSA sentences (indices 4-7)
        rsa_vecs = []
        for si in config.rsa_sentence_indices:
            key = (v, si)
            if key in all_extractions:
                rsa_vecs.append(all_extractions[key])
        if rsa_vecs:
            rsa_centroids[vi] = np.mean(rsa_vecs, axis=0)
        
        # B0 centroids: average over B0 sentences (indices 0-3)
        b0_vecs = []
        for si in [0, 1, 2, 3]:
            key = (v, si)
            if key in all_extractions:
                b0_vecs.append(all_extractions[key])
        if b0_vecs:
            b0_centroids[vi] = np.mean(b0_vecs, axis=0)
    
    return {
        "rsa_centroids": rsa_centroids,
        "b0_centroids": b0_centroids,
        "values": np.array(values),
        "n_layers": n_layers,
        "d_model": d_model,
    }


# =============================================================================
# 5. Identification Logit Extraction (Paradigm B0)
# =============================================================================

@torch.no_grad()
def extract_identification_logits(
    model,
    tokenizer,
    prompt: str,
    target_tokens: Dict[str, List[str]],
    device: str = "cuda",
) -> Dict[str, float]:
    """Extract logit probabilities for identification targets.
    
    For each identification prompt, get the logits at the final token
    and compute softmax probabilities for the target category tokens.
    
    Returns dict with probabilities for each category.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    
    # Logits at the last token position
    logits = outputs.logits[0, -1, :]  # (vocab_size,)
    
    results = {}
    for cat_name, token_list in target_tokens.items():
        # Sum probabilities over alternative spellings
        cat_logits = []
        for tok_str in token_list:
            tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if tok_ids:
                # Use the first token ID for the target word
                cat_logits.append(logits[tok_ids[0]].item())
        
        if cat_logits:
            # Take the max logit among alternatives (e.g., "small" vs "Small")
            results[cat_name] = max(cat_logits)
        else:
            results[cat_name] = float('-inf')
    
    # Convert to probabilities via softmax over the two categories
    logit_a = results.get("category_a", float('-inf'))
    logit_b = results.get("category_b", float('-inf'))
    
    # Numerically stable softmax
    max_logit = max(logit_a, logit_b)
    exp_a = np.exp(logit_a - max_logit)
    exp_b = np.exp(logit_b - max_logit)
    total = exp_a + exp_b
    
    results["prob_category_a"] = float(exp_a / total)
    results["prob_category_b"] = float(exp_b / total)
    
    return results


def run_identification(
    model,
    tokenizer,
    id_stimuli: List[dict],
    config: ExtractionConfig,
) -> List[dict]:
    """Run identification task (Paradigm B0) for all stimuli.
    
    Returns list of dicts with value, framing, and category probabilities.
    """
    results = []
    
    for i, stim in enumerate(id_stimuli):
        logits = extract_identification_logits(
            model, tokenizer,
            stim["prompt"],
            stim["targets"],
            config.device,
        )
        
        results.append({
            "value": stim["value"],
            "framing": stim["framing"],
            "prompt": stim["prompt"],
            "prob_category_a": logits["prob_category_a"],
            "prob_category_b": logits["prob_category_b"],
            "raw_logit_a": logits.get("category_a", None),
            "raw_logit_b": logits.get("category_b", None),
        })
        
        if (i + 1) % 10 == 0 or (i + 1) == len(id_stimuli):
            print(f"  Identification: {i+1}/{len(id_stimuli)}")
    
    return results


# =============================================================================
# 6. Save Results
# =============================================================================

def save_extractions(
    extraction_data: dict,
    identification_results: List[dict],
    condition_name: str,
    config: ExtractionConfig,
):
    """Save extraction results as .npz (centroids) and .json (metadata + B0)."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save centroids as .npz
    npz_path = config.output_dir / f"m3_centroids_{condition_name}_{config.model_short}.npz"
    np.savez_compressed(
        npz_path,
        rsa_centroids=extraction_data["rsa_centroids"],
        b0_centroids=extraction_data["b0_centroids"],
        values=extraction_data["values"],
    )
    print(f"  Centroids saved: {npz_path}")
    print(f"    RSA shape: {extraction_data['rsa_centroids'].shape}")
    print(f"    B0 shape: {extraction_data['b0_centroids'].shape}")
    
    # Save metadata + identification results as JSON
    meta_path = config.output_dir / f"m3_meta_{condition_name}_{config.model_short}.json"
    meta = {
        "model": config.model_name,
        "model_short": config.model_short,
        "precision": config.precision,
        "condition": condition_name,
        "n_values": len(extraction_data["values"]),
        "n_layers": extraction_data["n_layers"],
        "d_model": extraction_data["d_model"],
        "rsa_centroid_shape": list(extraction_data["rsa_centroids"].shape),
        "values": extraction_data["values"].tolist(),
        "seed": config.seed,
        "identification_results": identification_results,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {meta_path}")


# =============================================================================
# 7. Main Pipeline
# =============================================================================

def main():
    config = ExtractionConfig()
    
    print("=" * 70)
    print("M3 Pilot Hidden-State Extraction")
    print(f"Model: {config.model_name}")
    print(f"Precision: {config.precision}")
    print(f"Conditions: {config.conditions}")
    print(f"Seed: {config.seed}")
    print("=" * 70)
    
    # Check CUDA / ROCm
    if not torch.cuda.is_available():
        print("WARNING: CUDA/ROCm not available. Falling back to CPU.")
        config.device = "cpu"
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    
    for condition in config.conditions:
        print(f"\n{'='*60}")
        print(f"Processing condition: {condition}")
        print(f"{'='*60}")
        
        # Load stimulus file
        stim_path = config.stimulus_dir / f"m3_stimuli_{condition}.json"
        if not stim_path.exists():
            print(f"  ERROR: Stimulus file not found: {stim_path}")
            print(f"  Run m3_stimuli.py first.")
            continue
        
        with open(stim_path) as f:
            stim_data = json.load(f)
        
        stimuli = stim_data["probing_sentences"]
        id_stimuli = stim_data["identification_stimuli"]
        
        print(f"  Loaded {len(stimuli)} probing sentences, "
              f"{len(id_stimuli)} identification stimuli")
        
        # Step 1: Verify token positions (Weber: zero mismatches target)
        print("\n  Step 1: Token position verification")
        position_records, n_mismatches = verify_all_positions(tokenizer, stimuli)
        print(f"  Verified {len(position_records)} sentences, "
              f"{n_mismatches} mismatches")
        if n_mismatches > 0:
            print("  WARNING: Non-zero mismatches detected!")
            for rec in position_records:
                if not rec["verified"]:
                    print(f"    MISMATCH: value={rec['value']}, "
                          f"sent_idx={rec['sentence_idx']}, "
                          f"token='{rec['decoded_token']}', "
                          f"expected='{rec['magnitude_token']}'")
        
        # Save position verification log
        config.output_dir.mkdir(parents=True, exist_ok=True)
        pos_log_path = (config.output_dir / 
                        f"m3_positions_{condition}_{config.model_short}.json")
        with open(pos_log_path, "w") as f:
            json.dump({
                "n_total": len(position_records),
                "n_mismatches": n_mismatches,
                "records": position_records,
            }, f, indent=2)
        print(f"  Position log saved: {pos_log_path}")
        
        # Step 2: Extract hidden states
        print("\n  Step 2: Hidden-state extraction")
        extraction_data = extract_condition(
            model, tokenizer, stimuli, position_records, config
        )
        
        # Step 3: Run identification task (Paradigm B0)
        print("\n  Step 3: Identification task (Paradigm B0)")
        id_results = run_identification(model, tokenizer, id_stimuli, config)
        
        # Step 4: Save results
        print("\n  Step 4: Saving results")
        save_extractions(extraction_data, id_results, condition, config)
    
    print("\n" + "=" * 70)
    print("Extraction complete.")
    print(f"Results saved to: {config.output_dir.resolve()}")
    print("=" * 70)
    
    # Cleanup GPU memory
    if config.device == "cuda":
        del model
        torch.cuda.empty_cache()
        print("GPU memory cleared.")


if __name__ == "__main__":
    main()
