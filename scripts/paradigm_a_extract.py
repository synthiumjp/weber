"""
Weber's Law Project 4.2 — Paradigm A: Hidden State Extraction
Classical Minds, Modern Machines

Step 1: Extract hidden-state vectors at magnitude token position, every layer.
Step 2: Average across 5 carrier sentences to get centroid per magnitude per layer.

Pre-registration ref: v2.7 Section 5.3 Steps 1-2, Section 7 (preprocessing specs).

Usage:
    python paradigm_a_extract.py --model llama_instruct --domain numerical
    python paradigm_a_extract.py --model llama_instruct --domain all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add scripts dir to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS, DOMAINS, N_LAYERS_TOTAL, RESULTS_DIR, NUMERICAL_MAGNITUDES,
    FREQUENCY_MATCHED_NOUNS, NOUN_CARRIERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_model(model_key: str):
    """Load model and tokenizer in FP16 on GPU.

    Uses explicit .to('cuda') instead of device_map='auto' to avoid
    accelerate hook conflicts with ROCm (Phase 0 finding: accelerate's
    forward hooks cause HIP kernel errors on AMD 7900 GRE).

    Sets output_hidden_states on the model config rather than as a
    constructor arg (transformers 5.3+ treats it as a generation flag
    in the constructor, not a model config).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = MODELS[model_key]
    hf_id = model_cfg["hf_id"]
    log.info(f"Loading {hf_id} in FP16...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.config.output_hidden_states = True
    model.eval()

    log.info(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer


def find_magnitude_token_position(
    tokenizer,
    full_text: str,
    magnitude_str: str,
    model_key: str,
) -> int:
    """
    Find the token position of the magnitude in the tokenized sentence.

    Pre-reg spec (v2.7 Section 7):
    - Single-token magnitudes: at magnitude token position.
    - Multi-token magnitudes: at final token of magnitude expression.

    Returns the 0-indexed position in the FULL token sequence (including
    special tokens like BOS), ready to index into model hidden states.

    Uses offset_mapping (character → token mapping) as the primary method.
    Falls back to prefix differencing only if offset_mapping is unavailable.
    """
    mag_start_char = full_text.find(magnitude_str)
    if mag_start_char == -1:
        raise ValueError(f"Magnitude '{magnitude_str}' not found in '{full_text}'")
    mag_end_char = mag_start_char + len(magnitude_str)

    # ── Primary method: offset_mapping ──
    # This maps each token to its character span, immune to BPE boundary shifts.
    try:
        encoding = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoding["offset_mapping"][0].tolist()

        # Find all tokens that overlap with the magnitude character span
        magnitude_token_indices = []
        for idx, (char_start, char_end) in enumerate(offsets):
            if char_start is None or char_end is None:
                continue
            # Token overlaps with magnitude span
            if char_start < mag_end_char and char_end > mag_start_char:
                magnitude_token_indices.append(idx)

        if magnitude_token_indices:
            # Pre-reg: use FINAL token of the magnitude expression
            return magnitude_token_indices[-1]

        # If offset_mapping found nothing (shouldn't happen), fall through
        log.warning(
            f"offset_mapping found no magnitude tokens for '{magnitude_str}' "
            f"in '{full_text}'. Falling back to prefix method."
        )
    except Exception as e:
        log.warning(f"offset_mapping failed ({e}). Falling back to prefix method.")

    # ── Fallback: prefix differencing ──
    # Tokenize the prefix (before magnitude) and the prefix+magnitude,
    # then infer magnitude tokens from the difference.
    prefix = full_text[:mag_start_char]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

    mag_with_context = full_text[:mag_end_char]
    mag_context_ids = tokenizer.encode(mag_with_context, add_special_tokens=False)

    n_prefix_tokens = len(prefix_ids)
    n_mag_context_tokens = len(mag_context_ids)
    n_mag_tokens = n_mag_context_tokens - n_prefix_tokens

    if n_mag_tokens < 1:
        log.warning(
            f"Magnitude '{magnitude_str}' merged with prefix token. "
            f"Using last prefix token."
        )
        # Need to account for special tokens
        full_ids_with_special = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
        raw_ids = tokenizer.encode(full_text, add_special_tokens=False)
        offset = len(full_ids_with_special) - len(raw_ids)
        return offset + n_prefix_tokens - 1

    # Final magnitude token in the raw (no special tokens) encoding
    final_mag_pos_raw = n_mag_context_tokens - 1

    # Account for special tokens (BOS etc.)
    full_ids_with_special = tokenizer(full_text, return_tensors="pt")["input_ids"][0].tolist()
    raw_ids = tokenizer.encode(full_text, add_special_tokens=False)
    offset = 0
    for i in range(len(full_ids_with_special) - len(raw_ids) + 1):
        if full_ids_with_special[i:i + len(raw_ids)] == raw_ids:
            offset = i
            break

    return final_mag_pos_raw + offset


def extract_hidden_states_single(
    model,
    tokenizer,
    text: str,
    magnitude_str: str,
    model_key: str,
) -> np.ndarray:
    """
    Extract hidden states at the magnitude token position for all layers.

    Returns: np.ndarray of shape (n_layers, d_model)
    """
    # Find magnitude position (returns full-sequence index including BOS)
    mag_pos = find_magnitude_token_position(
        tokenizer, text, magnitude_str, model_key
    )

    # Tokenize with model's special tokens
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Sanity check
    if mag_pos >= inputs["input_ids"].shape[1]:
        raise ValueError(
            f"Magnitude position {mag_pos} exceeds sequence length "
            f"{inputs['input_ids'].shape[1]} for text: '{text}'"
        )

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.hidden_states: tuple of (n_layers+1,) tensors of shape (1, seq_len, d_model)
    # Index 0 = embedding layer, 1..32 = transformer layers
    hidden_states = outputs.hidden_states  # tuple of 33 tensors

    if len(hidden_states) != N_LAYERS_TOTAL:
        raise ValueError(
            f"Expected {N_LAYERS_TOTAL} hidden state layers, got {len(hidden_states)}"
        )

    # Extract vector at magnitude position for each layer
    vectors = np.zeros((N_LAYERS_TOTAL, hidden_states[0].shape[-1]), dtype=np.float32)
    for layer_idx in range(N_LAYERS_TOTAL):
        vec = hidden_states[layer_idx][0, mag_pos, :].float().cpu().numpy()

        # Exclusion check (v2.7 Section 13): NaN or all-zero
        if np.any(np.isnan(vec)):
            log.error(f"NaN detected at layer {layer_idx} for '{text}' pos {mag_pos}")
            vec[:] = np.nan  # Flag for downstream exclusion
        elif np.all(vec == 0):
            log.error(f"All-zero vector at layer {layer_idx} for '{text}' pos {mag_pos}")
            vec[:] = np.nan

        vectors[layer_idx] = vec

    return vectors


def extract_domain(
    model,
    tokenizer,
    model_key: str,
    domain_key: str,
) -> dict:
    """
    Extract hidden states for all magnitudes × all carriers in a domain.

    Returns dict with:
        - 'per_carrier': np.ndarray (n_magnitudes, n_carriers, n_layers, d_model)
        - 'centroids': np.ndarray (n_magnitudes, n_layers, d_model)  [carrier-averaged]
        - 'magnitudes_raw': list of magnitude strings
        - 'magnitudes_numeric': list of float values
        - 'metadata': extraction metadata
    """
    domain = DOMAINS[domain_key]
    magnitudes_raw = domain["magnitudes_raw"]
    magnitudes_numeric = domain["magnitudes_numeric"]
    carriers = domain["carriers"]
    placeholder = domain["placeholder"]
    n_mags = len(magnitudes_raw)
    n_carriers = len(carriers)

    d_model = MODELS[model_key]["d_model"]

    log.info(f"Extracting {domain_key}: {n_mags} magnitudes × {n_carriers} carriers")

    per_carrier = np.zeros((n_mags, n_carriers, N_LAYERS_TOTAL, d_model), dtype=np.float32)
    extraction_log = []
    n_excluded = 0

    for i, mag_raw in enumerate(magnitudes_raw):
        for j, carrier_template in enumerate(carriers):
            sentence = carrier_template.replace(placeholder, mag_raw)
            t0 = time.time()

            try:
                vectors = extract_hidden_states_single(
                    model, tokenizer, sentence, mag_raw, model_key
                )
                per_carrier[i, j] = vectors
                elapsed = time.time() - t0

                extraction_log.append({
                    "magnitude": mag_raw,
                    "carrier_idx": j,
                    "sentence": sentence,
                    "elapsed_s": round(elapsed, 4),
                    "status": "ok",
                })
            except Exception as e:
                log.error(f"Extraction failed for '{sentence}': {e}")
                per_carrier[i, j] = np.nan
                n_excluded += 1
                extraction_log.append({
                    "magnitude": mag_raw,
                    "carrier_idx": j,
                    "sentence": sentence,
                    "status": "error",
                    "error": str(e),
                })

        if (i + 1) % 5 == 0:
            log.info(f"  Extracted {i+1}/{n_mags} magnitudes")

    # Step 2: Average across carriers (v2.7 Section 5.3 Step 2)
    # "Average hidden-state vectors across the 5 carrier sentences to obtain
    #  a single centroid representation per magnitude per layer."
    centroids = np.nanmean(per_carrier, axis=1)  # (n_mags, n_layers, d_model)

    # Compute ICC across carriers (v2.7: "Report ICC across carriers for each layer")
    icc_per_layer = compute_carrier_icc(per_carrier)

    log.info(
        f"Extraction complete. {n_excluded} items excluded. "
        f"Centroid shape: {centroids.shape}"
    )

    return {
        "per_carrier": per_carrier,
        "centroids": centroids,
        "magnitudes_raw": magnitudes_raw,
        "magnitudes_numeric": magnitudes_numeric,
        "icc_per_layer": icc_per_layer,
        "extraction_log": extraction_log,
        "n_excluded": n_excluded,
    }


def compute_carrier_icc(per_carrier: np.ndarray) -> np.ndarray:
    """
    Compute ICC(3,k) across carriers for each layer.

    ICC(3,k) = (BMS - EMS) / BMS
    where BMS = between-magnitude mean squares, EMS = error mean squares.

    Input: (n_mags, n_carriers, n_layers, d_model)
    Output: (n_layers,) array of ICC values

    We compute ICC on the pairwise cosine distances, which is what matters
    for RSA. For each carrier, compute the full RDM, then compute ICC
    across the carrier RDMs.
    """
    from scipy.spatial.distance import pdist
    from itertools import combinations

    n_mags, n_carriers, n_layers, d_model = per_carrier.shape
    n_pairs = n_mags * (n_mags - 1) // 2
    icc_values = np.zeros(n_layers)

    for layer in range(n_layers):
        # Build RDM for each carrier
        rdms = np.zeros((n_carriers, n_pairs))
        valid_carriers = 0
        for c in range(n_carriers):
            vecs = per_carrier[:, c, layer, :]  # (n_mags, d_model)
            if np.any(np.isnan(vecs)):
                continue
            # Cosine distance
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # avoid div by zero
            normed = vecs / norms
            rdms[c] = pdist(normed, metric="cosine")
            valid_carriers += 1

        if valid_carriers < 2:
            icc_values[layer] = np.nan
            continue

        rdms = rdms[:valid_carriers]

        # ICC(3,k) on the RDM vectors
        # Two-way mixed model: magnitudes are random, carriers are fixed
        k = valid_carriers  # number of raters (carriers)
        n = n_pairs  # number of targets (pairwise distances)

        grand_mean = rdms.mean()
        row_means = rdms.mean(axis=0)  # mean across carriers for each pair
        col_means = rdms.mean(axis=1)  # mean across pairs for each carrier

        ss_total = np.sum((rdms - grand_mean) ** 2)
        ss_rows = k * np.sum((row_means - grand_mean) ** 2)
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)
        ss_error = ss_total - ss_rows - ss_cols

        ms_rows = ss_rows / (n - 1) if n > 1 else 0
        ms_error = ss_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0

        # ICC(3,k) = (BMS - EMS) / BMS
        if ms_rows == 0:
            icc_values[layer] = 0.0
        else:
            icc_values[layer] = (ms_rows - ms_error) / ms_rows

    return icc_values


def extract_frequency_matched_nouns(
    model,
    tokenizer,
    model_key: str,
) -> dict:
    """
    Extract hidden states for frequency-matched nouns (v2.8 Amendment 2).
    Llama-only control.

    Uses NOUN_CARRIERS with the 26 matched nouns.
    """
    if "llama" not in model_key.lower() and "llama" not in MODELS[model_key]["hf_id"].lower():
        log.warning("Frequency-matched noun control is Llama-only per v2.8. Skipping.")
        return None

    n_nouns = len(FREQUENCY_MATCHED_NOUNS)
    n_carriers = len(NOUN_CARRIERS)
    d_model = MODELS[model_key]["d_model"]

    log.info(f"Extracting frequency-matched nouns: {n_nouns} nouns × {n_carriers} carriers")

    per_carrier = np.zeros((n_nouns, n_carriers, N_LAYERS_TOTAL, d_model), dtype=np.float32)

    for i, noun in enumerate(FREQUENCY_MATCHED_NOUNS):
        for j, carrier_template in enumerate(NOUN_CARRIERS):
            sentence = carrier_template.replace("{W}", noun)
            try:
                vectors = extract_hidden_states_single(
                    model, tokenizer, sentence, noun, model_key
                )
                per_carrier[i, j] = vectors
            except Exception as e:
                log.error(f"Noun extraction failed for '{sentence}': {e}")
                per_carrier[i, j] = np.nan

    centroids = np.nanmean(per_carrier, axis=1)
    icc_per_layer = compute_carrier_icc(per_carrier)

    return {
        "per_carrier": per_carrier,
        "centroids": centroids,
        "nouns": FREQUENCY_MATCHED_NOUNS,
        "icc_per_layer": icc_per_layer,
    }


def save_results(data: dict, model_key: str, domain_key: str, results_dir: Path):
    """Save extraction results as .npz (arrays) and .json (metadata)."""
    out_dir = results_dir / "paradigm_a" / model_key / domain_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.savez_compressed(
        out_dir / "hidden_states.npz",
        per_carrier=data["per_carrier"],
        centroids=data["centroids"],
        icc_per_layer=data["icc_per_layer"],
    )

    # Save metadata
    metadata = {
        "model_key": model_key,
        "domain_key": domain_key,
        "magnitudes_raw": data["magnitudes_raw"],
        "magnitudes_numeric": [float(m) for m in data["magnitudes_numeric"]],
        "n_excluded": data["n_excluded"],
        "centroid_shape": list(data["centroids"].shape),
        "extraction_log": data["extraction_log"],
    }
    with open(out_dir / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Saved to {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Paradigm A: Hidden State Extraction"
    )
    parser.add_argument(
        "--model", required=True,
        choices=list(MODELS.keys()),
        help="Model key from config",
    )
    parser.add_argument(
        "--domain", required=True,
        choices=list(DOMAINS.keys()) + ["all"],
        help="Domain to extract (or 'all')",
    )
    parser.add_argument(
        "--include-nouns", action="store_true",
        help="Also extract frequency-matched nouns (Llama only)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Results directory",
    )
    args = parser.parse_args()

    log.info(f"=== Paradigm A Extraction: {args.model} / {args.domain} ===")

    # Load model once
    model, tokenizer = load_model(args.model)

    # Determine domains
    domains_to_run = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]

    for domain_key in domains_to_run:
        log.info(f"\n--- Domain: {domain_key} ---")
        data = extract_domain(model, tokenizer, args.model, domain_key)
        save_results(data, args.model, domain_key, args.results_dir)

    # Frequency-matched nouns (v2.8)
    if args.include_nouns:
        log.info("\n--- Frequency-Matched Nouns ---")
        noun_data = extract_frequency_matched_nouns(model, tokenizer, args.model)
        if noun_data is not None:
            out_dir = args.results_dir / "paradigm_a" / args.model / "freq_nouns"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / "hidden_states.npz",
                per_carrier=noun_data["per_carrier"],
                centroids=noun_data["centroids"],
                icc_per_layer=noun_data["icc_per_layer"],
            )
            with open(out_dir / "extraction_metadata.json", "w") as f:
                json.dump({
                    "model_key": args.model,
                    "control": "frequency_matched_nouns",
                    "nouns": noun_data["nouns"],
                }, f, indent=2)
            log.info(f"Noun control saved to {out_dir}")

    log.info("\n=== Extraction complete ===")


if __name__ == "__main__":
    main()
