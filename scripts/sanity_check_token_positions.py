"""
Weber's Law Project 4.2 — Token Position Sanity Check
Classical Minds, Modern Machines

Verifies that the magnitude token position finding logic correctly identifies
the right token for hidden state extraction. This is load-bearing: if we
extract at the wrong position, all Paradigm A results are invalid.

What this checks:
    1. For each magnitude × carrier, shows the full tokenisation
    2. Highlights which token the extraction code would select
    3. Compares against ground truth (the token that actually contains the magnitude)
    4. Flags any mismatches

Run this BEFORE paradigm_a_extract.py. If any mismatches, fix the position
logic before proceeding.

Usage:
    python sanity_check_token_positions.py --model llama_instruct
    python sanity_check_token_positions.py --model mistral_instruct
    python sanity_check_token_positions.py --model llama_instruct --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, DOMAINS, RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_tokenizer(model_key: str):
    """Load tokenizer only (no model weights needed for this check)."""
    from transformers import AutoTokenizer
    hf_id = MODELS[model_key]["hf_id"]
    log.info(f"Loading tokenizer for {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def analyse_tokenisation(
    tokenizer,
    sentence: str,
    magnitude_str: str,
    model_key: str,
    verbose: bool = False,
) -> dict:
    """
    Analyse how a sentence is tokenised and where the magnitude token(s) are.

    Returns a dict with:
        - tokens: list of token strings
        - token_ids: list of token IDs
        - magnitude_token_indices: list of indices that correspond to the magnitude
        - selected_index: the index our extraction code would pick
        - expected_index: the index we SHOULD pick (ground truth)
        - match: whether selected == expected
    """
    # Full tokenisation WITH special tokens (as the model sees it)
    full_encoding = tokenizer(sentence, return_tensors="pt")
    full_ids = full_encoding["input_ids"][0].tolist()
    full_tokens = [tokenizer.decode([tid]) for tid in full_ids]

    # Also get the "clean" token strings for display
    full_token_strs = []
    for tid in full_ids:
        # Show the raw token with repr to see spaces/special chars
        decoded = tokenizer.decode([tid])
        full_token_strs.append(decoded)

    # Find magnitude in the character-level string
    mag_start_char = sentence.find(magnitude_str)
    if mag_start_char == -1:
        return {
            "error": f"Magnitude '{magnitude_str}' not found in sentence",
            "sentence": sentence,
        }

    # Method 1: Character offset mapping (most reliable)
    # Use the tokenizer's offset mapping if available
    try:
        encoding_with_offsets = tokenizer(
            sentence, return_offsets_mapping=True, return_tensors="pt"
        )
        offsets = encoding_with_offsets.get("offset_mapping")
        if offsets is not None:
            offsets = offsets[0].tolist()
        else:
            offsets = None
    except Exception:
        offsets = None

    mag_end_char = mag_start_char + len(magnitude_str)
    magnitude_token_indices = []

    if offsets is not None:
        # Use offset mapping — this is the gold standard
        for idx, (start, end) in enumerate(offsets):
            if start is None or end is None:
                continue
            # Token overlaps with magnitude span
            if start < mag_end_char and end > mag_start_char:
                magnitude_token_indices.append(idx)
    else:
        # Fallback: reconstruct from prefix tokenisation
        # Tokenize everything before the magnitude
        prefix = sentence[:mag_start_char]
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

        # Tokenize up to end of magnitude
        prefix_plus_mag = sentence[:mag_end_char]
        prefix_mag_ids = tokenizer.encode(prefix_plus_mag, add_special_tokens=False)

        n_prefix = len(prefix_ids)
        n_prefix_mag = len(prefix_mag_ids)

        # Account for BOS/special tokens
        # Find where the raw tokens start in the full sequence
        raw_ids = tokenizer.encode(sentence, add_special_tokens=False)
        offset = len(full_ids) - len(raw_ids)

        magnitude_token_indices = list(range(
            offset + n_prefix,
            offset + n_prefix_mag,
        ))

    # What our extraction code would pick
    # Import the ACTUAL function from the extraction module
    from paradigm_a_extract import find_magnitude_token_position
    selected_index = find_magnitude_token_position(
        tokenizer, sentence, magnitude_str, model_key
    )

    # Expected: per pre-reg spec
    # Single-token: at the magnitude token
    # Multi-token: at the FINAL token of the magnitude expression
    if magnitude_token_indices:
        expected_index = magnitude_token_indices[-1]  # Final token
    else:
        expected_index = None

    match = selected_index == expected_index

    result = {
        "sentence": sentence,
        "magnitude_str": magnitude_str,
        "n_tokens_total": len(full_ids),
        "tokens": full_token_strs,
        "token_ids": full_ids,
        "magnitude_token_indices": magnitude_token_indices,
        "n_magnitude_tokens": len(magnitude_token_indices),
        "selected_index": selected_index,
        "expected_index": expected_index,
        "match": match,
    }

    if verbose or not match:
        _print_tokenisation(result)

    return result


def _print_tokenisation(result: dict):
    """Pretty-print a tokenisation analysis."""
    status = "✓ MATCH" if result["match"] else "✗ MISMATCH"
    print(f"\n{'─'*70}")
    print(f"  {status} | \"{result['sentence']}\"")
    print(f"  Magnitude: \"{result['magnitude_str']}\" "
          f"({result['n_magnitude_tokens']} token{'s' if result['n_magnitude_tokens'] != 1 else ''})")

    # Show token sequence with highlighting
    tokens = result["tokens"]
    mag_indices = set(result["magnitude_token_indices"])
    selected = result["selected_index"]
    expected = result["expected_index"]

    token_display = []
    for i, tok in enumerate(tokens):
        marker = ""
        if i == selected and i == expected:
            marker = " ←[SELECTED=EXPECTED]"
        elif i == selected:
            marker = " ←[SELECTED]"
        elif i == expected:
            marker = " ←[EXPECTED]"

        highlight = ">>>" if i in mag_indices else "   "
        token_display.append(f"    {highlight} [{i:2d}] {repr(tok)}{marker}")

    print("\n".join(token_display))

    if not result["match"]:
        print(f"\n  ⚠ EXTRACTION WOULD USE INDEX {selected}, "
              f"BUT CORRECT IS {expected}")


def run_check(
    model_key: str,
    verbose: bool = False,
) -> dict:
    """Run the full sanity check across all domains."""
    tokenizer = load_tokenizer(model_key)

    results = {"model_key": model_key, "domains": {}, "summary": {}}
    total_checked = 0
    total_matched = 0
    mismatches = []

    for domain_key, domain_cfg in DOMAINS.items():
        log.info(f"\n{'='*60}")
        log.info(f"Domain: {domain_key}")
        log.info(f"{'='*60}")

        domain_results = []
        magnitudes = domain_cfg["magnitudes_raw"]
        carriers = domain_cfg["carriers"]
        placeholder = domain_cfg["placeholder"]

        for mag in magnitudes:
            for carrier_template in carriers:
                sentence = carrier_template.replace(placeholder, mag)
                result = analyse_tokenisation(
                    tokenizer, sentence, mag, model_key, verbose
                )
                domain_results.append(result)
                total_checked += 1

                if result.get("match", False):
                    total_matched += 1
                elif "error" not in result:
                    mismatches.append(result)

        # Domain summary
        n_domain = len(domain_results)
        n_match = sum(1 for r in domain_results if r.get("match", False))
        n_errors = sum(1 for r in domain_results if "error" in r)

        log.info(f"\n  {domain_key}: {n_match}/{n_domain} match, {n_errors} errors")

        # Token count summary
        token_counts = [r["n_magnitude_tokens"] for r in domain_results if "n_magnitude_tokens" in r]
        if token_counts:
            single = sum(1 for c in token_counts if c == 1)
            multi = sum(1 for c in token_counts if c > 1)
            log.info(f"  Tokenisation: {single} single-token, {multi} multi-token magnitudes")

        results["domains"][domain_key] = {
            "n_checked": n_domain,
            "n_matched": n_match,
            "n_errors": n_errors,
            "n_single_token": sum(1 for c in token_counts if c == 1) if token_counts else 0,
            "n_multi_token": sum(1 for c in token_counts if c > 1) if token_counts else 0,
        }

    # Overall summary
    results["summary"] = {
        "total_checked": total_checked,
        "total_matched": total_matched,
        "total_mismatched": len(mismatches),
        "pass": len(mismatches) == 0,
    }

    log.info(f"\n{'='*60}")
    log.info(f"OVERALL: {total_matched}/{total_checked} positions correct")
    if mismatches:
        log.error(f"  ⚠ {len(mismatches)} MISMATCHES — extraction code needs fixing!")
        log.error("  Mismatched items:")
        for m in mismatches[:10]:
            log.error(f"    \"{m['sentence']}\" mag=\"{m['magnitude_str']}\" "
                      f"selected={m['selected_index']} expected={m['expected_index']}")
        if len(mismatches) > 10:
            log.error(f"    ... and {len(mismatches) - 10} more")
    else:
        log.info("  ✓ ALL POSITIONS CORRECT — safe to proceed with extraction")
    log.info(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Token Position Sanity Check — run BEFORE extraction"
    )
    parser.add_argument(
        "--model", required=True,
        choices=list(MODELS.keys()),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full tokenisation for ALL items (not just mismatches)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Where to save the check results",
    )
    args = parser.parse_args()

    results = run_check(args.model, verbose=args.verbose)

    # Save results
    out_dir = args.results_dir / "sanity_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"token_positions_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_path}")

    # Exit code reflects pass/fail
    sys.exit(0 if results["summary"]["pass"] else 1)


if __name__ == "__main__":
    main()
