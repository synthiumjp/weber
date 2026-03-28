"""
M3 Stimulus Generation — E10 Random Remapping Control
======================================================
Paper M3, "Classical Minds, Modern Machines" programme.
Author: JP Cacioli
Research assistant: Claude (Anthropic)

Exploratory analysis E10 (pre-registered):
  Map numbers to arbitrary nonce tokens, preserving ordering in the task
  but removing all linguistic and tokenisation structure.

Two conditions:
  1. nonce_no_order — nonce tokens in carrier sentences with NO ordering
     information. Pure control: do nonce tokens spontaneously produce
     CP-like geometry at the boundary position?
  2. nonce_ordered — system prompt establishes the ordering explicitly.
     Tests whether ordinal structure alone (without linguistic/tokenisation
     discontinuity) is sufficient to induce CP warping.

Theoretical RDMs use ORDINAL POSITION as the continuous baseline:
  d_ij = |rank_i - rank_j|
Not log(magnitude), because nonce tokens have no magnitude.

Boundary at rank 7 (corresponding to value 10 in decade_10 mapping).
Below boundary: ranks 1-6 (mapped from values 4-9).
At/above boundary: ranks 7-17 (mapped from values 10-20).

Output format matches m3_stimuli.py exactly for m3_extract.py compatibility.

Seed = 42 throughout.
"""

import json
import itertools
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

SEED = 42
rng = np.random.default_rng(SEED)

# =============================================================================
# 1. Nonce Token Mapping
# =============================================================================

# 17 nonce tokens corresponding to values 4-20 (ranks 1-17).
# Selected for: pronounceable, no obvious numerical associations,
# no substring overlap with common English words, varied phonology.
# "glorp", "blicket", "tazmo" from pre-registration examples.
NONCE_TOKENS = [
    "glorp",    # rank 1  (value 4)
    "blicket",  # rank 2  (value 5)
    "tazmo",    # rank 3  (value 6)
    "fenwick",  # rank 4  (value 7)
    "plovent",  # rank 5  (value 8)
    "durshaw",  # rank 6  (value 9)   ← last before boundary
    "kelmis",   # rank 7  (value 10)  ← first at/above boundary
    "wibnar",   # rank 8  (value 11)
    "spundle",  # rank 9  (value 12)
    "croffen",  # rank 10 (value 13)
    "jivtex",   # rank 11 (value 14)
    "morplat",  # rank 12 (value 15)
    "zelquon",  # rank 13 (value 16)
    "braxite",  # rank 14 (value 17)
    "fumpley",  # rank 15 (value 18)
    "gostrin",  # rank 16 (value 19)
    "halvusk",  # rank 17 (value 20)
]

# Original values for reference (not used in RDMs)
ORIGINAL_VALUES = list(range(4, 21))  # [4, 5, ..., 20]

# Ranks (used for ordinal RDMs)
RANKS = list(range(1, 18))  # [1, 2, ..., 17]

# Boundary at rank 7 (corresponding to value 10)
BOUNDARY_RANK = 7

# Mapping for reference
NONCE_MAP = {
    nonce: {"rank": rank, "original_value": val}
    for nonce, rank, val in zip(NONCE_TOKENS, RANKS, ORIGINAL_VALUES)
}


# =============================================================================
# 2. Carrier Sentences
# =============================================================================

# Same structure as m3_stimuli.py: 8 sentences, split 0-3 (B0) / 4-7 (RSA).
# B0 sentences are included for format compatibility but identification
# is not meaningful for nonce tokens.
#
# Sentences are neutral — they embed the nonce token without implying
# any numerical meaning.

CARRIER_SENTENCES = [
    # --- B0 sentences (indices 0-3) — format compatibility ---
    "The symbol {N} was recorded.",
    "There were {N} units in the set.",
    "A total of {N} items were noted.",
    "The label showed {N}.",
    # --- RSA centroid sentences (indices 4-7) ---
    "Approximately {N} entries were observed.",
    "The code reached {N} in the log.",
    "A designation of {N} was reported.",
    "The register found {N} listed.",
]

B0_SENTENCE_INDICES = [0, 1, 2, 3]
RSA_SENTENCE_INDICES = [4, 5, 6, 7]

# Ordering preamble for nonce_ordered condition.
# This is prepended to each carrier sentence as context.
ORDERING_PREAMBLE = (
    "In a system where items are ordered from smallest to largest as: "
    + ", ".join(NONCE_TOKENS)
    + ". "
)


# =============================================================================
# 3. Probing Sentence Generation
# =============================================================================

def generate_probing_sentences(
    nonce_tokens: List[str],
    carrier_sentences: List[str],
    prepend_ordering: bool = False,
) -> List[dict]:
    """Generate all probing sentences for nonce tokens.

    Args:
        nonce_tokens: List of nonce token strings.
        carrier_sentences: List of carrier sentence templates.
        prepend_ordering: If True, prepend the ordering preamble to each sentence.

    Returns:
        List of dicts matching m3_stimuli.py ProbingSentence format.
    """
    sentences = []
    for idx_n, nonce in enumerate(nonce_tokens):
        rank = idx_n + 1
        for idx_s, template in enumerate(carrier_sentences):
            stype = "b0" if idx_s in B0_SENTENCE_INDICES else "rsa"
            base_text = template.format(N=nonce)
            if prepend_ordering:
                text = ORDERING_PREAMBLE + base_text
            else:
                text = base_text
            sentences.append({
                "value": rank,  # Use rank as the "value" for extraction compatibility
                "sentence_idx": idx_s,
                "sentence_type": stype,
                "text": text,
                "magnitude_token": nonce,  # The nonce token to locate
            })
    return sentences


# =============================================================================
# 4. Theoretical RDMs (Ordinal Position Basis)
# =============================================================================

def build_theoretical_rdm_ordinal(ranks: List[int]) -> np.ndarray:
    """Ordinal continuous: d_ij = |rank_i - rank_j|.

    This replaces the log-distance continuous model used for numbers.
    For nonce tokens, ordinal position is the only meaningful distance.
    """
    n = len(ranks)
    r = np.array(ranks, dtype=float)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(r[i] - r[j])
    return rdm


def build_theoretical_rdm_categorical(
    ranks: List[int], boundary_rank: int
) -> np.ndarray:
    """Categorical: d_ij = 0 if same category, 1 if different.

    Category defined by boundary_rank: < boundary = category 0,
    >= boundary = category 1.
    """
    n = len(ranks)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ci = 0 if ranks[i] < boundary_rank else 1
            cj = 0 if ranks[j] < boundary_rank else 1
            rdm[i, j] = 0.0 if ci == cj else 1.0
    return rdm


def build_theoretical_rdm_cp_additive(
    ranks: List[int], boundary_rank: int, lam: float = 1.0
) -> np.ndarray:
    """CP-Additive (ordinal): d_ij = |rank_i - rank_j| + lambda * 1[diff category].

    Lambda = 1.0 template (relative weighting estimated in analysis).
    """
    n = len(ranks)
    r = np.array(ranks, dtype=float)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ord_dist = abs(r[i] - r[j])
            ci = 0 if ranks[i] < boundary_rank else 1
            cj = 0 if ranks[j] < boundary_rank else 1
            cross = 1.0 if ci != cj else 0.0
            rdm[i, j] = ord_dist + lam * cross
    return rdm


def build_theoretical_rdm_linear(ranks: List[int]) -> np.ndarray:
    """Linear (null): d_ij = |rank_i - rank_j|.

    For ordinal data, linear = ordinal. Included for format compatibility.
    """
    return build_theoretical_rdm_ordinal(ranks)


def build_all_theoretical_rdms(
    ranks: List[int], boundary_rank: int
) -> Dict[str, np.ndarray]:
    """Build all theoretical RDMs for model comparison."""
    return {
        "continuous": build_theoretical_rdm_ordinal(ranks),
        "categorical": build_theoretical_rdm_categorical(ranks, boundary_rank),
        "cp_additive": build_theoretical_rdm_cp_additive(ranks, boundary_rank),
        "linear": build_theoretical_rdm_linear(ranks),
    }


# =============================================================================
# 5. Pairwise Distance Computation
# =============================================================================

def compute_pairwise_distances(
    ranks: List[int],
    nonce_tokens: List[str],
    boundary_rank: int,
) -> List[dict]:
    """Compute all pairwise distances with boundary classification.

    Uses ordinal distance (|rank_i - rank_j|) instead of log-distance.
    """
    pairs = []
    for (ri, ni), (rj, nj) in itertools.combinations(
        zip(ranks, nonce_tokens), 2
    ):
        ord_dist = abs(rj - ri)
        ci = 0 if ri < boundary_rank else 1
        cj = 0 if rj < boundary_rank else 1
        if ci == cj:
            if ci == 0:
                ptype = "within_below"
            else:
                ptype = "within_above"
        else:
            ptype = "cross_boundary"

        pairs.append({
            "value_a": ri,
            "value_b": rj,
            "nonce_a": ni,
            "nonce_b": nj,
            "ordinal_distance": ord_dist,
            "linear_distance": ord_dist,  # Same for ordinal data
            "log_distance": float(np.log(rj) - np.log(ri)) if ri > 0 else 0.0,
            "crosses_boundary": ptype == "cross_boundary",
            "pair_type": ptype,
            "boundary": boundary_rank,
        })
    return pairs


# =============================================================================
# 6. JSON Output (m3_extract.py compatible)
# =============================================================================

def rdm_to_serialisable(rdm: np.ndarray) -> List[List[float]]:
    """Convert numpy RDM to JSON-serialisable nested list."""
    return rdm.tolist()


def build_nonce_stimulus_file(
    condition_name: str,
    nonce_tokens: List[str],
    ranks: List[int],
    boundary_rank: int,
    prepend_ordering: bool,
    output_dir: Path,
) -> Path:
    """Build and save a complete nonce stimulus file.

    Output format matches m3_stimuli.py for m3_extract.py compatibility.
    Key difference: 'value' field contains rank (int), 'magnitude_token'
    contains the nonce string.
    """
    # Generate probing sentences
    probing_sents = generate_probing_sentences(
        nonce_tokens, CARRIER_SENTENCES, prepend_ordering=prepend_ordering
    )

    # Pairwise distances
    pairs = compute_pairwise_distances(ranks, nonce_tokens, boundary_rank)

    # Theoretical RDMs
    rdms = build_all_theoretical_rdms(ranks, boundary_rank)

    # No identification stimuli — nonce tokens have no category labels
    # Include empty list for format compatibility
    id_stimuli = []

    # Assemble output
    output = {
        "metadata": {
            "condition": condition_name,
            "boundary": boundary_rank,
            "seed": SEED,
            "version": "0.5-E10",
            "paper": "M3",
            "programme": "Classical Minds, Modern Machines",
            "analysis": "E10",
            "description": (
                f"E10 nonce remapping control — condition '{condition_name}'. "
                f"Boundary at rank {boundary_rank}. "
                f"{len(nonce_tokens)} nonce tokens, "
                f"{len(CARRIER_SENTENCES)} carrier sentences, "
                f"{len(probing_sents)} total probing sentences. "
                f"Ordering preamble: {'YES' if prepend_ordering else 'NO'}."
            ),
            "ordering_preamble": ORDERING_PREAMBLE if prepend_ordering else None,
            "nonce_map": NONCE_MAP,
            "rdm_basis": "ordinal_position",
        },
        "probing_values": ranks,
        "carrier_sentences": {
            "all": CARRIER_SENTENCES,
            "b0_indices": B0_SENTENCE_INDICES,
            "rsa_indices": RSA_SENTENCE_INDICES,
        },
        "probing_sentences": probing_sents,
        "identification_stimuli": id_stimuli,
        "pairwise_distances": pairs,
        "theoretical_rdms": {
            name: rdm_to_serialisable(rdm) for name, rdm in rdms.items()
        },
        "rdm_labels": ranks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"m3_stimuli_{condition_name}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


# =============================================================================
# 7. Summary Statistics
# =============================================================================

def print_condition_summary(
    condition_name: str,
    nonce_tokens: List[str],
    ranks: List[int],
    boundary_rank: int,
    pairs: List[dict],
    prepend_ordering: bool,
):
    """Print summary for verification."""
    n_within_below = sum(1 for p in pairs if p["pair_type"] == "within_below")
    n_within_above = sum(1 for p in pairs if p["pair_type"] == "within_above")
    n_cross = sum(1 for p in pairs if p["pair_type"] == "cross_boundary")

    print(f"\n{'='*60}")
    print(f"Condition: {condition_name} (boundary = rank {boundary_rank})")
    print(f"{'='*60}")
    print(f"Ordering preamble: {'YES' if prepend_ordering else 'NO'}")
    print(f"N nonce tokens: {len(nonce_tokens)}")
    print(f"Tokens: {', '.join(nonce_tokens)}")
    print(f"Ranks: {ranks}")
    print(f"Boundary between: {nonce_tokens[boundary_rank-2]} (rank {boundary_rank-1}) "
          f"| {nonce_tokens[boundary_rank-1]} (rank {boundary_rank})")
    print(f"N pairs: {len(pairs)}")
    print(f"  Within-below: {n_within_below}")
    print(f"  Within-above: {n_within_above}")
    print(f"  Cross-boundary: {n_cross}")
    print(f"N probing sentences: {len(nonce_tokens) * len(CARRIER_SENTENCES)}")
    print(f"  B0: {len(nonce_tokens) * len(B0_SENTENCE_INDICES)}")
    print(f"  RSA: {len(nonce_tokens) * len(RSA_SENTENCE_INDICES)}")

    # Show first RSA sentence for verification
    sents = generate_probing_sentences(
        nonce_tokens, CARRIER_SENTENCES, prepend_ordering=prepend_ordering
    )
    rsa_example = next(s for s in sents if s["sentence_type"] == "rsa")
    print(f"\nExample RSA sentence:")
    print(f"  '{rsa_example['text']}'")
    print(f"  magnitude_token = '{rsa_example['magnitude_token']}'")


# =============================================================================
# 8. Main
# =============================================================================

def main():
    output_dir = Path("stimuli")

    print("M3 E10 Nonce Remapping Stimulus Generation")
    print(f"Version: 0.5-E10 | Seed: {SEED}")
    print(f"Output directory: {output_dir.resolve()}")

    # --- Condition 1: nonce_no_order (no ordering information) ---
    pairs_no = compute_pairwise_distances(RANKS, NONCE_TOKENS, BOUNDARY_RANK)
    print_condition_summary(
        "nonce_no_order", NONCE_TOKENS, RANKS, BOUNDARY_RANK, pairs_no,
        prepend_ordering=False,
    )
    f1 = build_nonce_stimulus_file(
        "nonce_no_order", NONCE_TOKENS, RANKS, BOUNDARY_RANK,
        prepend_ordering=False, output_dir=output_dir,
    )
    print(f"  Saved: {f1}")

    # --- Condition 2: nonce_ordered (ordering preamble) ---
    pairs_ord = compute_pairwise_distances(RANKS, NONCE_TOKENS, BOUNDARY_RANK)
    print_condition_summary(
        "nonce_ordered", NONCE_TOKENS, RANKS, BOUNDARY_RANK, pairs_ord,
        prepend_ordering=True,
    )
    f2 = build_nonce_stimulus_file(
        "nonce_ordered", NONCE_TOKENS, RANKS, BOUNDARY_RANK,
        prepend_ordering=True, output_dir=output_dir,
    )
    print(f"  Saved: {f2}")

    # --- Verify theoretical RDM shapes ---
    for cond in ["nonce_no_order", "nonce_ordered"]:
        rdms = build_all_theoretical_rdms(RANKS, BOUNDARY_RANK)
        n = len(RANKS)
        for name, rdm in rdms.items():
            assert rdm.shape == (n, n), f"RDM shape mismatch: {cond}/{name}"
            assert np.allclose(rdm, rdm.T), f"RDM not symmetric: {cond}/{name}"
            assert np.allclose(np.diag(rdm), 0), f"RDM diagonal not zero: {cond}/{name}"
        print(f"  RDM checks passed for {cond} ({n}×{n})")

    # --- Verify nonce tokens are unique and no substring collisions ---
    assert len(set(NONCE_TOKENS)) == len(NONCE_TOKENS), "Duplicate nonce tokens!"
    for i, t1 in enumerate(NONCE_TOKENS):
        for j, t2 in enumerate(NONCE_TOKENS):
            if i != j:
                assert t1 not in t2, f"Substring collision: '{t1}' in '{t2}'"
    print("  Nonce token uniqueness verified (no duplicates, no substring collisions)")

    # --- Print mapping table ---
    print(f"\n{'='*60}")
    print("Nonce Token Mapping")
    print(f"{'='*60}")
    print(f"{'Rank':>4}  {'Original':>8}  {'Nonce':<10}  {'Category'}")
    print(f"{'-'*4}  {'-'*8}  {'-'*10}  {'-'*8}")
    for rank, val, nonce in zip(RANKS, ORIGINAL_VALUES, NONCE_TOKENS):
        cat = "BELOW" if rank < BOUNDARY_RANK else "AT/ABOVE"
        marker = " ← BOUNDARY" if rank == BOUNDARY_RANK else ""
        print(f"{rank:>4}  {val:>8}  {nonce:<10}  {cat}{marker}")

    print(f"\nDone. Stimulus files saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
