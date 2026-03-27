"""
M3 Stimulus Generation — Categorical Perception in LLM Hidden States
=====================================================================
Paper M3, "Classical Minds, Modern Machines" programme.
Author: JP Cacioli
Research assistant: Claude (Anthropic)

Generates stimulus files for the M3 pilot experiment:
  - Probing values for decade_10 and control_15 conditions
  - 8 carrier sentences (split: 0-3 for B0 identification, 4-7 for RSA)
  - Identification prompts (two framings)
  - Pairwise distance computation with boundary classification
  - Output: JSON stimulus files

Following m3_project_outline.md v0.4 specification.
Seed = 42 throughout.
"""

import json
import itertools
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional

SEED = 42
rng = np.random.default_rng(SEED)

# =============================================================================
# 1. Probing Values (Section 3.1 of v0.4)
# =============================================================================

# Primary condition: decade boundary at 10
# "Within-category (below 10): 4, 5, 6, 7, 8, 9
#  Boundary region:            9, 10, 11, 12
#  Within-category (above 10): 11, 12, 13, 14, 15, 16"
# Deduplicated and sorted:
DECADE_10_VALUES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Control condition: non-linguistic boundary at 15 (Condition NB)
# "Non-linguistic probing values centred on 15: 11, 12, 13, 14, 15, 16, 17, 18, 19"
CONTROL_15_VALUES = [11, 12, 13, 14, 15, 16, 17, 18, 19]

# Boundary definitions
DECADE_10_BOUNDARY = 10  # Category: < 10 vs >= 10
CONTROL_15_BOUNDARY = 15  # No real boundary — used as null comparison


# =============================================================================
# 2. Carrier Sentences (Section 3.1 of v0.4)
# =============================================================================

# 8 carrier sentences from the outline.
# Split: 0-3 for B0 (identification), 4-7 for Paradigm A (RSA centroids).
# This follows Rogers & Davis (2009) to avoid repetition artefacts.

CARRIER_SENTENCES = [
    # --- B0 identification sentences (indices 0-3) ---
    "The number {N} is a quantity.",
    "There are {N} items in the collection.",
    "A total of {N} units were recorded.",
    "The measurement showed {N}.",
    # --- RSA centroid sentences (indices 4-7) ---
    "Approximately {N} cases were observed.",
    "The count reached {N} in total.",
    "A value of {N} was reported.",
    "The survey found {N} instances.",
]

B0_SENTENCE_INDICES = [0, 1, 2, 3]
RSA_SENTENCE_INDICES = [4, 5, 6, 7]


# =============================================================================
# 3. Identification Prompts (Paradigm B0, Section 5.0 of v0.4)
# =============================================================================

# Two framings for identification robustness check.
# Framing 1: "small/large" — general magnitude category
# Framing 2: "single-digit/multi-digit" — precise for decade boundary

IDENTIFICATION_PROMPTS = {
    "small_large": (
        "Is {N} a small number or a large number? "
        "Answer with one word: small or large."
    ),
    "single_multi": (
        "Is {N} a single-digit number or a multi-digit number? "
        "Answer with one word: single-digit or multi-digit."
    ),
}

# Target tokens for logit extraction per framing
IDENTIFICATION_TARGETS = {
    "small_large": {
        "category_a": ["small", "Small"],
        "category_b": ["large", "Large"],
    },
    "single_multi": {
        "category_a": ["single", "Single"],
        "category_b": ["multi", "Multi"],
    },
}


# =============================================================================
# 4. Probing Sentence Generation
# =============================================================================

@dataclass
class ProbingSentence:
    """A single probing sentence: carrier + magnitude value."""
    value: int
    sentence_idx: int
    sentence_type: str  # "b0" or "rsa"
    text: str
    magnitude_token: str  # the token string for the number


def generate_probing_sentences(
    values: List[int],
    carrier_sentences: List[str] = CARRIER_SENTENCES,
) -> List[ProbingSentence]:
    """Generate all probing sentences for a set of values."""
    sentences = []
    for v in values:
        mag_token = str(v)
        for idx, template in enumerate(carrier_sentences):
            stype = "b0" if idx in B0_SENTENCE_INDICES else "rsa"
            text = template.format(N=mag_token)
            sentences.append(ProbingSentence(
                value=v,
                sentence_idx=idx,
                sentence_type=stype,
                text=text,
                magnitude_token=mag_token,
            ))
    return sentences


# =============================================================================
# 5. Pairwise Distance Computation with Boundary Classification
# =============================================================================

@dataclass
class StimPair:
    """A pair of probing values with distance and boundary classification."""
    value_a: int
    value_b: int
    log_distance: float
    linear_distance: int
    crosses_boundary: bool
    boundary: int
    pair_type: str  # "within_below", "within_above", "cross_boundary"


def classify_pair(
    a: int, b: int, boundary: int
) -> str:
    """Classify a pair relative to a boundary."""
    a_below = a < boundary
    b_below = b < boundary
    if a_below and b_below:
        return "within_below"
    elif not a_below and not b_below:
        return "within_above"
    else:
        return "cross_boundary"


def compute_pairwise_distances(
    values: List[int],
    boundary: int,
) -> List[StimPair]:
    """Compute all pairwise distances with boundary classification.
    
    Log-distance = |log(a) - log(b)| following Weber methodology.
    Boundary crossing: whether the pair straddles the boundary.
    """
    pairs = []
    for a, b in itertools.combinations(sorted(values), 2):
        log_dist = abs(np.log(b) - np.log(a))
        lin_dist = abs(b - a)
        crosses = classify_pair(a, b, boundary) == "cross_boundary"
        ptype = classify_pair(a, b, boundary)
        pairs.append(StimPair(
            value_a=a,
            value_b=b,
            log_distance=float(log_dist),
            linear_distance=int(lin_dist),
            crosses_boundary=crosses,
            boundary=boundary,
            pair_type=ptype,
        ))
    return pairs


# =============================================================================
# 6. Theoretical RDMs (Section 5.1 of v0.4)
# =============================================================================

def build_theoretical_rdm_continuous(values: List[int]) -> np.ndarray:
    """Continuous (Weber/Log): d_ij = |log(x_i) - log(x_j)|"""
    n = len(values)
    rdm = np.zeros((n, n))
    log_vals = np.log(np.array(values, dtype=float))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(log_vals[i] - log_vals[j])
    return rdm


def build_theoretical_rdm_categorical(
    values: List[int], boundary: int
) -> np.ndarray:
    """Categorical: d_ij = 0 if same category, 1 if different category."""
    n = len(values)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cat_i = 0 if values[i] < boundary else 1
            cat_j = 0 if values[j] < boundary else 1
            rdm[i, j] = 0.0 if cat_i == cat_j else 1.0
    return rdm


def build_theoretical_rdm_cp_additive(
    values: List[int], boundary: int, lam: float = 1.0
) -> np.ndarray:
    """CP-Additive: d_ij = |log(x_i) - log(x_j)| + lambda * 1[different category].
    
    Lambda=1.0 is a placeholder; in analysis, lambda is estimated by ML.
    For the pilot, we use lambda=1.0 for the theoretical RDM template.
    """
    n = len(values)
    log_vals = np.log(np.array(values, dtype=float))
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            log_dist = abs(log_vals[i] - log_vals[j])
            cat_i = 0 if values[i] < boundary else 1
            cat_j = 0 if values[j] < boundary else 1
            cross = 1.0 if cat_i != cat_j else 0.0
            rdm[i, j] = log_dist + lam * cross
    return rdm


def build_theoretical_rdm_linear(values: List[int]) -> np.ndarray:
    """Linear (null model): d_ij = |x_i - x_j|"""
    n = len(values)
    vals = np.array(values, dtype=float)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(vals[i] - vals[j])
    return rdm


def build_all_theoretical_rdms(
    values: List[int], boundary: int
) -> Dict[str, np.ndarray]:
    """Build all theoretical RDMs for model comparison."""
    return {
        "continuous": build_theoretical_rdm_continuous(values),
        "categorical": build_theoretical_rdm_categorical(values, boundary),
        "cp_additive": build_theoretical_rdm_cp_additive(values, boundary),
        "linear": build_theoretical_rdm_linear(values),
    }


# =============================================================================
# 7. JSON Serialisation and Output
# =============================================================================

def rdm_to_serialisable(rdm: np.ndarray) -> List[List[float]]:
    """Convert numpy RDM to JSON-serialisable nested list."""
    return rdm.tolist()


def build_condition_stimulus_file(
    condition_name: str,
    values: List[int],
    boundary: int,
    output_dir: Path,
) -> Path:
    """Build and save a complete stimulus file for one condition.
    
    Contains:
      - metadata (condition, boundary, seed, version)
      - probing_values
      - carrier_sentences with allocation
      - probing_sentences (all sentences for extraction)
      - identification_prompts (B0 stimuli)
      - pairwise_distances (with boundary classification)
      - theoretical_rdms (Continuous, Categorical, CP-Additive, Linear)
    """
    # Generate all probing sentences
    probing_sents = generate_probing_sentences(values)
    
    # Generate pairwise distances
    pairs = compute_pairwise_distances(values, boundary)
    
    # Build theoretical RDMs
    rdms = build_all_theoretical_rdms(values, boundary)
    
    # Build identification stimuli (one per value × framing)
    id_stimuli = []
    for v in values:
        for framing_key, prompt_template in IDENTIFICATION_PROMPTS.items():
            id_stimuli.append({
                "value": v,
                "framing": framing_key,
                "prompt": prompt_template.format(N=str(v)),
                "targets": IDENTIFICATION_TARGETS[framing_key],
            })
    
    # Assemble output
    output = {
        "metadata": {
            "condition": condition_name,
            "boundary": boundary,
            "seed": SEED,
            "version": "0.4",
            "paper": "M3",
            "programme": "Classical Minds, Modern Machines",
            "description": (
                f"Stimulus file for M3 pilot — condition '{condition_name}'. "
                f"Boundary at {boundary}. "
                f"{len(values)} probing values, {len(CARRIER_SENTENCES)} carrier sentences, "
                f"{len(probing_sents)} total probing sentences, "
                f"{len(pairs)} pairwise distance entries."
            ),
        },
        "probing_values": values,
        "carrier_sentences": {
            "all": CARRIER_SENTENCES,
            "b0_indices": B0_SENTENCE_INDICES,
            "rsa_indices": RSA_SENTENCE_INDICES,
        },
        "probing_sentences": [
            {
                "value": s.value,
                "sentence_idx": s.sentence_idx,
                "sentence_type": s.sentence_type,
                "text": s.text,
                "magnitude_token": s.magnitude_token,
            }
            for s in probing_sents
        ],
        "identification_stimuli": id_stimuli,
        "pairwise_distances": [
            {
                "value_a": p.value_a,
                "value_b": p.value_b,
                "log_distance": round(p.log_distance, 6),
                "linear_distance": p.linear_distance,
                "crosses_boundary": p.crosses_boundary,
                "pair_type": p.pair_type,
                "boundary": p.boundary,
            }
            for p in pairs
        ],
        "theoretical_rdms": {
            name: rdm_to_serialisable(rdm) for name, rdm in rdms.items()
        },
        "rdm_labels": values,
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"m3_stimuli_{condition_name}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    return filepath


# =============================================================================
# 8. Summary Statistics
# =============================================================================

def print_condition_summary(
    condition_name: str,
    values: List[int],
    boundary: int,
    pairs: List[StimPair],
):
    """Print a summary of the condition for verification."""
    n_within_below = sum(1 for p in pairs if p.pair_type == "within_below")
    n_within_above = sum(1 for p in pairs if p.pair_type == "within_above")
    n_cross = sum(1 for p in pairs if p.pair_type == "cross_boundary")
    
    # Log-distance stats for cross vs within
    cross_dists = [p.log_distance for p in pairs if p.crosses_boundary]
    within_dists = [p.log_distance for p in pairs if not p.crosses_boundary]
    
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name} (boundary = {boundary})")
    print(f"{'='*60}")
    print(f"Probing values: {values}")
    print(f"N values: {len(values)}")
    print(f"N pairs: {len(pairs)}")
    print(f"  Within-below: {n_within_below}")
    print(f"  Within-above: {n_within_above}")
    print(f"  Cross-boundary: {n_cross}")
    print(f"Log-distance range (cross): "
          f"[{min(cross_dists):.3f}, {max(cross_dists):.3f}]" if cross_dists else "N/A")
    print(f"Log-distance range (within): "
          f"[{min(within_dists):.3f}, {max(within_dists):.3f}]" if within_dists else "N/A")
    print(f"N probing sentences: {len(values) * len(CARRIER_SENTENCES)}")
    print(f"  B0 (identification): {len(values) * len(B0_SENTENCE_INDICES)}")
    print(f"  RSA (centroids): {len(values) * len(RSA_SENTENCE_INDICES)}")
    print(f"N identification stimuli: {len(values) * len(IDENTIFICATION_PROMPTS)}")


# =============================================================================
# 9. Main
# =============================================================================

def main():
    output_dir = Path("stimuli")
    
    print("M3 Pilot Stimulus Generation")
    print(f"Version: 0.4 | Seed: {SEED}")
    print(f"Output directory: {output_dir.resolve()}")
    
    # --- Condition 1: decade_10 ---
    pairs_10 = compute_pairwise_distances(DECADE_10_VALUES, DECADE_10_BOUNDARY)
    print_condition_summary("decade_10", DECADE_10_VALUES, DECADE_10_BOUNDARY, pairs_10)
    f1 = build_condition_stimulus_file(
        "decade_10", DECADE_10_VALUES, DECADE_10_BOUNDARY, output_dir
    )
    print(f"  Saved: {f1}")
    
    # --- Condition 2: control_15 ---
    pairs_15 = compute_pairwise_distances(CONTROL_15_VALUES, CONTROL_15_BOUNDARY)
    print_condition_summary("control_15", CONTROL_15_VALUES, CONTROL_15_BOUNDARY, pairs_15)
    f2 = build_condition_stimulus_file(
        "control_15", CONTROL_15_VALUES, CONTROL_15_BOUNDARY, output_dir
    )
    print(f"  Saved: {f2}")
    
    # --- Cross-check: overlapping values ---
    overlap = set(DECADE_10_VALUES) & set(CONTROL_15_VALUES)
    print(f"\nOverlapping values between conditions: {sorted(overlap)}")
    print("  (Values 11-19 appear in both — enables within-value cross-condition comparison)")
    
    # --- Verify theoretical RDM shapes ---
    for cond, vals in [("decade_10", DECADE_10_VALUES), ("control_15", CONTROL_15_VALUES)]:
        rdms = build_all_theoretical_rdms(
            vals, DECADE_10_BOUNDARY if cond == "decade_10" else CONTROL_15_BOUNDARY
        )
        n = len(vals)
        for name, rdm in rdms.items():
            assert rdm.shape == (n, n), f"RDM shape mismatch: {cond}/{name}"
            assert np.allclose(rdm, rdm.T), f"RDM not symmetric: {cond}/{name}"
            assert np.allclose(np.diag(rdm), 0), f"RDM diagonal not zero: {cond}/{name}"
        print(f"  RDM checks passed for {cond} ({n}×{n})")
    
    print(f"\nDone. Stimulus files saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
