#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — Stimulus Generation
Generates all stimuli for Paradigms A, B, C, D and archives with checksums.

Pre-registered specifications from v2.6:
- Seed 42 for all randomisation
- Jittered baselines: 30 of 50 pairs per cell (numerical), ±15%
- Counterbalanced presentation order
- Paradigm D: 200 prompts, 40 per baseline, seed 42

Author: JP Cacioli
Programme: Classical Minds, Modern Machines — Project 4.2
"""

import json
import hashlib
import os
import random
import math
from pathlib import Path
from datetime import datetime

# ── Fixed seed ──
SEED = 42
random.seed(SEED)

# ── Output directory ──
OUT_DIR = Path("stimuli")
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════
# 1. PROBING STIMULI (Paradigm A)
# ═══════════════════════════════════════════════════

# 1a. Numerical magnitude — 26 values
NUMERICAL_MAGNITUDES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    150, 200, 300, 500, 700, 1000
]

NUMERICAL_CARRIERS = [
    "The number {N} is a quantity.",
    "There are {N} items.",
    "{N} was the value.",
    "The count reached {N}.",
    "Exactly {N} were measured.",
]

# 1b. Temporal duration — ~7 orders of magnitude
TEMPORAL_MAGNITUDES = [
    "1 second", "2 seconds", "5 seconds", "10 seconds", "30 seconds",
    "1 minute", "2 minutes", "5 minutes", "10 minutes", "30 minutes",
    "1 hour", "2 hours", "5 hours", "12 hours",
    "1 day", "3 days", "1 week", "1 month", "1 year"
]

# Magnitude in seconds (for geometric model fitting)
TEMPORAL_SECONDS = [
    1, 2, 5, 10, 30,
    60, 120, 300, 600, 1800,
    3600, 7200, 18000, 43200,
    86400, 259200, 604800, 2592000, 31536000
]

TEMPORAL_CARRIERS = [
    "The event lasted {D}.",
    "It took {D} to complete.",
    "The delay was {D}.",
    "After {D} had passed, the process finished.",
    "The total duration was {D}.",
]

# 1c. Spatial distance — ~6 orders of magnitude
SPATIAL_MAGNITUDES = [
    "1 metre", "2 metres", "5 metres", "10 metres", "50 metres",
    "100 metres", "500 metres", "1 km", "5 km", "10 km",
    "50 km", "100 km", "500 km", "1000 km"
]

# Magnitude in metres
SPATIAL_METRES = [
    1, 2, 5, 10, 50,
    100, 500, 1000, 5000, 10000,
    50000, 100000, 500000, 1000000
]

SPATIAL_CARRIERS = [
    "The distance was {D}.",
    "They travelled {D} to get there.",
    "It was {D} away.",
    "The gap measured {D}.",
    "The total route covered {D}.",
]


def generate_probing_stimuli():
    """Generate all Paradigm A probing sentences."""
    probing = {"numerical": [], "temporal": [], "spatial": []}

    # Numerical
    for mag in NUMERICAL_MAGNITUDES:
        for carrier in NUMERICAL_CARRIERS:
            sentence = carrier.format(N=mag)
            probing["numerical"].append({
                "magnitude": mag,
                "magnitude_log": round(math.log(mag), 6) if mag > 0 else 0,
                "carrier_template": carrier,
                "sentence": sentence,
                "domain": "numerical",
            })

    # Temporal
    for mag_str, mag_sec in zip(TEMPORAL_MAGNITUDES, TEMPORAL_SECONDS):
        for carrier in TEMPORAL_CARRIERS:
            sentence = carrier.format(D=mag_str)
            probing["temporal"].append({
                "magnitude_text": mag_str,
                "magnitude_seconds": mag_sec,
                "magnitude_log": round(math.log(mag_sec), 6),
                "carrier_template": carrier,
                "sentence": sentence,
                "domain": "temporal",
            })

    # Spatial
    for mag_str, mag_m in zip(SPATIAL_MAGNITUDES, SPATIAL_METRES):
        for carrier in SPATIAL_CARRIERS:
            sentence = carrier.format(D=mag_str)
            probing["spatial"].append({
                "magnitude_text": mag_str,
                "magnitude_metres": mag_m,
                "magnitude_log": round(math.log(mag_m), 6),
                "carrier_template": carrier,
                "sentence": sentence,
                "domain": "spatial",
            })

    # Counts
    n_num = len(probing["numerical"])
    n_tmp = len(probing["temporal"])
    n_spa = len(probing["spatial"])
    print(f"Probing stimuli: {n_num} numerical, {n_tmp} temporal, {n_spa} spatial")
    print(f"  Total: {n_num + n_tmp + n_spa}")
    assert n_num == 26 * 5  # 130
    assert n_tmp == 19 * 5  # 95
    assert n_spa == 14 * 5  # 70

    return probing


# ═══════════════════════════════════════════════════
# 2. COMPARISON STIMULI (Paradigm B)
# ═══════════════════════════════════════════════════

def generate_numerical_comparisons():
    """
    Numerical: 5 baselines × 6 ratios × 50 pairs = 1,500 pairs.
    30 of 50 use jittered baselines (±15%), 20 use exact baselines.
    Counterbalanced order.
    """
    baselines = [10, 30, 100, 300, 1000]
    ratios = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
    pairs_per_cell = 50
    jittered_per_cell = 30
    exact_per_cell = 20

    random.seed(SEED)
    pairs = []
    pair_id = 0

    for baseline in baselines:
        for ratio in ratios:
            # Generate jittered baselines
            jittered_bases = []
            for _ in range(jittered_per_cell):
                jb = random.uniform(0.85 * baseline, 1.15 * baseline)
                jb = round(jb)
                jb = max(1, jb)  # floor at 1
                jittered_bases.append(jb)

            # Generate exact baselines
            exact_bases = [baseline] * exact_per_cell

            all_bases = jittered_bases + exact_bases
            assert len(all_bases) == pairs_per_cell

            for i, b in enumerate(all_bases):
                comparison = round(b * ratio)
                comparison = max(b + 1, comparison)  # ensure distinct

                # Counterbalance: first half present baseline first
                if i % 2 == 0:
                    first, second = b, comparison
                    correct = "B"
                else:
                    first, second = comparison, b
                    correct = "A"

                pairs.append({
                    "pair_id": f"NUM-{pair_id:04d}",
                    "domain": "numerical",
                    "nominal_baseline": baseline,
                    "actual_baseline": b,
                    "comparison_value": comparison,
                    "ratio": round(comparison / b, 4),
                    "nominal_ratio": ratio,
                    "absolute_difference": abs(comparison - b),
                    "first_presented": first,
                    "second_presented": second,
                    "correct_answer": correct,
                    "is_jittered": i < jittered_per_cell,
                    "pair_index_in_cell": i,
                })
                pair_id += 1

    assert len(pairs) == 1500
    print(f"Numerical comparison pairs: {len(pairs)}")
    return pairs


def generate_temporal_comparisons():
    """
    Temporal: 5 baselines × 6 ratios × 30 pairs = 900 pairs.
    Baselines: 10 seconds, 2 minutes, 15 minutes, 2 hours, 1 day
    Ratios: 1.10, 1.20, 1.50, 2.00, 3.00, 5.00
    """
    baselines = [
        (10, "seconds"),
        (120, "minutes"),
        (900, "minutes"),
        (7200, "hours"),
        (86400, "days"),
    ]
    baseline_labels = ["10 seconds", "2 minutes", "15 minutes", "2 hours", "1 day"]
    ratios = [1.10, 1.20, 1.50, 2.00, 3.00, 5.00]
    pairs_per_cell = 30

    def seconds_to_text(secs):
        """Convert seconds to natural language temporal expression."""
        if secs < 60:
            s = round(secs, 1)
            if s == int(s):
                s = int(s)
            return f"{s} second{'s' if s != 1 else ''}"
        elif secs < 3600:
            m = round(secs / 60, 1)
            if m == int(m):
                m = int(m)
            return f"{m} minute{'s' if m != 1 else ''}"
        elif secs < 86400:
            h = round(secs / 3600, 1)
            if h == int(h):
                h = int(h)
            return f"{h} hour{'s' if h != 1 else ''}"
        else:
            d = round(secs / 86400, 1)
            if d == int(d):
                d = int(d)
            return f"{d} day{'s' if d != 1 else ''}"

    random.seed(SEED + 1)  # different stream from numerical
    pairs = []
    pair_id = 0

    for (base_secs, base_unit), base_label in zip(baselines, baseline_labels):
        for ratio in ratios:
            for i in range(pairs_per_cell):
                # Jitter baseline ±15%
                jb = random.uniform(0.85 * base_secs, 1.15 * base_secs)
                comp_secs = jb * ratio

                base_text = seconds_to_text(jb)
                comp_text = seconds_to_text(comp_secs)

                if i % 2 == 0:
                    first, second = base_text, comp_text
                    correct = "B"
                else:
                    first, second = comp_text, base_text
                    correct = "A"

                pairs.append({
                    "pair_id": f"TMP-{pair_id:04d}",
                    "domain": "temporal",
                    "nominal_baseline": base_label,
                    "baseline_seconds": round(jb, 2),
                    "comparison_seconds": round(comp_secs, 2),
                    "first_presented": first,
                    "second_presented": second,
                    "correct_answer": correct,
                    "nominal_ratio": ratio,
                    "actual_ratio": round(comp_secs / jb, 4),
                    "pair_index_in_cell": i,
                })
                pair_id += 1

    assert len(pairs) == 900
    print(f"Temporal comparison pairs: {len(pairs)}")
    return pairs


def generate_spatial_comparisons():
    """
    Spatial: 5 baselines × 6 ratios × 30 pairs = 900 pairs.
    Baselines: 10 metres, 200 metres, 5 km, 50 km, 500 km
    Ratios: 1.05, 1.10, 1.20, 1.50, 2.00, 3.00
    """
    baselines = [
        (10, "metres"),
        (200, "metres"),
        (5000, "metres"),
        (50000, "metres"),
        (500000, "metres"),
    ]
    baseline_labels = ["10 metres", "200 metres", "5 km", "50 km", "500 km"]
    ratios = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
    pairs_per_cell = 30

    def metres_to_text(m):
        """Convert metres to natural language distance expression."""
        if m < 1000:
            v = round(m, 1)
            if v == int(v):
                v = int(v)
            return f"{v} metre{'s' if v != 1 else ''}"
        else:
            km = round(m / 1000, 1)
            if km == int(km):
                km = int(km)
            return f"{km} km"

    random.seed(SEED + 2)
    pairs = []
    pair_id = 0

    for (base_m, base_unit), base_label in zip(baselines, baseline_labels):
        for ratio in ratios:
            for i in range(pairs_per_cell):
                jb = random.uniform(0.85 * base_m, 1.15 * base_m)
                comp_m = jb * ratio

                base_text = metres_to_text(jb)
                comp_text = metres_to_text(comp_m)

                if i % 2 == 0:
                    first, second = base_text, comp_text
                    correct = "B"
                else:
                    first, second = comp_text, base_text
                    correct = "A"

                pairs.append({
                    "pair_id": f"SPA-{pair_id:04d}",
                    "domain": "spatial",
                    "nominal_baseline": base_label,
                    "baseline_metres": round(jb, 2),
                    "comparison_metres": round(comp_m, 2),
                    "first_presented": first,
                    "second_presented": second,
                    "correct_answer": correct,
                    "nominal_ratio": ratio,
                    "actual_ratio": round(comp_m / jb, 4),
                    "pair_index_in_cell": i,
                })
                pair_id += 1

    assert len(pairs) == 900
    print(f"Spatial comparison pairs: {len(pairs)}")
    return pairs


# ═══════════════════════════════════════════════════
# 3. PARADIGM B PROMPT TEMPLATES
# ═══════════════════════════════════════════════════

# Number word forms for Task B1 cross-format comparison
NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty", 21: "twenty-one", 22: "twenty-two",
    23: "twenty-three", 24: "twenty-four", 25: "twenty-five",
    26: "twenty-six", 27: "twenty-seven", 28: "twenty-eight",
    29: "twenty-nine", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    100: "one hundred", 200: "two hundred", 300: "three hundred",
    500: "five hundred", 1000: "one thousand",
}


def number_to_expression(n, format_type):
    """
    Convert a number to a cross-format expression for Task B1.
    format_type: 'word', 'fraction', 'multiplication', 'percentage'
    """
    if format_type == "word":
        # Direct word form or approximate
        if n in NUMBER_WORDS:
            return NUMBER_WORDS[n]
        # Compose: e.g., 120 -> "one hundred and twenty"
        if n < 100:
            tens = (n // 10) * 10
            ones = n % 10
            if tens in NUMBER_WORDS and ones > 0:
                return f"{NUMBER_WORDS[tens]}-{NUMBER_WORDS[ones]}"
            elif tens in NUMBER_WORDS:
                return NUMBER_WORDS[tens]
        return str(n)  # fallback to digits

    elif format_type == "multiplication":
        # Find a clean factorisation
        for factor in [12, 10, 8, 7, 6, 5, 4, 3, 2]:
            if n % factor == 0 and n // factor >= 2:
                q = n // factor
                if q in NUMBER_WORDS:
                    return f"{NUMBER_WORDS[q]} times {NUMBER_WORDS.get(factor, str(factor))}"
        return str(n)

    elif format_type == "fraction":
        # Express as fraction of a round number
        for denom in [100, 50, 20, 10]:
            if denom > n:
                ratio_val = n / denom
                pct = round(ratio_val * 100)
                return f"{pct}% of {denom}"
        for mult in [2, 3, 4, 5]:
            bigger = n * mult
            return f"{n * mult} divided by {mult}"
        return str(n)

    return str(n)


def generate_b1_prompts(numerical_pairs):
    """
    Task B1 — Cross-format comparison (PRIMARY for H2).
    One value in digits, one in words/expressions.
    """
    random.seed(SEED + 10)
    prompts = []

    formats = ["word", "multiplication", "fraction"]

    for pair in numerical_pairs:
        first = pair["first_presented"]
        second = pair["second_presented"]

        # Randomly assign which gets transformed
        fmt = random.choice(formats)
        if random.random() < 0.5:
            a_expr = number_to_expression(first, fmt)
            b_expr = str(second)
        else:
            a_expr = str(first)
            b_expr = number_to_expression(second, fmt)

        prompt = (f"Which represents a larger quantity: {a_expr} or {b_expr}? "
                  f"Answer with only A or B.")

        prompts.append({
            "pair_id": pair["pair_id"],
            "task": "B1",
            "prompt": prompt,
            "a_expression": a_expr,
            "b_expression": b_expr,
            "correct_answer": pair["correct_answer"],
            "nominal_ratio": pair["nominal_ratio"],
            "nominal_baseline": pair["nominal_baseline"],
        })

    print(f"B1 prompts: {len(prompts)}")
    return prompts


def generate_b2_prompts(numerical_pairs):
    """
    Task B2 — Approximate arithmetic comparison.
    'Without calculating exactly, which is larger: [A] or [B]?'
    """
    random.seed(SEED + 20)
    prompts = []

    for pair in numerical_pairs:
        first = pair["first_presented"]
        second = pair["second_presented"]

        # Generate approximate arithmetic expressions
        # Strategy: express as percentage of a larger number
        pct_a = random.randint(30, 70)
        base_a = round(first / (pct_a / 100))
        expr_a = f"{pct_a}% of {base_a}"

        pct_b = random.randint(30, 70)
        base_b = round(second / (pct_b / 100))
        expr_b = f"{pct_b}% of {base_b}"

        prompt = (f"Without calculating exactly, which is larger: {expr_a} or "
                  f"{expr_b}? Answer with only A or B.")

        prompts.append({
            "pair_id": pair["pair_id"],
            "task": "B2",
            "prompt": prompt,
            "a_expression": expr_a,
            "b_expression": expr_b,
            "a_actual_value": first,
            "b_actual_value": second,
            "correct_answer": pair["correct_answer"],
            "nominal_ratio": pair["nominal_ratio"],
            "nominal_baseline": pair["nominal_baseline"],
        })

    print(f"B2 prompts: {len(prompts)}")
    return prompts


def generate_b3_prompts(numerical_pairs):
    """
    Task B3 — Contextual comparison.
    '[Context]. Which was more? Answer with only A or B.'
    """
    random.seed(SEED + 30)

    contexts = [
        "About {a} people attended on Monday (A). About {b} attended on Tuesday (B).",
        "Team A scored about {a} points. Team B scored about {b} points.",
        "Store A sold roughly {a} units (A). Store B sold roughly {b} units (B).",
        "City A has approximately {a} parks (A). City B has approximately {b} parks (B).",
        "Project A took about {a} hours (A). Project B took about {b} hours (B).",
    ]

    prompts = []
    for pair in numerical_pairs:
        first = pair["first_presented"]
        second = pair["second_presented"]

        ctx_template = random.choice(contexts)
        context = ctx_template.format(a=first, b=second)

        prompt = f"{context} Which was more? Answer with only A or B."

        prompts.append({
            "pair_id": pair["pair_id"],
            "task": "B3",
            "prompt": prompt,
            "context": context,
            "a_value": first,
            "b_value": second,
            "correct_answer": pair["correct_answer"],
            "nominal_ratio": pair["nominal_ratio"],
            "nominal_baseline": pair["nominal_baseline"],
        })

    print(f"B3 prompts: {len(prompts)}")
    return prompts


def generate_symbolic_control(numerical_pairs):
    """
    Symbolic control: 'Which is larger, [A] or [B]? Answer with only the larger number.'
    """
    prompts = []
    for pair in numerical_pairs:
        first = pair["first_presented"]
        second = pair["second_presented"]
        larger = max(first, second)

        prompt = f"Which is larger, {first} or {second}? Answer with only the larger number."

        prompts.append({
            "pair_id": pair["pair_id"],
            "task": "symbolic_control",
            "prompt": prompt,
            "first_presented": first,
            "second_presented": second,
            "correct_answer": str(larger),
            "nominal_ratio": pair["nominal_ratio"],
            "nominal_baseline": pair["nominal_baseline"],
        })

    print(f"Symbolic control prompts: {len(prompts)}")
    return prompts


# ═══════════════════════════════════════════════════
# 4. PARADIGM D — SUBSET SELECTION
# ═══════════════════════════════════════════════════

def select_paradigm_d_prompts(numerical_pairs):
    """
    Select 200 prompts for Paradigm D: 40 per baseline, seed 42.
    Uses symbolic comparison format.
    """
    random.seed(SEED)

    baselines = [10, 30, 100, 300, 1000]
    selected = []

    for baseline in baselines:
        pool = [p for p in numerical_pairs if p["nominal_baseline"] == baseline]
        chosen = random.sample(pool, 40)
        for pair in chosen:
            first = pair["first_presented"]
            second = pair["second_presented"]
            larger = max(first, second)

            selected.append({
                "paradigm_d_id": f"PD-{len(selected):03d}",
                "source_pair_id": pair["pair_id"],
                "prompt": f"Which is larger, {first} or {second}? Answer with only the larger number.",
                "first_presented": first,
                "second_presented": second,
                "correct_answer": str(larger),
                "nominal_baseline": baseline,
                "nominal_ratio": pair["nominal_ratio"],
            })

    assert len(selected) == 200
    print(f"Paradigm D prompts: {len(selected)}")
    return selected


# ═══════════════════════════════════════════════════
# 5. DIGIT-BOUNDARY DIAGNOSTIC PAIRS
# ═══════════════════════════════════════════════════

def generate_digit_boundary_pairs():
    """
    Generate within-digit, cross-digit, and matched-ratio control pairs
    for the digit-boundary diagnostic (Section 5.7.1).
    """
    pairs = []

    # Cross-digit pairs (crossing 1→2 digit boundary)
    cross_digit_1to2 = [
        (8, 12), (9, 11), (7, 13), (8, 11), (9, 12),
        (7, 11), (8, 13), (9, 13), (7, 12), (8, 10),
    ]

    # Cross-digit pairs (crossing 2→3 digit boundary)
    cross_digit_2to3 = [
        (90, 110), (95, 105), (92, 108), (88, 112), (98, 102),
        (85, 115), (91, 109), (93, 107), (96, 104), (99, 101),
    ]

    # For each cross-digit pair, compute ratio and find a within-digit match
    for cd_pairs, boundary in [(cross_digit_1to2, "1to2"), (cross_digit_2to3, "2to3")]:
        for a, b in cd_pairs:
            ratio = round(b / a, 4)

            # Within-digit matched pair (same ratio, no boundary crossing)
            if boundary == "1to2":
                # Use 2-digit numbers
                wd_a = round(a * 2)  # shift into 2-digit range
                wd_b = round(wd_a * ratio)
            else:
                # Use 2-digit or 3-digit numbers that don't cross boundary
                wd_a = round(a * 0.6)  # stay in 2-digit range
                wd_b = round(wd_a * ratio)

            pairs.append({
                "type": "cross_digit",
                "boundary": boundary,
                "a": a,
                "b": b,
                "ratio": ratio,
                "digits_a": len(str(a)),
                "digits_b": len(str(b)),
            })
            pairs.append({
                "type": "within_digit",
                "boundary": boundary,
                "a": wd_a,
                "b": wd_b,
                "ratio": round(wd_b / wd_a, 4),
                "digits_a": len(str(wd_a)),
                "digits_b": len(str(wd_b)),
                "matched_to_cross": (a, b),
            })

    print(f"Digit-boundary diagnostic pairs: {len(pairs)}")
    return pairs


# ═══════════════════════════════════════════════════
# 6. UNIT-BOUNDARY MANIPULATION CHECK
# ═══════════════════════════════════════════════════

def generate_unit_boundary_check():
    """
    Equivalent magnitude pairs in different units for temporal and spatial.
    Section 5.7.3.
    """
    temporal_equivalents = [
        ("60 seconds", "1 minute"),
        ("120 seconds", "2 minutes"),
        ("300 seconds", "5 minutes"),
        ("3600 seconds", "1 hour"),
        ("7200 seconds", "2 hours"),
        ("86400 seconds", "1 day"),
    ]

    spatial_equivalents = [
        ("1000 metres", "1 km"),
        ("5000 metres", "5 km"),
        ("10000 metres", "10 km"),
        ("50000 metres", "50 km"),
        ("100000 metres", "100 km"),
    ]

    check = {
        "temporal": temporal_equivalents,
        "spatial": spatial_equivalents,
    }
    print(f"Unit-boundary check: {len(temporal_equivalents)} temporal, "
          f"{len(spatial_equivalents)} spatial")
    return check


# ═══════════════════════════════════════════════════
# 7. SHUFFLED-MAGNITUDE SANITY CHECK
# ═══════════════════════════════════════════════════

def generate_shuffled_magnitudes():
    """
    Randomly reassign magnitudes to carrier sentences (seed 42).
    Section 5.7.6.
    """
    random.seed(SEED)

    shuffled = {}
    for domain, mags, carriers in [
        ("numerical", NUMERICAL_MAGNITUDES, NUMERICAL_CARRIERS),
        ("temporal", TEMPORAL_MAGNITUDES, TEMPORAL_CARRIERS),
        ("spatial", SPATIAL_MAGNITUDES, SPATIAL_CARRIERS),
    ]:
        # For each carrier, shuffle which magnitude goes where
        shuffled_mags = list(mags)
        random.shuffle(shuffled_mags)

        shuffled[domain] = {
            "original_order": list(mags),
            "shuffled_order": shuffled_mags,
            "mapping": {str(orig): str(shuf) for orig, shuf in zip(mags, shuffled_mags)},
        }

    print("Shuffled-magnitude assignments generated")
    return shuffled


# ═══════════════════════════════════════════════════
# 8. RANDOM DIRECTIONS FOR PARADIGM D CONTROL
# ═══════════════════════════════════════════════════

def generate_random_directions(d_model=4096, n_directions=10):
    """
    10 random unit vectors in d_model-dimensional space.
    Seed 42, standard normal, normalised.
    """
    import struct

    random.seed(SEED)
    directions = []
    for i in range(n_directions):
        # Generate d_model random normal values using Box-Muller
        vec = []
        for _ in range(d_model):
            # Use random.gauss for standard normal
            vec.append(random.gauss(0, 1))
        # Normalise to unit length
        norm = math.sqrt(sum(v * v for v in vec))
        vec = [v / norm for v in vec]
        directions.append(vec)

    # Save as compact format (just seeds and verification hashes)
    verification = []
    for i, vec in enumerate(directions):
        # Store first 10 values and norm for verification
        verification.append({
            "direction_index": i,
            "first_10_values": [round(v, 8) for v in vec[:10]],
            "norm": round(math.sqrt(sum(v * v for v in vec)), 10),
            "d_model": d_model,
        })

    print(f"Random directions: {n_directions} vectors in R^{d_model}")
    return verification


# ═══════════════════════════════════════════════════
# 9. WORD-FORM ROBUSTNESS (26 probing magnitudes)
# ═══════════════════════════════════════════════════

WORD_FORMS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    15: "fifteen", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    100: "one hundred", 150: "one hundred and fifty",
    200: "two hundred", 300: "three hundred", 500: "five hundred",
    700: "seven hundred", 1000: "one thousand",
}

def generate_word_form_probing():
    """Secondary robustness: word-form numbers in carrier sentences."""
    probing = []
    for mag in NUMERICAL_MAGNITUDES:
        word = WORD_FORMS.get(mag, str(mag))
        for carrier in NUMERICAL_CARRIERS:
            sentence = carrier.format(N=word)
            probing.append({
                "magnitude": mag,
                "word_form": word,
                "carrier_template": carrier,
                "sentence": sentence,
            })
    print(f"Word-form probing stimuli: {len(probing)}")
    return probing


# ═══════════════════════════════════════════════════
# MAIN: Generate everything, save, checksum
# ═══════════════════════════════════════════════════

def compute_md5(filepath):
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256(filepath):
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(data, filename):
    """Save data as JSON and return path."""
    path = OUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def main():
    print("=" * 60)
    print("Weber's Law Project 4.2 — Stimulus Generation")
    print(f"Seed: {SEED}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    print()

    # Generate all stimuli
    print("── Paradigm A: Probing Stimuli ──")
    probing = generate_probing_stimuli()

    print("\n── Paradigm B: Comparison Pairs ──")
    num_pairs = generate_numerical_comparisons()
    tmp_pairs = generate_temporal_comparisons()
    spa_pairs = generate_spatial_comparisons()

    print("\n── Paradigm B: Prompt Templates ──")
    b1_prompts = generate_b1_prompts(num_pairs)
    b2_prompts = generate_b2_prompts(num_pairs)
    b3_prompts = generate_b3_prompts(num_pairs)
    sym_control = generate_symbolic_control(num_pairs)

    print("\n── Paradigm D: Subset Selection ──")
    pd_prompts = select_paradigm_d_prompts(num_pairs)

    print("\n── Robustness Checks ──")
    digit_pairs = generate_digit_boundary_pairs()
    unit_check = generate_unit_boundary_check()
    shuffled = generate_shuffled_magnitudes()
    random_dirs = generate_random_directions()
    word_forms = generate_word_form_probing()

    # Save all files
    print("\n── Saving Files ──")
    files = {}

    files["probing_numerical.json"] = save_json(probing["numerical"], "probing_numerical.json")
    files["probing_temporal.json"] = save_json(probing["temporal"], "probing_temporal.json")
    files["probing_spatial.json"] = save_json(probing["spatial"], "probing_spatial.json")

    files["comparison_numerical.json"] = save_json(num_pairs, "comparison_numerical.json")
    files["comparison_temporal.json"] = save_json(tmp_pairs, "comparison_temporal.json")
    files["comparison_spatial.json"] = save_json(spa_pairs, "comparison_spatial.json")

    files["prompts_b1.json"] = save_json(b1_prompts, "prompts_b1.json")
    files["prompts_b2.json"] = save_json(b2_prompts, "prompts_b2.json")
    files["prompts_b3.json"] = save_json(b3_prompts, "prompts_b3.json")
    files["prompts_symbolic_control.json"] = save_json(sym_control, "prompts_symbolic_control.json")

    files["paradigm_d_prompts.json"] = save_json(pd_prompts, "paradigm_d_prompts.json")

    files["digit_boundary_pairs.json"] = save_json(digit_pairs, "digit_boundary_pairs.json")
    files["unit_boundary_check.json"] = save_json(unit_check, "unit_boundary_check.json")
    files["shuffled_magnitudes.json"] = save_json(shuffled, "shuffled_magnitudes.json")
    files["random_directions_verification.json"] = save_json(random_dirs, "random_directions_verification.json")
    files["word_form_probing.json"] = save_json(word_forms, "word_form_probing.json")

    # Save metadata with specifications
    metadata = {
        "project": "Classical Minds, Modern Machines — Project 4.2",
        "version": "v2.6",
        "generated": datetime.now().isoformat(),
        "seed": SEED,
        "specifications": {
            "numerical_magnitudes": NUMERICAL_MAGNITUDES,
            "numerical_baselines": [10, 30, 100, 300, 1000],
            "numerical_ratios": [1.05, 1.10, 1.20, 1.50, 2.00, 3.00],
            "numerical_pairs_per_cell": 50,
            "numerical_jittered_per_cell": 30,
            "temporal_baselines": ["10 seconds", "2 minutes", "15 minutes", "2 hours", "1 day"],
            "temporal_ratios": [1.10, 1.20, 1.50, 2.00, 3.00, 5.00],
            "temporal_pairs_per_cell": 30,
            "spatial_baselines": ["10 metres", "200 metres", "5 km", "50 km", "500 km"],
            "spatial_ratios": [1.05, 1.10, 1.20, 1.50, 2.00, 3.00],
            "spatial_pairs_per_cell": 30,
            "paradigm_d_total": 200,
            "paradigm_d_per_baseline": 40,
        },
        "counts": {
            "probing_numerical": len(probing["numerical"]),
            "probing_temporal": len(probing["temporal"]),
            "probing_spatial": len(probing["spatial"]),
            "comparison_numerical": len(num_pairs),
            "comparison_temporal": len(tmp_pairs),
            "comparison_spatial": len(spa_pairs),
            "prompts_b1": len(b1_prompts),
            "prompts_b2": len(b2_prompts),
            "prompts_b3": len(b3_prompts),
            "prompts_symbolic_control": len(sym_control),
            "paradigm_d": len(pd_prompts),
            "digit_boundary": len(digit_pairs),
            "word_form_probing": len(word_forms),
        },
    }
    files["metadata.json"] = save_json(metadata, "metadata.json")

    # Compute checksums
    print("\n── Checksums ──")
    checksums = {}
    for name, path in sorted(files.items()):
        md5 = compute_md5(path)
        sha256 = compute_sha256(path)
        checksums[name] = {"md5": md5, "sha256": sha256, "path": str(path)}
        print(f"  {name}: MD5={md5[:12]}... SHA256={sha256[:12]}...")

    save_json(checksums, "CHECKSUMS.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_probing = sum(len(v) for v in probing.values())
    total_comparison = len(num_pairs) + len(tmp_pairs) + len(spa_pairs)
    total_prompts = len(b1_prompts) + len(b2_prompts) + len(b3_prompts) + len(sym_control)
    print(f"  Probing sentences:    {total_probing}")
    print(f"  Comparison pairs:     {total_comparison}")
    print(f"  B1+B2+B3+Control:     {total_prompts}")
    print(f"  Paradigm D prompts:   {len(pd_prompts)}")
    print(f"  Diagnostic pairs:     {len(digit_pairs)}")
    print(f"  Files generated:      {len(files) + 1}")  # +1 for checksums
    print(f"  Output directory:     {OUT_DIR.absolute()}")
    print()
    print("All stimulus files archived. Ready for OSF upload.")


if __name__ == "__main__":
    main()
