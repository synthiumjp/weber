"""
Weber's Law Project 4.2 — Configuration
Classical Minds, Modern Machines

All constants locked to pre-registration v2.7 + v2.8 amendment.
Do not modify after data collection begins.
"""

import os
from pathlib import Path

# ── Project paths ──
PROJECT_ROOT = Path(os.environ.get("WEBER_ROOT", r"C:\weber"))
STIMULI_DIR = PROJECT_ROOT / "stimuli"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ── Models (v2.7 Section 6, v2.8 Amendment 3) ──
MODELS = {
    "llama_instruct": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "commit": "8afb486c1db24fe5",  # truncated; full in config.json
        "role": "primary_1",
        "d_model": 4096,
        "n_layers": 32,  # 32 transformer layers + 1 embedding = 33 total
        "precision": "float16",
    },
    "mistral_instruct": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "commit": "c170c708c41dac92",
        "role": "primary_2",
        "d_model": 4096,
        "n_layers": 32,
        "precision": "float16",
    },
    "llama_base": {
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "commit": "8cde5ca8380496c9",
        "role": "exploratory",
        "d_model": 4096,
        "n_layers": 32,
        "precision": "float16",
    },
}

# ── Numerical magnitude stimuli (v2.7 Section 5.2.1) ──
NUMERICAL_MAGNITUDES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    150, 200, 300, 500, 700, 1000,
]  # 26 values, three orders of magnitude

NUMERICAL_CARRIERS = [
    "The number {N} is a quantity.",
    "There are {N} items.",
    "{N} was the value.",
    "The count reached {N}.",
    "Exactly {N} were measured.",
]  # 5 carrier sentences

# ── Temporal magnitude stimuli (v2.7 Section 5.2.2) ──
TEMPORAL_MAGNITUDES_RAW = [
    "1 second", "2 seconds", "5 seconds", "10 seconds", "30 seconds",
    "1 minute", "2 minutes", "5 minutes", "10 minutes", "30 minutes",
    "1 hour", "2 hours", "5 hours", "12 hours",
    "1 day", "3 days", "1 week", "1 month", "1 year",
]
# Corresponding values in seconds for geometric analysis
TEMPORAL_MAGNITUDES_SECONDS = [
    1, 2, 5, 10, 30,
    60, 120, 300, 600, 1800,
    3600, 7200, 18000, 43200,
    86400, 259200, 604800, 2592000, 31536000,
]

TEMPORAL_CARRIERS = [
    "The event lasted {D}.",
    "It took {D} to complete.",
    "The delay was {D}.",
    "After {D} had passed, the process finished.",
    "The total duration was {D}.",
]

# ── Spatial magnitude stimuli (v2.7 Section 5.2.3) ──
SPATIAL_MAGNITUDES_RAW = [
    "1 metre", "2 metres", "5 metres", "10 metres", "50 metres",
    "100 metres", "500 metres", "1 km", "5 km", "10 km",
    "50 km", "100 km", "500 km", "1000 km",
]
SPATIAL_MAGNITUDES_METRES = [
    1, 2, 5, 10, 50,
    100, 500, 1000, 5000, 10000,
    50000, 100000, 500000, 1000000,
]

SPATIAL_CARRIERS = [
    "The distance was {D}.",
    "They travelled {D} to get there.",
    "It was {D} away.",
    "The gap measured {D}.",
    "The total route covered {D}.",
]

# ── Analysis parameters (v2.7 Sections 5.3, 7, 8) ──
PRIMARY_LAYER_RANGE = (16, 32)  # inclusive start, exclusive end (layers 16-31)
N_LAYERS_TOTAL = 33  # embedding (0) + 32 transformer layers
BONFERRONI_ALPHA = 0.017  # 0.05 / 3 primary hypotheses
SECONDARY_ALPHA = 0.05

# RSA (v2.7 Section 5.3, Step 5)
MANTEL_PERMUTATIONS = 10_000

# Model fitting (v2.7 Section 5.3, Step 4)
STEVENS_BETA_INIT = 0.5
STEVENS_BETA_BOUNDS = (0.01, 2.0)

# Bootstrap (v2.7 Section 7)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42

# H1 success criteria (v2.7 Section 4.1)
H1_MIN_LAYERS = 9   # out of 17 (layers 16-32 inclusive = 17 layers)
H1_MIN_DOMAINS = 2  # out of 3

# H3 success criteria (v2.7 Section 4.1)
H3_MIN_LAYERS = 17  # out of 32 total layers
H3_MIN_DOMAINS = 2

# Model-fit floor (v2.7 Section 4.1, H1 addendum)
MODEL_FIT_FLOOR_R2 = 0.20

# ── Frequency-matched nouns (v2.8 Amendment 2) ──
FREQUENCY_MATCHED_NOUNS = [
    "window", "student", "state", "road", "bird", "word", "study", "water",
    "game", "food", "car", "work", "family", "problem", "building", "world",
    "book", "power", "way", "group", "man", "story", "time", "number", "lot", "life",
]  # 26 nouns, matched to NUMERICAL_MAGNITUDES by Hungarian algorithm

NOUN_CARRIERS = [
    "The word {W} is common.",
    "There is a {W} nearby.",
    "{W} was mentioned.",
    "The {W} appeared again.",
    "Exactly one {W} was found.",
]

# ── Random seeds (locked) ──
SEED_JITTER = 42
SEED_BOOTSTRAP = 42
SEED_PARADIGM_D = 42
SEED_SHUFFLED = 42
SEED_RANDOM_DIRECTIONS = 42

# ── Domain registry ──
DOMAINS = {
    "numerical": {
        "magnitudes_raw": [str(n) for n in NUMERICAL_MAGNITUDES],
        "magnitudes_numeric": NUMERICAL_MAGNITUDES,
        "carriers": NUMERICAL_CARRIERS,
        "placeholder": "{N}",
        "n_magnitudes": 26,
    },
    "temporal": {
        "magnitudes_raw": TEMPORAL_MAGNITUDES_RAW,
        "magnitudes_numeric": TEMPORAL_MAGNITUDES_SECONDS,
        "carriers": TEMPORAL_CARRIERS,
        "placeholder": "{D}",
        "n_magnitudes": 19,
    },
    "spatial": {
        "magnitudes_raw": SPATIAL_MAGNITUDES_RAW,
        "magnitudes_numeric": SPATIAL_MAGNITUDES_METRES,
        "carriers": SPATIAL_CARRIERS,
        "placeholder": "{D}",
        "n_magnitudes": 14,
    },
}
