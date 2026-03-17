#!/usr/bin/env python3
"""
Paradigm D: Causal Intervention via Activation Patching
Weber's Law in Transformer Magnitude Representations (Project 4.2)

Pre-registration reference: v2.7 Section 5.6 + v2.2→v2.3 Change 5
Hypothesis: H7 (Functional Relevance)

Steps:
  1. Go/no-go gate: Train ridge probe on Paradigm A hidden states → predict log(magnitude).
     Gate passes if any layer R² > 0.50.
  2. Identify patching layer L: highest probe R². Ties within 1% → earliest.
  3. Activation patching: 200 comparison prompts, 4 dose levels, magnitude direction.
  4. Random direction control: 10 random unit vectors, same procedure.
  5. PCA backup direction: PC1 of 26 magnitudes at layer L.
  6. Evaluate H7: mean |Δp| under mag direction > 97.5th percentile of random, for ≥75% of prompts.

Author: JP Cacioli
Date: March 2026
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Must match config.py
MAGNITUDES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50,
              60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1000]

DOSE_LEVELS = [0.25, 0.50, 0.75, 1.00]

N_RANDOM_DIRECTIONS = 10
RANDOM_SEED = 42

# H7 criterion
H7_PERCENTILE = 97.5  # one-tailed α = 0.025
H7_PROMPT_THRESHOLD = 0.75  # ≥75% of 200 prompts

# Go/no-go gate
GATE_R2_THRESHOLD = 0.50

# PCA backup
PCA_CORR_THRESHOLD = 0.80

# Primary layers for go/no-go (layers 16-32 for 33-layer model, 0-indexed)
# Adjusted per model: Llama has 33 layers (0-32), Mistral has 33 layers (0-32)

MODELS = {
    "llama": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "n_layers": 33,
        "primary_layers": list(range(16, 33)),  # layers 16-32
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_layers": 33,
        "primary_layers": list(range(16, 33)),
    },
}

# Chat templates for instruct models (matching Paradigm B deviation)
LLAMA_CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

MISTRAL_CHAT_TEMPLATE = (
    "<s>[INST] {prompt} [/INST]"
)


def get_chat_template(model_key):
    if model_key == "llama":
        return LLAMA_CHAT_TEMPLATE
    elif model_key == "mistral":
        return MISTRAL_CHAT_TEMPLATE
    else:
        raise ValueError(f"Unknown model key: {model_key}")


# ---------------------------------------------------------------------------
# Step 0: Load Paradigm A data (hidden states + centroids)
# ---------------------------------------------------------------------------

def load_paradigm_a_centroids(results_dir, model_key, domain="numerical"):
    """Load centroid hidden states from Paradigm A extraction.
    
    Actual path: {results_dir}/paradigm_a/{model_key}_instruct/{domain}/hidden_states.npz
    Contains: 'centroids' (26, 33, 4096), 'per_carrier' (26, 5, 33, 4096), 'icc_per_layer' (33,)
    """
    hs_path = Path(results_dir) / "paradigm_a" / f"{model_key}_instruct" / domain / "hidden_states.npz"
    if not hs_path.exists():
        raise FileNotFoundError(
            f"Paradigm A hidden states not found: {hs_path}\n"
            f"Run paradigm_a_extract.py first."
        )
    
    data = np.load(str(hs_path), allow_pickle=True)
    
    # Expected keys: 'centroids' (n_mags × n_layers × d_model)
    # or 'hidden_states' (n_mags × n_carriers × n_layers × d_model) 
    if 'centroids' in data:
        centroids = data['centroids']  # (n_mags, n_layers, d_model)
    elif 'hidden_states' in data:
        # Average across carriers to get centroids
        hs = data['hidden_states']  # (n_mags, n_carriers, n_layers, d_model)
        centroids = hs.mean(axis=1)  # (n_mags, n_layers, d_model)
    else:
        raise KeyError(
            f"Expected 'centroids' or 'hidden_states' in {hs_path}. "
            f"Found keys: {list(data.keys())}"
        )
    
    print(f"  Loaded centroids: shape {centroids.shape}")
    print(f"  Magnitudes: {len(MAGNITUDES)}, Layers: {centroids.shape[1]}, d_model: {centroids.shape[2]}")
    
    return centroids


# ---------------------------------------------------------------------------
# Step 1: Go/No-Go Gate — Linear Probe
# ---------------------------------------------------------------------------

def train_probes(centroids, alpha=1.0):
    """Train ridge regression probes at each layer to predict log(magnitude).
    
    Args:
        centroids: (n_mags, n_layers, d_model) array
        alpha: ridge regularisation parameter (default 1.0, per pre-reg)
        
    Returns:
        probe_r2: (n_layers,) array of R² values
        probe_weights: (n_layers, d_model) array of weight vectors
        probe_intercepts: (n_layers,) array of intercepts
    """
    n_mags, n_layers, d_model = centroids.shape
    log_mags = np.log(np.array(MAGNITUDES, dtype=np.float64))
    
    probe_r2 = np.zeros(n_layers)
    probe_weights = np.zeros((n_layers, d_model))
    probe_intercepts = np.zeros(n_layers)
    
    for layer in range(n_layers):
        X = centroids[:, layer, :]  # (n_mags, d_model)
        y = log_mags  # (n_mags,)
        
        probe = Ridge(alpha=alpha)
        probe.fit(X, y)
        
        probe_r2[layer] = probe.score(X, y)
        probe_weights[layer] = probe.coef_
        probe_intercepts[layer] = probe.intercept_
    
    return probe_r2, probe_weights, probe_intercepts


def train_probes_cv(centroids):
    """Train ridge probes with LOO-CV α selection (robustness check).
    
    With n=26, p=4096, the primary α=1.0 probe overfits (R²→1.0).
    This robustness check selects α via leave-one-out cross-validation
    to find a direction that generalises, not just memorises.
    
    Returns same format as train_probes, plus:
        cv_alphas: (n_layers,) array of selected α values
        cv_r2: (n_layers,) array of LOO-CV R² values
    """
    from sklearn.linear_model import RidgeCV
    
    n_mags, n_layers, d_model = centroids.shape
    log_mags = np.log(np.array(MAGNITUDES, dtype=np.float64))
    
    # Search over a wide range — with p>>n, optimal α is typically large
    alphas = np.logspace(-2, 6, 50)
    
    probe_r2 = np.zeros(n_layers)
    probe_weights = np.zeros((n_layers, d_model))
    probe_intercepts = np.zeros(n_layers)
    cv_alphas = np.zeros(n_layers)
    cv_r2 = np.zeros(n_layers)
    
    for layer in range(n_layers):
        X = centroids[:, layer, :]
        y = log_mags
        
        # LOO-CV for α selection (scoring='r2' is default for RidgeCV)
        probe = RidgeCV(alphas=alphas, cv=None)  # cv=None → efficient LOO
        probe.fit(X, y)
        
        probe_r2[layer] = probe.score(X, y)  # training R² (for comparison)
        probe_weights[layer] = probe.coef_
        probe_intercepts[layer] = probe.intercept_
        cv_alphas[layer] = probe.alpha_
        
        # Compute LOO-CV R² manually for reporting
        # RidgeCV with cv=None uses GCV which is an efficient LOO approximation
        # The best_score_ attribute gives the mean LOO score
        cv_r2[layer] = probe.best_score_
    
    return probe_r2, probe_weights, probe_intercepts, cv_alphas, cv_r2


def evaluate_go_no_go(probe_r2, model_config):
    """Evaluate go/no-go gate: R² > 0.50 at any layer.
    
    Returns:
        gate_pass: bool
        best_layer: int (layer with highest R²; ties within 1% → earliest)
        report: dict with full details
    """
    n_layers = len(probe_r2)
    max_r2 = probe_r2.max()
    gate_pass = max_r2 > GATE_R2_THRESHOLD
    
    # Find best layer: highest R², ties within 1% → earliest
    threshold = max_r2 - 0.01
    candidate_layers = np.where(probe_r2 >= threshold)[0]
    best_layer = int(candidate_layers[0])  # earliest among ties
    
    report = {
        "gate_pass": bool(gate_pass),
        "gate_threshold": GATE_R2_THRESHOLD,
        "max_r2": float(max_r2),
        "best_layer": best_layer,
        "best_layer_r2": float(probe_r2[best_layer]),
        "n_layers_above_threshold": int((probe_r2 > GATE_R2_THRESHOLD).sum()),
        "all_r2": probe_r2.tolist(),
        "tie_candidates": candidate_layers.tolist(),
    }
    
    return gate_pass, best_layer, report


# ---------------------------------------------------------------------------
# Step 2: Magnitude Direction Identification
# ---------------------------------------------------------------------------

def get_magnitude_direction(probe_weights, layer):
    """Extract and normalise the magnitude direction from the probe weight vector.
    
    Returns unit-length v_mag at the specified layer.
    """
    v_mag = probe_weights[layer].copy()
    norm = np.linalg.norm(v_mag)
    if norm < 1e-10:
        raise ValueError(f"Probe weight vector at layer {layer} has near-zero norm: {norm}")
    v_mag /= norm
    return v_mag


def get_pca_direction(centroids, layer, log_mags):
    """Compute PCA-based magnitude direction as robustness check.
    
    Returns:
        pca_direction: unit vector (d_model,)
        pca_corr: Pearson correlation between PC1 scores and log(magnitude)
        pca_valid: bool (|corr| > 0.80)
    """
    X = centroids[:, layer, :]  # (n_mags, d_model)
    
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(X).ravel()  # (n_mags,)
    
    pca_direction = pca.components_[0]  # (d_model,)
    pca_direction /= np.linalg.norm(pca_direction)
    
    corr, p_val = pearsonr(pc1_scores, log_mags)
    
    # Ensure direction points in same direction as increasing magnitude
    if corr < 0:
        pca_direction = -pca_direction
        corr = -corr
    
    pca_valid = abs(corr) > PCA_CORR_THRESHOLD
    
    return pca_direction, float(corr), pca_valid


# ---------------------------------------------------------------------------
# Step 3: Anchor Computation
# ---------------------------------------------------------------------------

def get_anchor_projection(centroids, layer, v_mag, anchor_magnitude=1000):
    """Compute the projection of the anchor (magnitude=1000) centroid onto v_mag.
    
    Pre-reg: "the centroid representation of a fixed large anchor (magnitude = 1000,
    averaged across the 5 carrier sentences from Paradigm A)"
    """
    mag_idx = MAGNITUDES.index(anchor_magnitude)
    anchor_hidden = centroids[mag_idx, layer, :]  # (d_model,) — already centroid
    
    proj_anchor = np.dot(anchor_hidden, v_mag)
    return proj_anchor


# ---------------------------------------------------------------------------
# Step 4: Activation Patching via Forward Pass Hooks
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id, device="cuda"):
    """Load HuggingFace model for activation patching.
    
    CRITICAL: 
    - Do NOT use device_map='auto' (ROCm incompatibility)
    - Set output_hidden_states AFTER model creation
    - Move ALL tensors to device
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    model.config.output_hidden_states = True  # AFTER creation
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def find_first_magnitude_position(tokenizer, full_text, magnitude_str, model_key):
    """Find the token position of the first magnitude in a comparison prompt.
    
    Uses offset_mapping as primary method (validated in sanity check).
    For A/B labelled prompts like "Which is larger, A) 50 or B) 80? ..."
    we find the first occurrence of the magnitude string after "A) ".
    """
    encoding = tokenizer(
        full_text, 
        return_tensors="pt", 
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    
    offsets = encoding["offset_mapping"][0]  # (seq_len, 2)
    
    # Find character position of the first magnitude (after "A) ")
    search_marker = f"A) {magnitude_str}"
    marker_pos = full_text.find(search_marker)
    if marker_pos >= 0:
        char_start = marker_pos + 3  # skip "A) "
    else:
        # Fallback: find the magnitude string directly
        char_start = full_text.find(magnitude_str)
        if char_start == -1:
            raise ValueError(f"Magnitude '{magnitude_str}' not found in prompt")
    
    char_end = char_start + len(magnitude_str)
    
    # Find the token(s) covering this character span
    mag_tokens = []
    for idx, (s, e) in enumerate(offsets.tolist()):
        if s < char_end and e > char_start:
            mag_tokens.append(idx)
    
    if not mag_tokens:
        raise ValueError(f"No tokens found for magnitude '{magnitude_str}' at chars {char_start}-{char_end}")
    
    # Pre-reg: "at the final token of the magnitude expression" for multi-token
    return mag_tokens[-1]


def run_patched_forward_pass(
    model, tokenizer, prompt_text, mag_position, 
    patch_layer, patch_vector, device="cuda"
):
    """Run a forward pass with activation patching at the specified layer.
    
    Args:
        model: HuggingFace CausalLM
        tokenizer: corresponding tokenizer
        prompt_text: full formatted prompt string
        mag_position: token index to patch
        patch_layer: which layer to intervene at
        patch_vector: (d_model,) vector to ADD to the hidden state
        device: torch device
        
    Returns:
        logits at the answer position (after the prompt)
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Register a hook to patch the hidden state
    patch_tensor = torch.tensor(patch_vector, dtype=torch.float16, device=device)
    
    hook_handles = []
    
    def patch_hook(module, input, output):
        # output is a tuple: (hidden_states, ...) or just hidden_states
        # For transformer layers, output[0] is the hidden states
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        
        # Patch at the magnitude token position
        hs[0, mag_position, :] += patch_tensor
        
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs
    
    # Identify the correct layer module
    # Llama: model.model.layers[L]
    # Mistral: model.model.layers[L]
    layer_module = model.model.layers[patch_layer]
    handle = layer_module.register_forward_hook(patch_hook)
    hook_handles.append(handle)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        for h in hook_handles:
            h.remove()
    
    return outputs.logits


def get_unpatched_logits(model, tokenizer, prompt_text, device="cuda"):
    """Run a clean forward pass without patching."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.logits


def compute_comparison_probability(logits, tokenizer, correct_answer, model_key):
    """Extract p(correct) from logits at the answer position.
    
    Pre-reg: "extract the logit for each candidate answer token at the answer position 
    and compute p(correct) vs p(incorrect) from the softmax over candidate tokens only"
    
    For A/B format: candidate tokens are 'A' and 'B'.
    """
    # Answer position is the last token position (model predicts next token)
    answer_logits = logits[0, -1, :]  # (vocab_size,)
    
    # Get token IDs for A and B
    token_a = tokenizer.encode("A", add_special_tokens=False)
    token_b = tokenizer.encode("B", add_special_tokens=False)
    
    # Handle potential tokenization issues — take first token if multi-token
    id_a = token_a[0] if token_a else None
    id_b = token_b[0] if token_b else None
    
    if id_a is None or id_b is None:
        raise ValueError("Could not tokenize A or B answer tokens")
    
    logit_a = answer_logits[id_a].float()
    logit_b = answer_logits[id_b].float()
    
    # Softmax over candidate tokens only
    logits_ab = torch.stack([logit_a, logit_b])
    probs_ab = torch.softmax(logits_ab, dim=0)
    
    p_a = probs_ab[0].item()
    p_b = probs_ab[1].item()
    
    if correct_answer == "A":
        return p_a
    elif correct_answer == "B":
        return p_b
    else:
        raise ValueError(f"Unknown correct_answer: {correct_answer}")


# ---------------------------------------------------------------------------
# Step 5: Generate Random Directions
# ---------------------------------------------------------------------------

def generate_random_directions(d_model, n_directions=N_RANDOM_DIRECTIONS, seed=RANDOM_SEED):
    """Generate random unit vectors for the specificity control.
    
    Pre-reg: "10 random unit vectors in d_model-dimensional space, sampled from a 
    standard normal distribution (seed 42) and normalised."
    """
    rng = np.random.RandomState(seed)
    directions = rng.randn(n_directions, d_model)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms
    return directions


# ---------------------------------------------------------------------------
# Step 6: Load Comparison Prompts (200 from stimulus files)
# ---------------------------------------------------------------------------

def load_paradigm_d_prompts(stimuli_dir):
    """Load the 200 pre-generated Paradigm D comparison prompts.
    
    File: {stimuli_dir}/paradigm_d_prompts.json
    Schema: paradigm_d_id, source_pair_id, prompt, first_presented, second_presented,
            correct_answer, nominal_baseline, nominal_ratio
    """
    prompt_path = Path(stimuli_dir) / "paradigm_d_prompts.json"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Paradigm D prompts not found: {prompt_path}\n"
            f"Run stimuli_generation.py first."
        )
    
    with open(prompt_path) as f:
        prompts = json.load(f)
    
    print(f"  Loaded {len(prompts)} Paradigm D prompts from {prompt_path.name}")
    
    # Validate schema
    required = {"first_presented", "second_presented", "correct_answer"}
    if prompts and not required.issubset(prompts[0].keys()):
        missing = required - set(prompts[0].keys())
        raise ValueError(f"Stimulus schema missing keys: {missing}")
    
    return prompts


def format_comparison_prompt(prompt_data, model_key):
    """Format a comparison prompt with A/B labels and chat template.
    
    Matches the Paradigm B deviation: explicit A/B labels + chat template.
    
    Stimulus schema:
        first_presented: int, second_presented: int, correct_answer: str (the number),
        prompt: str (raw symbolic format, ignored — we rebuild with A/B labels)
    """
    first = prompt_data["first_presented"]
    second = prompt_data["second_presented"]
    correct_number = str(prompt_data["correct_answer"])
    
    # A = first_presented, B = second_presented
    # Determine which letter is correct
    if str(first) == correct_number:
        correct = "A"
    elif str(second) == correct_number:
        correct = "B"
    else:
        raise ValueError(
            f"correct_answer '{correct_number}' doesn't match "
            f"first={first} or second={second}"
        )
    
    # Rebuild prompt with A/B labels (matching Paradigm B deviation)
    raw_prompt = f"Which is larger, A) {first} or B) {second}? Answer with only A or B."
    
    # Apply chat template
    template = get_chat_template(model_key)
    formatted = template.format(prompt=raw_prompt)
    
    # First magnitude string (for position finding — the A) value)
    first_mag_str = str(first)
    
    return formatted, correct, first_mag_str


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_paradigm_d(
    model_key,
    project_root,
    domain="numerical",
    device="cuda",
    skip_patching=False,
):
    """Run the complete Paradigm D pipeline.
    
    Args:
        model_key: "llama" or "mistral"
        project_root: path to C:\\weber
        domain: "numerical" (primary)
        device: "cuda" or "cpu"
        skip_patching: if True, only run go/no-go gate (for testing)
    """
    results_dir = Path(project_root) / "results"
    stimuli_dir = Path(project_root) / "stimuli"
    output_dir = results_dir / "paradigm_d"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_config = MODELS[model_key]
    
    print(f"\n{'='*70}")
    print(f"PARADIGM D: Causal Intervention — {model_key} / {domain}")
    print(f"{'='*70}")
    
    # ----- Step 1: Load Paradigm A centroids and train probes -----
    print(f"\n--- Step 1: Go/No-Go Gate (Linear Probe) ---")
    centroids = load_paradigm_a_centroids(results_dir, model_key, domain)
    
    # Primary: α=1.0 (pre-reg default)
    probe_r2, probe_weights, probe_intercepts = train_probes(centroids, alpha=1.0)
    gate_pass, best_layer, gate_report = evaluate_go_no_go(probe_r2, model_config)
    
    # CV robustness: LOO-CV α selection
    print(f"  Training CV-regularised probes (robustness check)...")
    cv_r2_train, cv_weights, cv_intercepts, cv_alphas, cv_r2_loo = train_probes_cv(centroids)
    
    # Find CV best layer (same tie-breaking rule)
    cv_max = cv_r2_loo.max()
    cv_threshold = cv_max - 0.01
    cv_candidates = np.where(cv_r2_loo >= cv_threshold)[0]
    cv_best_layer = int(cv_candidates[0])
    
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")
    print(f"  Primary (α=1.0): max R² = {gate_report['max_r2']:.4f}, "
          f"best layer = {best_layer} (R² = {probe_r2[best_layer]:.4f})")
    print(f"  CV robustness:   max LOO-R² = {cv_max:.4f}, "
          f"best layer = {cv_best_layer} (LOO-R² = {cv_r2_loo[cv_best_layer]:.4f}, "
          f"α = {cv_alphas[cv_best_layer]:.1f})")
    print(f"  Layers > 0.50: {gate_report['n_layers_above_threshold']} (primary), "
          f"{int((cv_r2_loo > 0.50).sum())} (CV)")
    
    # Direction alignment between primary and CV at the primary best layer
    v_primary = probe_weights[best_layer].copy()
    v_primary /= np.linalg.norm(v_primary) + 1e-10
    v_cv = cv_weights[best_layer].copy()
    v_cv /= np.linalg.norm(v_cv) + 1e-10
    primary_cv_alignment = float(np.dot(v_primary, v_cv))
    print(f"  Primary-CV direction alignment at layer {best_layer}: {primary_cv_alignment:.4f}")
    
    # Add CV info to gate report
    gate_report["cv_robustness"] = {
        "cv_best_layer": cv_best_layer,
        "cv_best_loo_r2": float(cv_r2_loo[cv_best_layer]),
        "cv_best_alpha": float(cv_alphas[cv_best_layer]),
        "cv_all_loo_r2": cv_r2_loo.tolist(),
        "cv_all_alphas": cv_alphas.tolist(),
        "primary_cv_direction_alignment": primary_cv_alignment,
        "n_cv_layers_above_050": int((cv_r2_loo > 0.50).sum()),
    }
    
    # Save gate report
    gate_path = output_dir / f"{model_key}_{domain}_gate_report.json"
    with open(gate_path, 'w') as f:
        json.dump(gate_report, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved gate report: {gate_path}")
    
    if not gate_pass:
        print(f"\n  *** GO/NO-GO GATE FAILED ***")
        print(f"  No layer achieved R² > {GATE_R2_THRESHOLD}")
        print(f"  Paradigm D cannot proceed. Fallback: report probe R² and proceed to PCA.")
        
        # Still compute PCA direction as fallback
        log_mags = np.log(np.array(MAGNITUDES, dtype=np.float64))
        pca_dir, pca_corr, pca_valid = get_pca_direction(centroids, best_layer, log_mags)
        print(f"  PCA fallback: corr = {pca_corr:.4f}, valid = {pca_valid}")
        
        gate_report["pca_fallback"] = {
            "direction_corr": pca_corr,
            "valid": pca_valid,
        }
        with open(gate_path, 'w') as f:
            json.dump(gate_report, f, indent=2, cls=NumpyEncoder)
        
        return gate_report
    
    if skip_patching:
        print(f"\n  --skip-patching set. Stopping after gate evaluation.")
        return gate_report
    
    # ----- Step 2: Identify magnitude direction -----
    print(f"\n--- Step 2: Magnitude Direction ---")
    v_mag = get_magnitude_direction(probe_weights, best_layer)
    print(f"  v_mag norm (should be 1.0): {np.linalg.norm(v_mag):.6f}")
    
    # Compute anchor projection
    log_mags = np.log(np.array(MAGNITUDES, dtype=np.float64))
    proj_anchor = get_anchor_projection(centroids, best_layer, v_mag, anchor_magnitude=1000)
    print(f"  Anchor (1000) projection onto v_mag: {proj_anchor:.4f}")
    
    # PCA direction (robustness check)
    pca_dir, pca_corr, pca_valid = get_pca_direction(centroids, best_layer, log_mags)
    print(f"  PCA direction: corr(PC1, log_mag) = {pca_corr:.4f}, valid = {pca_valid}")
    
    # Direction alignment
    alignment = np.dot(v_mag, pca_dir)
    print(f"  Probe-PCA alignment (cosine): {alignment:.4f}")
    
    # ----- Step 3: Load model for patching -----
    print(f"\n--- Step 3: Load Model ---")
    model, tokenizer = load_model_and_tokenizer(model_config["model_id"], device)
    d_model = model.config.hidden_size
    print(f"  d_model: {d_model}")
    
    # ----- Step 4: Load comparison prompts -----
    print(f"\n--- Step 4: Load Comparison Prompts ---")
    prompts = load_paradigm_d_prompts(stimuli_dir)
    
    # ----- Step 5: Generate random directions -----
    print(f"\n--- Step 5: Random Directions ---")
    random_dirs = generate_random_directions(d_model)
    print(f"  Generated {N_RANDOM_DIRECTIONS} random unit vectors (seed {RANDOM_SEED})")
    
    # ----- Step 6: Activation Patching -----
    print(f"\n--- Step 6: Activation Patching ---")
    
    results = {
        "model": model_key,
        "domain": domain,
        "best_layer": best_layer,
        "probe_alpha": 1.0,
        "probe_r2_at_layer": float(probe_r2[best_layer]),
        "probe_note": "R² is training R² with α=1.0; p>>n means this overfits. See cv_robustness.",
        "cv_robustness": {
            "cv_best_layer": cv_best_layer,
            "cv_alpha_at_primary_layer": float(cv_alphas[best_layer]),
            "cv_loo_r2_at_primary_layer": float(cv_r2_loo[best_layer]),
            "cv_best_loo_r2": float(cv_r2_loo[cv_best_layer]),
            "primary_cv_direction_alignment": primary_cv_alignment,
        },
        "pca_corr": pca_corr,
        "pca_valid": pca_valid,
        "probe_pca_alignment": float(alignment),
        "anchor_magnitude": 1000,
        "anchor_projection": float(proj_anchor),
        "n_prompts": len(prompts),
        "dose_levels": DOSE_LEVELS,
        "n_random_directions": N_RANDOM_DIRECTIONS,
        "prompts": [],
    }
    
    t_start = time.time()
    
    for i, prompt_data in enumerate(prompts):
        formatted, correct, first_mag_str = format_comparison_prompt(prompt_data, model_key)
        
        # Find magnitude token position
        try:
            mag_pos = find_first_magnitude_position(
                tokenizer, formatted, first_mag_str, model_key
            )
        except ValueError as e:
            print(f"  WARNING: Prompt {i} skipped — {e}")
            results["prompts"].append({"index": i, "error": str(e)})
            continue
        
        # Get unpatched baseline
        unpatched_logits = get_unpatched_logits(model, tokenizer, formatted, device)
        p_correct_baseline = compute_comparison_probability(
            unpatched_logits, tokenizer, correct, model_key
        )
        
        prompt_result = {
            "index": i,
            "paradigm_d_id": prompt_data.get("paradigm_d_id", f"PD-{i:03d}"),
            "first_presented": prompt_data["first_presented"],
            "second_presented": prompt_data["second_presented"],
            "correct_answer": correct,
            "nominal_baseline": prompt_data.get("nominal_baseline"),
            "nominal_ratio": prompt_data.get("nominal_ratio"),
            "first_mag": first_mag_str,
            "mag_position": mag_pos,
            "p_correct_baseline": p_correct_baseline,
            "magnitude_direction": {},
            "random_directions": {},
        }
        
        # ----- Magnitude direction patching at each dose -----
        # Get hidden state projection of first magnitude onto v_mag
        # We need the actual hidden state at this position to compute proj
        # Extract it from an unpatched forward pass
        with torch.no_grad():
            inputs = tokenizer(formatted, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            h_mag = outputs.hidden_states[best_layer + 1][0, mag_pos, :].cpu().numpy().astype(np.float64)
            # +1 because hidden_states[0] is embedding output, hidden_states[L+1] is layer L output
        
        proj_mag = np.dot(h_mag, v_mag)
        prompt_result["proj_first_mag"] = float(proj_mag)
        
        for dose in DOSE_LEVELS:
            # Compute patch: d × (proj_anchor − proj_mag) × v_mag
            patch_magnitude = dose * (proj_anchor - proj_mag)
            patch_vector = patch_magnitude * v_mag
            
            patched_logits = run_patched_forward_pass(
                model, tokenizer, formatted, mag_pos,
                best_layer, patch_vector, device
            )
            p_correct_patched = compute_comparison_probability(
                patched_logits, tokenizer, correct, model_key
            )
            
            delta_p = p_correct_patched - p_correct_baseline
            prompt_result["magnitude_direction"][str(dose)] = {
                "p_correct": p_correct_patched,
                "delta_p": delta_p,
                "patch_magnitude": float(patch_magnitude),
            }
        
        # ----- Random direction control -----
        for r_idx in range(N_RANDOM_DIRECTIONS):
            r_dir = random_dirs[r_idx]
            
            # Use same total perturbation magnitude as dose=1.0 magnitude direction
            # Pre-reg: "same magnitude of perturbation"
            full_patch_magnitude = proj_anchor - proj_mag
            patch_vector_random = full_patch_magnitude * r_dir
            
            patched_logits = run_patched_forward_pass(
                model, tokenizer, formatted, mag_pos,
                best_layer, patch_vector_random, device
            )
            p_correct_patched = compute_comparison_probability(
                patched_logits, tokenizer, correct, model_key
            )
            
            delta_p = p_correct_patched - p_correct_baseline
            prompt_result["random_directions"][str(r_idx)] = {
                "p_correct": p_correct_patched,
                "delta_p": delta_p,
            }
        
        # ----- PCA direction patching (if valid) -----
        if pca_valid:
            proj_mag_pca = np.dot(h_mag, pca_dir)
            # Compute anchor projection onto PCA direction
            anchor_hidden = centroids[MAGNITUDES.index(1000), best_layer, :]
            proj_anchor_pca = np.dot(anchor_hidden, pca_dir)
            
            pca_results = {}
            for dose in DOSE_LEVELS:
                patch_magnitude_pca = dose * (proj_anchor_pca - proj_mag_pca)
                patch_vector_pca = patch_magnitude_pca * pca_dir
                
                patched_logits = run_patched_forward_pass(
                    model, tokenizer, formatted, mag_pos,
                    best_layer, patch_vector_pca, device
                )
                p_correct_patched = compute_comparison_probability(
                    patched_logits, tokenizer, correct, model_key
                )
                
                delta_p = p_correct_patched - p_correct_baseline
                pca_results[str(dose)] = {
                    "p_correct": p_correct_patched,
                    "delta_p": delta_p,
                }
            
            prompt_result["pca_direction"] = pca_results
        
        results["prompts"].append(prompt_result)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(prompts) - i - 1) / rate
            print(f"  Processed {i+1}/{len(prompts)} prompts "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
    
    elapsed_total = time.time() - t_start
    results["elapsed_seconds"] = elapsed_total
    print(f"\n  Total patching time: {elapsed_total:.1f}s")
    
    # ----- Step 7: Evaluate H7 -----
    print(f"\n--- Step 7: Evaluate H7 ---")
    h7_result = evaluate_h7(results)
    results["h7"] = h7_result
    
    # Save full results
    results_path = output_dir / f"{model_key}_{domain}_paradigm_d_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved results: {results_path}")
    
    # Print summary
    print_summary(results, h7_result)
    
    return results


# ---------------------------------------------------------------------------
# H7 Evaluation
# ---------------------------------------------------------------------------

def evaluate_h7(results):
    """Evaluate H7: Functional Relevance.
    
    Pre-reg criterion:
    "The mean absolute comparison-probability shift under magnitude-direction patching
    exceeds the 97.5th percentile of the random-direction shift distribution,
    for at least 75% of the 200 comparison prompts."
    
    Evaluated at dose = 1.00 (full patching).
    """
    valid_prompts = [p for p in results["prompts"] if "error" not in p]
    n_valid = len(valid_prompts)
    
    if n_valid == 0:
        return {"pass": False, "error": "No valid prompts"}
    
    prompts_exceeding = 0
    prompt_details = []
    
    for p in valid_prompts:
        # Magnitude direction |Δp| at dose=1.00
        mag_delta = abs(p["magnitude_direction"]["1.0"]["delta_p"])
        
        # Random direction |Δp| distribution
        random_deltas = [abs(p["random_directions"][str(r)]["delta_p"]) 
                        for r in range(N_RANDOM_DIRECTIONS)]
        
        # 97.5th percentile of random distribution
        threshold = np.percentile(random_deltas, H7_PERCENTILE)
        
        exceeds = mag_delta > threshold
        if exceeds:
            prompts_exceeding += 1
        
        prompt_details.append({
            "index": p["index"],
            "mag_abs_delta_p": mag_delta,
            "random_97_5_pct": threshold,
            "random_mean": float(np.mean(random_deltas)),
            "random_max": float(np.max(random_deltas)),
            "exceeds": exceeds,
        })
    
    proportion = prompts_exceeding / n_valid
    h7_pass = proportion >= H7_PROMPT_THRESHOLD
    
    # Also compute aggregate statistics
    all_mag_deltas = [abs(p["magnitude_direction"]["1.0"]["delta_p"]) for p in valid_prompts]
    all_random_deltas = []
    for p in valid_prompts:
        for r in range(N_RANDOM_DIRECTIONS):
            all_random_deltas.append(abs(p["random_directions"][str(r)]["delta_p"]))
    
    # Dose-response summary
    dose_response = {}
    for dose in DOSE_LEVELS:
        dose_str = str(dose)
        deltas = [p["magnitude_direction"][dose_str]["delta_p"] for p in valid_prompts]
        dose_response[dose_str] = {
            "mean_delta_p": float(np.mean(deltas)),
            "mean_abs_delta_p": float(np.mean(np.abs(deltas))),
            "median_delta_p": float(np.median(deltas)),
            "std_delta_p": float(np.std(deltas)),
        }
    
    result = {
        "pass": h7_pass,
        "n_valid_prompts": n_valid,
        "n_prompts_exceeding": prompts_exceeding,
        "proportion_exceeding": proportion,
        "criterion_threshold": H7_PROMPT_THRESHOLD,
        "percentile_used": H7_PERCENTILE,
        "aggregate": {
            "mag_mean_abs_delta": float(np.mean(all_mag_deltas)),
            "mag_median_abs_delta": float(np.median(all_mag_deltas)),
            "random_mean_abs_delta": float(np.mean(all_random_deltas)),
            "random_median_abs_delta": float(np.median(all_random_deltas)),
            "mag_to_random_ratio": float(np.mean(all_mag_deltas) / max(np.mean(all_random_deltas), 1e-10)),
        },
        "dose_response": dose_response,
        "prompt_details": prompt_details,
    }
    
    return result


# ---------------------------------------------------------------------------
# Summary Printing
# ---------------------------------------------------------------------------

def print_summary(results, h7_result):
    print(f"\n{'='*70}")
    print(f"PARADIGM D SUMMARY: {results['model']} / {results['domain']}")
    print(f"{'='*70}")
    print(f"  Patching layer: {results['best_layer']} (probe R² = {results['probe_r2_at_layer']:.4f})")
    print(f"  PCA correlation: {results['pca_corr']:.4f} (valid = {results['pca_valid']})")
    print(f"  Probe-PCA alignment: {results['probe_pca_alignment']:.4f}")
    print(f"")
    print(f"  H7 (Functional Relevance): {'PASS' if h7_result['pass'] else 'FAIL'}")
    print(f"  Prompts exceeding 97.5th pct: {h7_result['n_prompts_exceeding']}/{h7_result['n_valid_prompts']} "
          f"({h7_result['proportion_exceeding']:.1%})")
    print(f"  Criterion: ≥{H7_PROMPT_THRESHOLD:.0%}")
    print(f"")
    print(f"  Magnitude direction mean |Δp|: {h7_result['aggregate']['mag_mean_abs_delta']:.4f}")
    print(f"  Random direction mean |Δp|:    {h7_result['aggregate']['random_mean_abs_delta']:.4f}")
    print(f"  Ratio (mag/random):            {h7_result['aggregate']['mag_to_random_ratio']:.2f}x")
    print(f"")
    print(f"  Dose-response (mean Δp):")
    for dose_str, stats in h7_result['dose_response'].items():
        print(f"    dose {dose_str}: Δp = {stats['mean_delta_p']:+.4f} (|Δp| = {stats['mean_abs_delta_p']:.4f})")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paradigm D: Causal Intervention via Activation Patching"
    )
    parser.add_argument(
        "--model", choices=["llama", "mistral"], required=True,
        help="Model to run"
    )
    parser.add_argument(
        "--project-root", type=str, default=r"C:\weber",
        help="Project root directory"
    )
    parser.add_argument(
        "--domain", type=str, default="numerical",
        help="Magnitude domain (default: numerical)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)"
    )
    parser.add_argument(
        "--skip-patching", action="store_true",
        help="Only run go/no-go gate, skip patching"
    )
    
    args = parser.parse_args()
    
    # Set environment variable for ROCm
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    
    run_paradigm_d(
        model_key=args.model,
        project_root=args.project_root,
        domain=args.domain,
        device=args.device,
        skip_patching=args.skip_patching,
    )


if __name__ == "__main__":
    main()
