"""
run_exploratory_models.py — Weber 4.2 Exploratory Generalisation
================================================================
Runs Paradigms A (geometry) + B (numerical behaviour) + C (precision) on
additional exploratory models to test cross-architecture generalisability.

Hardware: AMD Radeon 7900 GRE (16GB VRAM), ROCm 6.4
Prerequisite: Existing stimulus files from the main experiment (stimuli/ dir)

Usage:
    python run_exploratory_models.py --model gemma
    python run_exploratory_models.py --model qwen
    python run_exploratory_models.py --model both

This script is self-contained — it does not import from the main pipeline.
It replicates the core extraction/analysis logic for portability.

Author: JP Cacioli / Classical Minds, Modern Machines — Project 4.2
Status: Post-registration exploratory addition
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import cosine as cosine_dist

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — matches pre-registered specs exactly
# ═══════════════════════════════════════════════════════════════

NUMERICAL_MAGNITUDES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50,
    60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1000
]

CARRIER_SENTENCES = [
    "The number {N} is a quantity.",
    "There are {N} items.",
    "{N} was the value.",
    "The count reached {N}.",
    "Exactly {N} were measured.",
]

SEED = 42
DEVICE = "cuda"

# Model registry
MODELS = {
    "gemma": {
        "hf_id": "google/gemma-2-9b-it",
        "dtype": torch.bfloat16,   # Gemma-2 native precision; try BF16 first
        "fallback_dtype": None,     # If BF16 OOMs, this model may not fit
        "short_name": "gemma2_9b",
    },
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": torch.float16,
        "fallback_dtype": torch.bfloat16,
        "short_name": "qwen25_7b",
    },
}


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {path}")


def get_vram_gb():
    """Report current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0, 0


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model(model_key):
    """Load model and tokenizer. Handle OOM gracefully."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODELS[model_key]
    hf_id = spec["hf_id"]
    dtype = spec["dtype"]

    print(f"\n{'='*60}")
    print(f"Loading {hf_id} in {dtype}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            # Do NOT use device_map='auto' — causes ROCm kernel errors on 7900 GRE
        )
        model = model.to(DEVICE)
        model.config.output_hidden_states = True
        model.eval()

        alloc, res = get_vram_gb()
        print(f"  Loaded successfully. VRAM: {alloc:.1f}GB allocated, {res:.1f}GB reserved")

        # Verify hidden states work
        test_ids = tokenizer("test", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**test_ids)
        n_layers = len(out.hidden_states)
        d_model = out.hidden_states[0].shape[-1]
        print(f"  Layers: {n_layers} (including embedding), d_model: {d_model}")

        return model, tokenizer, n_layers, d_model

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if spec["fallback_dtype"] is not None:
            print(f"  OOM with {dtype}. Trying fallback: {spec['fallback_dtype']}")
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=spec["fallback_dtype"],
                trust_remote_code=True,
            )
            model = model.to(DEVICE)
            model.config.output_hidden_states = True
            model.eval()
            alloc, res = get_vram_gb()
            print(f"  Loaded with fallback. VRAM: {alloc:.1f}GB allocated")
            test_ids = tokenizer("test", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**test_ids)
            n_layers = len(out.hidden_states)
            d_model = out.hidden_states[0].shape[-1]
            print(f"  Layers: {n_layers}, d_model: {d_model}")
            return model, tokenizer, n_layers, d_model
        else:
            print(f"  FATAL: {hf_id} does not fit in 16GB VRAM. Skipping.")
            return None, None, None, None


# ═══════════════════════════════════════════════════════════════
# PARADIGM A: Hidden State Extraction
# ═══════════════════════════════════════════════════════════════

def find_magnitude_position(tokenizer, text, magnitude_str):
    """Find token position of magnitude using offset_mapping (primary) or search (fallback)."""
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoding["offset_mapping"][0]

    # Find character span of magnitude in text
    mag_start = text.find(magnitude_str)
    if mag_start == -1:
        raise ValueError(f"Magnitude '{magnitude_str}' not found in '{text}'")
    mag_end = mag_start + len(magnitude_str)

    # Find the last token that overlaps with the magnitude span
    last_tok_pos = None
    for idx, (s, e) in enumerate(offsets):
        if s < mag_end and e > mag_start and e > 0:
            last_tok_pos = idx

    if last_tok_pos is None:
        # Fallback: search for magnitude tokens by decoding
        input_ids = encoding["input_ids"][0]
        for idx in range(len(input_ids) - 1, 0, -1):
            tok_text = tokenizer.decode(input_ids[idx])
            if magnitude_str.endswith(tok_text.strip()):
                last_tok_pos = idx
                break

    if last_tok_pos is None:
        raise ValueError(f"Could not find token position for '{magnitude_str}' in '{text}'")

    return last_tok_pos


def extract_hidden_states(model, tokenizer, n_layers):
    """Extract hidden states for all magnitudes × carriers. Returns dict of arrays."""
    print("\n--- Paradigm A: Extracting hidden states ---")

    all_hidden = {}  # {magnitude: [n_layers × d_model] averaged across carriers}
    per_carrier = {}  # {magnitude: {carrier_idx: [n_layers × d_model]}}

    for mag in NUMERICAL_MAGNITUDES:
        mag_str = str(mag)
        carrier_states = []

        for c_idx, carrier in enumerate(CARRIER_SENTENCES):
            text = carrier.replace("{N}", mag_str)
            inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            tok_pos = find_magnitude_position(tokenizer, text, mag_str)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract hidden state at magnitude token position, all layers
            layer_vecs = []
            for layer_idx in range(n_layers):
                h = outputs.hidden_states[layer_idx][0, tok_pos, :].cpu().float().numpy()
                layer_vecs.append(h)
            carrier_states.append(np.array(layer_vecs))  # [n_layers, d_model]

        # Average across carriers → centroid per magnitude per layer
        carrier_stack = np.array(carrier_states)  # [5, n_layers, d_model]
        centroid = carrier_stack.mean(axis=0)  # [n_layers, d_model]
        all_hidden[mag] = centroid
        per_carrier[mag] = carrier_stack

    print(f"  Extracted: {len(NUMERICAL_MAGNITUDES)} magnitudes × {len(CARRIER_SENTENCES)} carriers × {n_layers} layers")
    return all_hidden, per_carrier


# ═══════════════════════════════════════════════════════════════
# PARADIGM A: Analysis (RSA, model fitting)
# ═══════════════════════════════════════════════════════════════

def compute_pairwise_distances(hidden_states, n_layers):
    """Compute cosine and Euclidean pairwise distances at each layer."""
    mags = sorted(hidden_states.keys())
    pairs = list(combinations(mags, 2))
    n_pairs = len(pairs)

    cosine_dists = np.zeros((n_layers, n_pairs))
    euclid_dists = np.zeros((n_layers, n_pairs))

    for layer in range(n_layers):
        for p_idx, (m1, m2) in enumerate(pairs):
            v1 = hidden_states[m1][layer]
            v2 = hidden_states[m2][layer]
            # Cast to float64 to avoid overflow (Mistral FP16 issue also applies to others)
            v1_64 = v1.astype(np.float64)
            v2_64 = v2.astype(np.float64)
            cosine_dists[layer, p_idx] = cosine_dist(v1_64, v2_64)
            euclid_dists[layer, p_idx] = np.linalg.norm(v1_64 - v2_64)

    return cosine_dists, euclid_dists, pairs


def fit_geometric_models(distances, pairs):
    """Fit Linear, Weber, Stevens models to pairwise distances."""
    mags_i = np.array([p[0] for p in pairs], dtype=float)
    mags_j = np.array([p[1] for p in pairs], dtype=float)

    # Predictors
    linear_pred = np.abs(mags_i - mags_j)
    weber_pred = np.abs(np.log(mags_i) - np.log(mags_j))

    results = {}
    for name, pred in [("linear", linear_pred), ("weber", weber_pred)]:
        # OLS fit: d = a + b * predictor
        X = np.column_stack([np.ones(len(pred)), pred])
        beta, residuals, _, _ = np.linalg.lstsq(X, distances, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((distances - y_hat) ** 2)
        ss_tot = np.sum((distances - distances.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(distances)
        k = 2
        aic = n * np.log(ss_res / n + 1e-15) + 2 * k
        results[name] = {"r2": r2, "aic": aic, "params": beta.tolist()}

    # Stevens: d = a + b * |n1^beta - n2^beta|
    try:
        def stevens_func(X, a, b, beta):
            n1, n2 = X
            return a + b * np.abs(np.power(n1, beta) - np.power(n2, beta))

        popt, _ = curve_fit(
            stevens_func, (mags_i, mags_j), distances,
            p0=[0.0, 1.0, 0.5], bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 2.0]),
            maxfev=5000
        )
        y_hat = stevens_func((mags_i, mags_j), *popt)
        ss_res = np.sum((distances - y_hat) ** 2)
        ss_tot = np.sum((distances - distances.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(distances)
        k = 3
        aic = n * np.log(ss_res / n + 1e-15) + 2 * k
        results["stevens"] = {"r2": r2, "aic": aic, "params": popt.tolist(), "beta": float(popt[2])}
    except Exception as e:
        results["stevens"] = {"r2": 0, "aic": 1e10, "params": [], "beta": None, "error": str(e)}

    return results


def mantel_test(model_rdm, theoretical_rdm, n_perms=10000):
    """Mantel permutation test for RSA."""
    n_items = int(0.5 + 0.5 * np.sqrt(1 + 8 * len(model_rdm)))
    observed_rho, _ = stats.spearmanr(model_rdm, theoretical_rdm)

    # Reconstruct square form for permutation
    sq = np.zeros((n_items, n_items))
    idx = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            sq[i, j] = sq[j, i] = model_rdm[idx]
            idx += 1

    rng = np.random.RandomState(SEED)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(n_items)
        sq_perm = sq[np.ix_(perm, perm)]
        perm_vec = sq_perm[np.triu_indices(n_items, k=1)]
        rho, _ = stats.spearmanr(perm_vec, theoretical_rdm)
        if rho >= observed_rho:
            count += 1

    p_value = (count + 1) / (n_perms + 1)
    return observed_rho, p_value


def run_paradigm_a_analysis(hidden_states, n_layers):
    """Full Paradigm A analysis: distances, model fits, RSA."""
    print("\n--- Paradigm A: Running analysis ---")

    cosine_dists, euclid_dists, pairs = compute_pairwise_distances(hidden_states, n_layers)
    mags_i = np.array([p[0] for p in pairs], dtype=float)
    mags_j = np.array([p[1] for p in pairs], dtype=float)

    # Theoretical RDMs
    linear_rdm = np.abs(mags_i - mags_j)
    weber_rdm = np.abs(np.log(mags_i) - np.log(mags_j))

    # Z-score theoretical RDMs
    linear_rdm_z = (linear_rdm - linear_rdm.mean()) / (linear_rdm.std() + 1e-15)
    weber_rdm_z = (weber_rdm - weber_rdm.mean()) / (weber_rdm.std() + 1e-15)

    # Determine primary layer range (middle-to-late: roughly layers n//2 to n-1)
    n_transformer_layers = n_layers - 1  # subtract embedding layer
    primary_start = n_transformer_layers // 2
    primary_end = n_transformer_layers
    primary_range = list(range(primary_start, primary_end))

    results = {"layers": {}, "summary": {}}

    h1_pass_cosine = 0
    h1_pass_euclid = 0
    stevens_betas = []

    for layer in range(n_layers):
        layer_key = f"layer_{layer:02d}"
        layer_res = {}

        for metric_name, dists in [("cosine", cosine_dists[layer]), ("euclidean", euclid_dists[layer])]:
            # Model fits
            fits = fit_geometric_models(dists, pairs)

            # RSA
            weber_rho, weber_p = mantel_test(dists, weber_rdm_z, n_perms=10000)
            linear_rho, _ = stats.spearmanr(dists, linear_rdm_z)

            layer_res[metric_name] = {
                "model_fits": fits,
                "rsa": {
                    "weber": {"rho": weber_rho, "p": weber_p},
                    "linear_rho": linear_rho,
                }
            }

            # H1 check at primary layers
            if layer in primary_range:
                weber_wins_aic = fits["weber"]["aic"] < fits["linear"]["aic"]
                weber_wins_rsa = (weber_rho > linear_rho) and (weber_p < 0.017)
                if weber_wins_aic and weber_wins_rsa:
                    if metric_name == "cosine":
                        h1_pass_cosine += 1
                    else:
                        h1_pass_euclid += 1

        # Track Stevens beta
        if "stevens" in layer_res.get("cosine", {}).get("model_fits", {}):
            beta = layer_res["cosine"]["model_fits"]["stevens"].get("beta")
            if beta is not None:
                stevens_betas.append(beta)

        results["layers"][layer_key] = layer_res

        if layer % 8 == 0:
            wr2 = layer_res["cosine"]["model_fits"]["weber"]["r2"]
            wrho = layer_res["cosine"]["rsa"]["weber"]["rho"]
            print(f"  Layer {layer:2d}: Weber R²={wr2:.3f}, RSA ρ={wrho:.3f}")

    n_primary = len(primary_range)
    h1_threshold = max(1, n_primary // 2 + 1)  # majority

    results["summary"] = {
        "primary_layer_range": [primary_start, primary_end - 1],
        "n_primary_layers": n_primary,
        "h1_pass_cosine": h1_pass_cosine,
        "h1_pass_euclidean": h1_pass_euclid,
        "h1_threshold": h1_threshold,
        "h1_supported_cosine": h1_pass_cosine >= h1_threshold,
        "h1_supported_euclidean": h1_pass_euclid >= h1_threshold,
        "mean_stevens_beta": float(np.mean(stevens_betas)) if stevens_betas else None,
        "median_stevens_beta": float(np.median(stevens_betas)) if stevens_betas else None,
    }

    print(f"\n  H1 summary: cosine {h1_pass_cosine}/{n_primary}, "
          f"euclidean {h1_pass_euclid}/{n_primary} "
          f"(threshold: {h1_threshold})")
    print(f"  H1 supported (cosine): {results['summary']['h1_supported_cosine']}")
    print(f"  Mean Stevens β: {results['summary']['mean_stevens_beta']:.4f}" if results['summary']['mean_stevens_beta'] else "")

    return results, cosine_dists, pairs


# ═══════════════════════════════════════════════════════════════
# PARADIGM C: Precision Gradient (computed from A data)
# ═══════════════════════════════════════════════════════════════

def run_paradigm_c(hidden_states, n_layers):
    """Compute precision gradient from Paradigm A hidden states."""
    print("\n--- Paradigm C: Precision gradient ---")

    mags = sorted(hidden_states.keys())
    adjacent_pairs = [(mags[i], mags[i + 1]) for i in range(len(mags) - 1)]
    midpoints = [(m1 + m2) / 2 for m1, m2 in adjacent_pairs]
    log_steps = [np.log(m2) - np.log(m1) for m1, m2 in adjacent_pairs]

    results = {"layers": [], "summary": {}}
    h3_sig_count = 0

    for layer in range(n_layers):
        raw_precision = []
        for m1, m2 in adjacent_pairs:
            v1 = hidden_states[m1][layer].astype(np.float64)
            v2 = hidden_states[m2][layer].astype(np.float64)
            dist = np.linalg.norm(v1 - v2)
            raw_precision.append(1.0 / (dist + 1e-15))

        # Log-normalised precision
        normalised = [p * ls for p, ls in zip(raw_precision, log_steps)]

        # Spearman: precision vs magnitude midpoint (H3: should be negative)
        rho, p_val = stats.spearmanr(midpoints, raw_precision)

        if p_val < 0.017 and rho < 0:
            h3_sig_count += 1

        results["layers"].append({
            "layer": layer,
            "raw_precision": raw_precision,
            "normalised_precision_log": normalised,
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
        })

    results["summary"] = {
        "h3_significant_layers": h3_sig_count,
        "h3_total_layers": n_layers,
        "midpoints": midpoints,
    }

    print(f"  H3 significant layers: {h3_sig_count}/{n_layers}")
    return results


# ═══════════════════════════════════════════════════════════════
# PARADIGM B: Behavioural Discrimination (numerical only)
# ═══════════════════════════════════════════════════════════════

def load_comparison_stimuli(stimuli_dir):
    """Load pre-generated comparison stimuli from the main experiment."""
    b1_path = os.path.join(stimuli_dir, "comparison_numerical.json")
    if not os.path.exists(b1_path):
        print(f"  WARNING: {b1_path} not found. Generating minimal comparison set.")
        return generate_minimal_comparisons()

    with open(b1_path) as f:
        stimuli = json.load(f)
    print(f"  Loaded {len(stimuli)} comparison stimuli from {b1_path}")
    return stimuli


def generate_minimal_comparisons():
    """Generate a minimal set of B1-style comparisons if stimulus files aren't available."""
    rng = np.random.RandomState(SEED)
    baselines = [10, 30, 100, 300, 1000]
    ratios = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
    items = []

    # Number word forms for cross-format comparison
    word_forms = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty",
        30: "thirty", 50: "fifty", 100: "a hundred",
    }

    for base in baselines:
        for ratio in ratios:
            for trial in range(30):  # 30 per cell
                if trial < 20:
                    jitter = rng.uniform(0.85, 1.15)
                    b = int(round(base * jitter))
                else:
                    b = base
                larger = int(round(b * ratio))

                # Create cross-format expressions
                a_expr = str(b)
                b_expr = str(larger)

                # Randomise which is A vs B
                if rng.random() < 0.5:
                    items.append({
                        "baseline": base, "ratio": ratio,
                        "smaller": b, "larger": larger,
                        "a_value": b, "b_value": larger,
                        "a_expression": a_expr, "b_expression": b_expr,
                        "correct": "B"
                    })
                else:
                    items.append({
                        "baseline": base, "ratio": ratio,
                        "smaller": b, "larger": larger,
                        "a_value": larger, "b_value": b,
                        "a_expression": b_expr, "b_expression": a_expr,
                        "correct": "A"
                    })
    return items


def run_paradigm_b_numerical(model, tokenizer):
    """Run B1 cross-format comparison on numerical stimuli."""
    print("\n--- Paradigm B: Behavioural discrimination (numerical) ---")

    # Try to load from existing stimulus directory
    stimuli_dir = os.path.join(os.path.dirname(__file__), "stimuli")
    if not os.path.exists(stimuli_dir):
        stimuli_dir = r"C:\weber\stimuli"  # JP's local path
    items = load_comparison_stimuli(stimuli_dir) if os.path.exists(stimuli_dir) else generate_minimal_comparisons()

    # Build prompts with chat template
    has_chat_template = hasattr(tokenizer, "apply_chat_template")

    results_items = []
    correct_count = 0
    total = 0

    for idx, item in enumerate(items):
        # Handle both script-generated items (a_expression) and JP's stimulus files (first_presented)
        val_a = item.get("a_expression", str(item.get("first_presented", "")))
        val_b = item.get("b_expression", str(item.get("second_presented", "")))
        correct = item.get("correct", item.get("correct_answer", "B"))

        prompt_text = (
            f"Which represents a larger quantity: "
            f"A) {val_a} or B) {val_b}? "
            f"Answer with only A or B."
        )

        if has_chat_template:
            try:
                messages = [{"role": "user", "content": prompt_text}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = prompt_text
        else:
            formatted = prompt_text

        inputs = tokenizer(formatted, return_tensors="pt")
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]  # last token position

        # Get logits for A and B tokens
        a_token_ids = tokenizer.encode("A", add_special_tokens=False)
        b_token_ids = tokenizer.encode("B", add_special_tokens=False)

        logit_a = logits[a_token_ids[0]].item() if a_token_ids else -float("inf")
        logit_b = logits[b_token_ids[0]].item() if b_token_ids else -float("inf")

        predicted = "A" if logit_a > logit_b else "B"
        is_correct = predicted == correct

        # Entropy of A/B distribution
        probs = torch.softmax(torch.tensor([logit_a, logit_b]), dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        results_items.append({
            "baseline": item.get("baseline", item.get("nominal_baseline")),
            "ratio": item.get("ratio", item.get("nominal_ratio")),
            "correct_answer": correct,
            "predicted": predicted,
            "is_correct": is_correct,
            "logit_a": logit_a,
            "logit_b": logit_b,
            "entropy": entropy,
        })

        if is_correct:
            correct_count += 1
        total += 1

        if (idx + 1) % 200 == 0:
            print(f"  Progress: {idx+1}/{len(items)} ({correct_count/total*100:.1f}% correct)")

    overall_accuracy = correct_count / total if total > 0 else 0
    print(f"\n  B1 Overall accuracy: {overall_accuracy:.3f} ({correct_count}/{total})")

    # Accuracy by ratio
    ratio_acc = {}
    for item in results_items:
        r = item["ratio"]
        if r not in ratio_acc:
            ratio_acc[r] = {"correct": 0, "total": 0}
        ratio_acc[r]["total"] += 1
        if item["is_correct"]:
            ratio_acc[r]["correct"] += 1
    for r in sorted(ratio_acc.keys()):
        acc = ratio_acc[r]["correct"] / ratio_acc[r]["total"]
        ratio_acc[r]["accuracy"] = acc
        print(f"  Ratio {r:.2f}: {acc:.3f} ({ratio_acc[r]['correct']}/{ratio_acc[r]['total']})")

    mean_entropy = np.mean([x["entropy"] for x in results_items])
    print(f"  Mean logit entropy: {mean_entropy:.3f}")

    return {
        "overall_accuracy": overall_accuracy,
        "n_items": total,
        "accuracy_by_ratio": {str(k): v for k, v in ratio_acc.items()},
        "mean_entropy": mean_entropy,
        "items": results_items,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_single_model(model_key, output_dir):
    """Run full exploratory analysis for one model."""
    spec = MODELS[model_key]
    short_name = spec["short_name"]
    model_dir = os.path.join(output_dir, short_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Running exploratory analysis: {spec['hf_id']}")
    print(f"# Output: {model_dir}")
    print(f"{'#'*60}")

    t_start = time.time()

    # Load model
    model, tokenizer, n_layers, d_model = load_model(model_key)
    if model is None:
        return None

    # Record model info
    model_info = {
        "hf_id": spec["hf_id"],
        "short_name": short_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "dtype": str(spec["dtype"]),
        "status": "exploratory (post-registration)",
    }
    save_json(model_info, os.path.join(model_dir, "model_info.json"))

    # ── Paradigm A: Extract ──
    hidden_states, per_carrier = extract_hidden_states(model, tokenizer, n_layers)

    # Save hidden states
    np.savez_compressed(
        os.path.join(model_dir, "hidden_states.npz"),
        **{str(k): v for k, v in hidden_states.items()}
    )

    # ── Paradigm A: Analyse ──
    a_results, cosine_dists, pairs = run_paradigm_a_analysis(hidden_states, n_layers)
    save_json(a_results, os.path.join(model_dir, "paradigm_a_analysis.json"))

    # ── Paradigm C ──
    c_results = run_paradigm_c(hidden_states, n_layers)
    save_json(c_results, os.path.join(model_dir, "paradigm_c_results.json"))

    # ── Paradigm B (numerical only) ──
    b_results = run_paradigm_b_numerical(model, tokenizer)
    save_json(b_results, os.path.join(model_dir, "paradigm_b_numerical.json"))

    # ── Summary ──
    t_elapsed = time.time() - t_start
    summary = {
        "model": spec["hf_id"],
        "h1_supported_cosine": a_results["summary"]["h1_supported_cosine"],
        "h1_supported_euclidean": a_results["summary"]["h1_supported_euclidean"],
        "h1_pass_cosine": a_results["summary"]["h1_pass_cosine"],
        "h1_pass_euclidean": a_results["summary"]["h1_pass_euclidean"],
        "mean_stevens_beta": a_results["summary"]["mean_stevens_beta"],
        "h3_significant_layers": c_results["summary"]["h3_significant_layers"],
        "b1_accuracy": b_results["overall_accuracy"],
        "b1_mean_entropy": b_results["mean_entropy"],
        "runtime_seconds": t_elapsed,
    }
    save_json(summary, os.path.join(model_dir, "summary.json"))

    print(f"\n{'='*60}")
    print(f"COMPLETE: {spec['hf_id']}")
    print(f"  H1 (geometry):  cosine={'PASS' if summary['h1_supported_cosine'] else 'FAIL'}, "
          f"euclidean={'PASS' if summary['h1_supported_euclidean'] else 'FAIL'}")
    print(f"  Stevens β:      {summary['mean_stevens_beta']:.4f}" if summary['mean_stevens_beta'] else "  Stevens β: N/A")
    print(f"  B1 accuracy:    {summary['b1_accuracy']:.3f}")
    print(f"  Runtime:        {t_elapsed:.0f}s")
    print(f"{'='*60}")

    # Unload model to free VRAM
    del model
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Weber 4.2 — Exploratory model generalisation")
    parser.add_argument("--model", choices=["gemma", "qwen", "both"], default="both",
                        help="Which model(s) to run")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: results/exploratory/)")
    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        # Try JP's local path first, fall back to relative
        if os.path.exists(r"C:\weber\results"):
            output_dir = r"C:\weber\results\exploratory"
        else:
            output_dir = os.path.join(os.path.dirname(__file__), "results", "exploratory")
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"Weber 4.2 — Exploratory Model Generalisation")
    print(f"Output: {output_dir}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    models_to_run = ["gemma", "qwen"] if args.model == "both" else [args.model]
    all_summaries = {}

    for model_key in models_to_run:
        summary = run_single_model(model_key, output_dir)
        if summary is not None:
            all_summaries[model_key] = summary

    # Combined summary
    if all_summaries:
        save_json(all_summaries, os.path.join(output_dir, "all_exploratory_summary.json"))
        print(f"\n{'#'*60}")
        print("ALL EXPLORATORY MODELS COMPLETE")
        print(f"{'#'*60}")
        for k, s in all_summaries.items():
            print(f"  {k}: H1={'PASS' if s['h1_supported_cosine'] else 'FAIL'}, "
                  f"B1={s['b1_accuracy']:.3f}, β={s['mean_stevens_beta']:.4f}" if s['mean_stevens_beta'] else f"  {k}: H1={'PASS' if s['h1_supported_cosine'] else 'FAIL'}, B1={s['b1_accuracy']:.3f}")


if __name__ == "__main__":
    main()
