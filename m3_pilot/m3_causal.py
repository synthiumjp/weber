"""
M3 E5: Causal Intervention — Activation Patching Along Category Direction
==========================================================================
Paper M3, "Classical Minds, Modern Machines" programme.
Author: JP Cacioli
Research assistant: Claude (Anthropic)

Exploratory analysis E5 (pre-registered):
  Does patching along the category direction during discrimination
  change the model's confidence signal?

Methodology (following Weber Paradigm D):
  1. Train a ridge-regression probe on decade_10 RSA centroids predicting
     binary category membership (< 10 vs >= 10). This defines the
     "category direction" v_cat at each layer.
  2. Validate: PCA on centroids, check PC1 alignment with category.
  3. For each discrimination trial, run the forward pass, then re-run
     with activation patching at a target layer: h' = h + alpha * v_cat.
     Four dose levels (alpha in {0.25, 0.5, 0.75, 1.0}).
  4. Measure change in confidence: Delta(|logit_A - logit_B|).
  5. Specificity control: 10 random directions (same norm as v_cat),
     compare |Delta_conf| to category direction.

Key adaptation from Weber:
  - Weber patched along log(magnitude) direction during approximate comparison.
  - M3 patches along category direction during digit comparison.
  - DV is confidence (|Δlogit|) rather than accuracy, because accuracy is
    at ceiling for digit comparison.

Predictions:
  - Patching toward opposite category should change confidence for
    cross-boundary pairs (making them appear same-category, reducing the
    geometric distance → lower confidence).
  - Patching should have less effect on within-category pairs (both
    numbers are on the same side of the boundary already).
  - Effect should be specific to the category direction (larger than
    random direction controls).

Usage:
  # Single model, single layer:
  python m3_causal.py --model meta-llama/Meta-Llama-3-8B-Instruct --layer 16

  # Multiple layers:
  python m3_causal.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
                      --layers 5 10 16 20 24

  # All primary layers:
  python m3_causal.py --model meta-llama/Meta-Llama-3-8B-Instruct --all-primary

Technical note:
  Activation patching requires hooks into the model's forward pass.
  We use PyTorch register_forward_hook to modify hidden states at
  the target layer before they propagate to subsequent layers.

Environment:
  Python 3.12, PyTorch 2.8.0a0 with ROCm 6.4
  AMD RX 7900 GRE (16GB VRAM)
  $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ==============================================================================
# 1. Configuration
# ==============================================================================

EXTRACTION_DIR = Path("extractions")
STIMULI_DIR = Path("stimuli")
DISCRIMINATION_STIMULI = Path("stimuli/m3_discrimination_stimuli.json")
OUTPUT_DIR = Path("causal_results")

SYSTEM_PROMPT = "Answer with only A or B."

DOSE_LEVELS = [0.25, 0.50, 0.75, 1.00]
N_RANDOM_DIRECTIONS = 10

# Model registry — primary layer ranges and layer accessor
MODEL_REGISTRY = {
    "llama3-8b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "precision": "float16",
        "primary_layers": (8, 25),
        "n_layers": 32,
        "layer_accessor": "model.layers",
    },
    "mistral-7b-instruct": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "precision": "float16",
        "primary_layers": (8, 25),
        "n_layers": 32,
        "layer_accessor": "model.layers",
    },
    "gemma2-9b-it": {
        "hf_id": "google/gemma-2-9b-it",
        "precision": "bfloat16",
        "primary_layers": (11, 34),
        "n_layers": 42,
        "layer_accessor": "model.layers",
    },
    "qwen25-7b-instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "precision": "float16",
        "primary_layers": (7, 22),
        "n_layers": 28,
        "layer_accessor": "model.layers",
    },
}


# ==============================================================================
# 2. Probe Training — Define the Category Direction
# ==============================================================================

def load_centroids_for_probe(model_short: str, condition: str = "decade_10"):
    """Load RSA centroids for probe training."""
    npz_path = EXTRACTION_DIR / f"m3_centroids_{condition}_{model_short}.npz"
    meta_path = EXTRACTION_DIR / f"m3_meta_{condition}_{model_short}.json"

    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return data["rsa_centroids"], data["values"], meta


def train_category_probe(
    centroids: np.ndarray,  # (n_values, n_layers, d_model)
    values: np.ndarray,
    boundary: int = 10,
) -> dict:
    """Train a ridge classifier at each layer to predict category membership.

    Category: 0 if value < boundary, 1 if value >= boundary.

    Returns dict with:
      - 'direction': (n_layers, d_model) — the category direction (unit vector)
      - 'accuracy': (n_layers,) — probe accuracy per layer
      - 'coef_norm': (n_layers,) — norm of the probe weights
    """
    n_values, n_layers, d_model = centroids.shape
    labels = (values >= boundary).astype(int)

    directions = np.zeros((n_layers, d_model))
    accuracies = np.zeros(n_layers)
    coef_norms = np.zeros(n_layers)

    for layer in range(n_layers):
        X = centroids[:, layer, :]  # (n_values, d_model)
        y = labels

        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X, y)

        acc = clf.score(X, y)
        coef = clf.coef_[0]  # (d_model,)
        norm = np.linalg.norm(coef)

        # Unit direction
        direction = coef / (norm + 1e-10)

        directions[layer] = direction
        accuracies[layer] = acc
        coef_norms[layer] = norm

    return {
        "direction": directions,
        "accuracy": accuracies,
        "coef_norm": coef_norms,
    }


def validate_direction_pca(
    centroids: np.ndarray,  # (n_values, n_layers, d_model)
    values: np.ndarray,
    directions: np.ndarray,  # (n_layers, d_model)
    boundary: int = 10,
) -> dict:
    """Validate category direction via PCA.

    Check that:
    1. PC1 of the centroids correlates with category membership.
    2. The probe direction aligns with PC1 (cosine similarity).
    """
    from scipy import stats

    n_values, n_layers, d_model = centroids.shape
    labels = (values >= boundary).astype(float)

    pc1_cat_corr = np.zeros(n_layers)
    probe_pc1_cos = np.zeros(n_layers)

    for layer in range(n_layers):
        X = centroids[:, layer, :]

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # PC1 correlation with category
        rho, _ = stats.spearmanr(X_pca[:, 0], labels)
        pc1_cat_corr[layer] = abs(rho)

        # Cosine similarity between probe direction and PC1
        pc1_dir = pca.components_[0]  # (d_model,)
        cos_sim = abs(np.dot(directions[layer], pc1_dir))
        probe_pc1_cos[layer] = cos_sim

    return {
        "pc1_category_correlation": pc1_cat_corr,
        "probe_pc1_cosine": probe_pc1_cos,
    }


# ==============================================================================
# 3. Random Direction Controls
# ==============================================================================

def generate_random_directions(
    d_model: int,
    n_directions: int = N_RANDOM_DIRECTIONS,
    seed: int = SEED,
) -> np.ndarray:
    """Generate random unit vectors for specificity control.

    Returns: (n_directions, d_model)
    """
    rng_local = np.random.default_rng(seed)
    directions = rng_local.standard_normal((n_directions, d_model))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / (norms + 1e-10)


# ==============================================================================
# 4. Activation Patching via Forward Hooks
# ==============================================================================

class ActivationPatcher:
    """Hook-based activation patcher for transformer layers.

    Adds a perturbation along a specified direction to the hidden states
    at a target layer. The perturbation is applied to ALL token positions
    (following Weber) — this is a blunt intervention that tests whether
    the direction is causally relevant, not where in the sequence it matters.

    Usage:
        patcher = ActivationPatcher(model, layer_idx, direction, alpha, model_type)
        patcher.attach()
        # run forward pass
        patcher.remove()
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        direction: np.ndarray,  # (d_model,) unit vector
        alpha: float,  # dose level
        coef_norm: float,  # norm of probe weights (scales the perturbation)
        model_type: str = "llama",  # for accessing the right layer
    ):
        self.model = model
        self.layer_idx = layer_idx
        # Store as numpy; convert to tensor lazily in hook to match dtype/device
        self.direction_np = direction
        self.alpha = alpha
        self.coef_norm = coef_norm
        self.hook_handle = None
        self._perturbation_cache = None

    def _hook_fn(self, module, input, output):
        """Hook function: add perturbation to the layer output.

        HuggingFace transformer layers return either:
          - A tuple: (hidden_states, ...)
          - A BaseModelOutputWithPast with indexable fields
        We modify hidden_states in-place to avoid output format issues.
        """
        # Extract hidden states — always the first element
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output[0]

        # Build perturbation tensor matching hidden_states dtype and device
        if self._perturbation_cache is None:
            pert = torch.tensor(
                self.direction_np * self.alpha * self.coef_norm,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self._perturbation_cache = pert
        perturbation = self._perturbation_cache

        # Modify in-place: add perturbation to all token positions
        # hidden_states may be [seq_len, d_model] or [batch, seq_len, d_model]
        if hidden_states.dim() == 2:
            hidden_states.add_(perturbation.unsqueeze(0))
        else:
            hidden_states.add_(perturbation.unsqueeze(0).unsqueeze(0))

        # Return None — in-place modification means no output replacement needed
        return None

    def attach(self):
        """Register the forward hook on the target layer."""
        self._perturbation_cache = None  # Reset cache for fresh dtype match
        layers = self.model.model.layers
        target_layer = layers[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def remove(self):
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


# ==============================================================================
# 5. Discrimination Trial with Patching
# ==============================================================================

def get_ab_token_ids(tokenizer):
    """Get token IDs for A and B (from m3_discrimination.py)."""
    a_ids = set()
    b_ids = set()

    for variant in ["A", " A", "a", " a"]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            a_ids.add(ids[-1])

    for variant in ["B", " B", "b", " b"]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            b_ids.add(ids[-1])

    a_primary = tokenizer.encode("A", add_special_tokens=False)[-1]
    b_primary = tokenizer.encode("B", add_special_tokens=False)[-1]

    return list(a_ids), list(b_ids), a_primary, b_primary


def build_chat_prompt(tokenizer, question: str):
    """Build chat-templated prompt (from m3_discrimination.py)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


@torch.no_grad()
def run_trial_with_patching(
    model,
    tokenizer,
    prompt: str,
    a_ids: list,
    b_ids: list,
    patcher: Optional[ActivationPatcher] = None,
) -> dict:
    """Run a single discrimination trial, optionally with activation patching.

    Returns dict with logit_a, logit_b, delta_logit, confidence.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    if patcher is not None:
        patcher.attach()

    outputs = model(**inputs)

    if patcher is not None:
        patcher.remove()

    last_logits = outputs.logits[0, -1, :]
    logit_a = max(last_logits[tid].item() for tid in a_ids)
    logit_b = max(last_logits[tid].item() for tid in b_ids)

    delta_logit = logit_a - logit_b
    confidence = abs(delta_logit)

    return {
        "logit_a": round(logit_a, 4),
        "logit_b": round(logit_b, 4),
        "delta_logit": round(delta_logit, 4),
        "confidence": round(confidence, 4),
    }


# ==============================================================================
# 6. Main Causal Experiment
# ==============================================================================

def run_causal_experiment(
    model,
    tokenizer,
    model_short: str,
    probe_data: dict,
    validation: dict,
    target_layers: list,
    n_trials: int = 100,  # subset of discrimination trials
):
    """Run the full causal intervention experiment.

    For each target layer:
      1. Baseline: run trials without patching
      2. Category direction: run trials with patching at 4 dose levels
      3. Random controls: run trials with 10 random directions at dose=1.0
    """
    # Load discrimination stimuli (subset)
    with open(DISCRIMINATION_STIMULI) as f:
        disc_stim = json.load(f)

    trials = disc_stim["trials"][:n_trials]
    a_ids, b_ids, a_primary, b_primary = get_ab_token_ids(tokenizer)

    print(f"\nA token IDs: {a_ids}")
    print(f"B token IDs: {b_ids}")
    print(f"N trials: {len(trials)}")
    print(f"Target layers: {target_layers}")
    print(f"Dose levels: {DOSE_LEVELS}")
    print(f"Random directions: {N_RANDOM_DIRECTIONS}")

    d_model = probe_data["direction"].shape[1]
    random_dirs = generate_random_directions(d_model, N_RANDOM_DIRECTIONS)

    all_layer_results = {}

    for layer_idx in target_layers:
        print(f"\n{'='*50}")
        print(f"Layer {layer_idx}")
        print(f"  Probe accuracy: {probe_data['accuracy'][layer_idx]:.3f}")
        print(f"  PC1-category ρ: {validation['pc1_category_correlation'][layer_idx]:.3f}")
        print(f"  Probe-PC1 cos: {validation['probe_pc1_cosine'][layer_idx]:.3f}")
        print(f"  Coef norm: {probe_data['coef_norm'][layer_idx]:.4f}")
        print(f"{'='*50}")

        cat_direction = probe_data["direction"][layer_idx]
        coef_norm = probe_data["coef_norm"][layer_idx]

        layer_results = {
            "layer": layer_idx,
            "probe_accuracy": float(probe_data["accuracy"][layer_idx]),
            "pc1_category_corr": float(validation["pc1_category_correlation"][layer_idx]),
            "probe_pc1_cos": float(validation["probe_pc1_cosine"][layer_idx]),
            "coef_norm": float(coef_norm),
        }

        # --- Baseline (no patching) ---
        print("  Baseline...", end="", flush=True)
        baseline_results = []
        for trial in trials:
            prompt = build_chat_prompt(
                tokenizer, trial["prompts"]["order1"]["prompt"]
            )
            result = run_trial_with_patching(
                model, tokenizer, prompt, a_ids, b_ids, patcher=None
            )
            result["position"] = trial["position_collapsed"]
            result["crosses_boundary"] = trial["position_collapsed"] == "cross_boundary"
            baseline_results.append(result)
        print(f" done (mean conf = {np.mean([r['confidence'] for r in baseline_results]):.3f})")

        layer_results["baseline"] = {
            "mean_confidence": float(np.mean([r["confidence"] for r in baseline_results])),
            "mean_conf_cross": float(np.mean([
                r["confidence"] for r in baseline_results if r["crosses_boundary"]
            ])) if any(r["crosses_boundary"] for r in baseline_results) else None,
            "mean_conf_within": float(np.mean([
                r["confidence"] for r in baseline_results if not r["crosses_boundary"]
            ])) if any(not r["crosses_boundary"] for r in baseline_results) else None,
        }

        # --- Category direction at each dose ---
        dose_results = {}
        for alpha in DOSE_LEVELS:
            print(f"  Category direction, α={alpha:.2f}...", end="", flush=True)
            patcher = ActivationPatcher(
                model, layer_idx, cat_direction, alpha, coef_norm
            )
            patched_results = []
            for trial in trials:
                prompt = build_chat_prompt(
                    tokenizer, trial["prompts"]["order1"]["prompt"]
                )
                result = run_trial_with_patching(
                    model, tokenizer, prompt, a_ids, b_ids, patcher=patcher
                )
                result["position"] = trial["position_collapsed"]
                result["crosses_boundary"] = trial["position_collapsed"] == "cross_boundary"
                patched_results.append(result)

            mean_conf = np.mean([r["confidence"] for r in patched_results])
            delta_conf = mean_conf - layer_results["baseline"]["mean_confidence"]
            print(f" mean conf = {mean_conf:.3f} (Δ = {delta_conf:+.3f})")

            dose_results[f"alpha_{alpha:.2f}"] = {
                "alpha": alpha,
                "mean_confidence": float(mean_conf),
                "delta_confidence": float(delta_conf),
                "mean_conf_cross": float(np.mean([
                    r["confidence"] for r in patched_results if r["crosses_boundary"]
                ])) if any(r["crosses_boundary"] for r in patched_results) else None,
                "mean_conf_within": float(np.mean([
                    r["confidence"] for r in patched_results if not r["crosses_boundary"]
                ])) if any(not r["crosses_boundary"] for r in patched_results) else None,
            }

        layer_results["category_direction"] = dose_results

        # --- Random direction controls at dose = 1.0 ---
        print(f"  Random directions (n={N_RANDOM_DIRECTIONS})...", end="", flush=True)
        random_results = []
        for rd_idx in range(N_RANDOM_DIRECTIONS):
            patcher = ActivationPatcher(
                model, layer_idx, random_dirs[rd_idx], 1.0, coef_norm
            )
            rd_confs = []
            for trial in trials:
                prompt = build_chat_prompt(
                    tokenizer, trial["prompts"]["order1"]["prompt"]
                )
                result = run_trial_with_patching(
                    model, tokenizer, prompt, a_ids, b_ids, patcher=patcher
                )
                rd_confs.append(result["confidence"])

            mean_rd_conf = np.mean(rd_confs)
            delta_rd = mean_rd_conf - layer_results["baseline"]["mean_confidence"]
            random_results.append({
                "direction_idx": rd_idx,
                "mean_confidence": float(mean_rd_conf),
                "delta_confidence": float(delta_rd),
            })

        mean_random_delta = np.mean([abs(r["delta_confidence"]) for r in random_results])
        cat_delta_at_1 = abs(dose_results["alpha_1.00"]["delta_confidence"])
        specificity = cat_delta_at_1 / (mean_random_delta + 1e-10)

        print(f" mean |Δconf| = {mean_random_delta:.4f}, "
              f"specificity = {specificity:.2f}×")

        layer_results["random_controls"] = {
            "n_directions": N_RANDOM_DIRECTIONS,
            "mean_abs_delta_confidence": float(mean_random_delta),
            "individual": random_results,
        }
        layer_results["specificity_ratio"] = float(specificity)

        # Dose-response monotonicity check
        deltas = [
            dose_results[f"alpha_{a:.2f}"]["delta_confidence"]
            for a in DOSE_LEVELS
        ]
        is_monotonic = all(
            abs(deltas[i]) <= abs(deltas[i + 1])
            for i in range(len(deltas) - 1)
        )
        layer_results["dose_response_monotonic"] = is_monotonic

        all_layer_results[f"layer_{layer_idx}"] = layer_results

    return all_layer_results


# ==============================================================================
# 7. Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="M3 E5: Causal Intervention")
    parser.add_argument(
        "--model-short", type=str, default="llama3-8b-instruct",
        help="Model short name (must be in MODEL_REGISTRY)",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Specific layers to test (e.g., --layers 5 10 16 20 24)",
    )
    parser.add_argument(
        "--all-primary", action="store_true",
        help="Test all primary layers",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Number of discrimination trials to use (default: 100)",
    )
    parser.add_argument(
        "--boundary", type=int, default=10,
        help="Category boundary (default: 10)",
    )
    args = parser.parse_args()

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    if args.model_short not in MODEL_REGISTRY:
        print(f"ERROR: Unknown model '{args.model_short}'")
        print(f"Available: {list(MODEL_REGISTRY.keys())}")
        return

    model_info = MODEL_REGISTRY[args.model_short]

    # Determine target layers
    if args.all_primary:
        l_start, l_end = model_info["primary_layers"]
        target_layers = list(range(l_start, l_end))
    elif args.layers:
        target_layers = args.layers
    else:
        # Default: early, mid, peak-RSA, late (4 representative layers)
        l_start, l_end = model_info["primary_layers"]
        n_layers = model_info["n_layers"]
        early = max(1, l_start - 3)
        mid = (l_start + l_end) // 2
        late = min(n_layers - 1, l_end + 2)
        peak = l_end - 2  # typically peak RSA is near end of primary range
        target_layers = sorted(set([early, l_start, mid, peak, late]))

    print("=" * 70)
    print("M3 E5: Causal Intervention — Activation Patching")
    print(f"Model: {args.model_short} ({model_info['hf_id']})")
    print(f"Target layers: {target_layers}")
    print(f"N trials: {args.n_trials}")
    print(f"Boundary: {args.boundary}")
    print(f"Dose levels: {DOSE_LEVELS}")
    print(f"Random controls: {N_RANDOM_DIRECTIONS}")
    print("=" * 70)

    # Step 1: Load centroids and train probe
    print("\nStep 1: Training category probe...")
    centroids, values, meta = load_centroids_for_probe(args.model_short)
    probe_data = train_category_probe(centroids, values, args.boundary)

    l_start, l_end = model_info["primary_layers"]
    primary_acc = probe_data["accuracy"][l_start:l_end]
    print(f"  Probe accuracy at primary layers: "
          f"mean={np.mean(primary_acc):.3f}, "
          f"min={np.min(primary_acc):.3f}, "
          f"max={np.max(primary_acc):.3f}")

    # Step 2: Validate direction via PCA
    print("\nStep 2: PCA validation...")
    validation = validate_direction_pca(
        centroids, values, probe_data["direction"], args.boundary
    )
    primary_pc1_corr = validation["pc1_category_correlation"][l_start:l_end]
    print(f"  PC1-category correlation at primary layers: "
          f"mean={np.mean(primary_pc1_corr):.3f}")

    # Step 3: Load model
    print(f"\nStep 3: Loading model {model_info['hf_id']}...")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    trust = "phi" in model_info["hf_id"].lower() or "lexius" in model_info["hf_id"].lower()

    tokenizer = AutoTokenizer.from_pretrained(
        model_info["hf_id"], trust_remote_code=trust
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_info["hf_id"],
        dtype=dtype_map.get(model_info["precision"], torch.float16),
        trust_remote_code=trust,
    ).to("cuda")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Step 4: Run causal experiment
    print("\nStep 4: Running causal experiment...")
    t0 = time.time()
    results = run_causal_experiment(
        model, tokenizer, args.model_short,
        probe_data, validation,
        target_layers, n_trials=args.n_trials,
    )
    elapsed = time.time() - t0

    # Step 5: Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model_short,
        "hf_id": model_info["hf_id"],
        "boundary": args.boundary,
        "n_trials": args.n_trials,
        "dose_levels": DOSE_LEVELS,
        "n_random_directions": N_RANDOM_DIRECTIONS,
        "target_layers": target_layers,
        "elapsed_seconds": round(elapsed, 1),
        "layer_results": results,
    }

    out_path = OUTPUT_DIR / f"m3_causal_{args.model_short}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("CAUSAL INTERVENTION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Layer':>6} {'ProbeAcc':>9} {'Cat |Δ|':>9} {'Rand |Δ|':>9} "
          f"{'Specificity':>12} {'Monotonic':>10}")
    print("-" * 62)

    for layer_idx in target_layers:
        key = f"layer_{layer_idx}"
        if key in results:
            r = results[key]
            cat_delta = abs(r["category_direction"]["alpha_1.00"]["delta_confidence"])
            rand_delta = r["random_controls"]["mean_abs_delta_confidence"]
            spec = r["specificity_ratio"]
            mono = "Yes" if r["dose_response_monotonic"] else "No"
            print(f"{layer_idx:>6} {r['probe_accuracy']:>9.3f} "
                  f"{cat_delta:>9.4f} {rand_delta:>9.4f} "
                  f"{spec:>11.2f}× {mono:>10}")

    print(f"\nResults saved: {out_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
