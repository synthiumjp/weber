"""
Weber's Law Project 4.2 — Paradigm B: Behavioural Magnitude Discrimination
Classical Minds, Modern Machines

Runs all comparison prompts (B1, B2, B3, symbolic control), extracts logits,
fits psychometric functions, estimates Weber fractions.

Pre-registration ref: v2.7 Sections 5.4, 4.1 (H2), 4.2 (H4, H6), 14 (F2, F3).

Usage:
    python paradigm_b_behaviour.py --model llama_instruct --domain numerical
    python paradigm_b_behaviour.py --model llama_instruct --domain all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import chi2, spearmanr
from scipy.special import ndtr  # cumulative normal

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS, DOMAINS, RESULTS_DIR, STIMULI_DIR,
    BONFERRONI_ALPHA, SECONDARY_ALPHA,
    BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_model(model_key: str):
    """Load model and tokenizer in FP16 on GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

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
    model.eval()
    return model, tokenizer


def load_comparison_stimuli(domain_key: str, stimuli_dir: Path) -> list[dict]:
    """
    Load pre-generated comparison stimuli from archived JSON files.

    Handles the actual stimulus file format from stimuli_generation.py:
        prompts_b1.json — B1 cross-format comparison (PRIMARY for H2)
        prompts_b2.json — B2 approximate arithmetic
        prompts_b3.json — B3 contextual comparison
        prompts_symbolic_control.json — symbolic comparison control

    Fields in the archive:
        pair_id, task, prompt, a_expression, b_expression,
        correct_answer, nominal_ratio, nominal_baseline

    We normalise field names to what the analysis pipeline expects.
    """
    # Map task types to filenames
    task_files = {
        "B1": stimuli_dir / "prompts_b1.json",
        "B2": stimuli_dir / "prompts_b2.json",
        "B3": stimuli_dir / "prompts_b3.json",
        "symbolic": stimuli_dir / "prompts_symbolic_control.json",
    }

    all_stimuli = []

    for task_type, path in task_files.items():
        if not path.exists():
            log.warning(f"Stimulus file not found: {path}")
            continue

        with open(path) as f:
            raw = json.load(f)

        for item in raw:
            # Fix prompt formatting: ensure A/B labels are explicit
            # The stimuli_generation script produces prompts like:
            #   "Which represents a larger quantity: X or Y? Answer with only A or B."
            # But the model needs explicit labels:
            #   "Which represents a larger quantity: A) X or B) Y? Answer with only A or B."
            prompt = item.get("prompt", "")
            a_expr = item.get("a_expression", "")
            b_expr = item.get("b_expression", "")

            # Check if prompt already has A)/B) labels
            if "A)" not in prompt and "B)" not in prompt and a_expr and b_expr:
                # Rebuild prompt with explicit A/B labels
                if task_type == "B1":
                    prompt = (
                        f"Which represents a larger quantity: "
                        f"A) {a_expr} or B) {b_expr}? "
                        f"Answer with only A or B."
                    )
                elif task_type == "B2":
                    prompt = (
                        f"Without calculating exactly, which is larger: "
                        f"A) {a_expr} or B) {b_expr}? "
                        f"Answer with only A or B."
                    )
                elif task_type == "B3":
                    # B3 is contextual — keep original but ensure labels
                    # Try to inject A/B before expressions in existing prompt
                    prompt = prompt.replace(
                        f"{a_expr}", f"A) {a_expr}", 1
                    ).replace(
                        f"{b_expr}", f"B) {b_expr}", 1
                    )
                elif task_type == "symbolic":
                    prompt = (
                        f"Which is larger: A) {a_expr} or B) {b_expr}? "
                        f"Answer with only A or B."
                    )

            # Normalise field names
            normalised = {
                "task_type": item.get("task", task_type),
                "domain": domain_key,
                "baseline": float(item.get("nominal_baseline", 0)),
                "ratio": float(item.get("nominal_ratio", 1)),
                "value_a": a_expr,
                "value_b": b_expr,
                "correct_answer": item.get("correct_answer", ""),
                "prompt": prompt,
                "pair_id": item.get("pair_id", ""),
            }
            all_stimuli.append(normalised)

        log.info(f"Loaded {len(raw)} {task_type} stimuli from {path.name}")

    if not all_stimuli:
        log.warning("No archived stimuli found. Generating on-the-fly from spec.")
        return generate_comparison_stimuli(domain_key)

    return all_stimuli


def generate_comparison_stimuli(domain_key: str) -> list[dict]:
    """
    Generate comparison stimuli from the pre-registration specification.

    v2.7 Section 5.2.1 (numerical):
    5 baselines (10, 30, 100, 300, 1000) × 6 ratios (1.05, 1.10, 1.20, 1.50, 2.00, 3.00)
    × 50 pairs per cell = 1,500 pairs.
    30 of 50 pairs per cell use jittered baselines (±15%, seed 42).
    """
    rng = np.random.default_rng(42)  # Locked seed

    if domain_key == "numerical":
        baselines = [10, 30, 100, 300, 1000]
        ratios = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
        n_pairs = 50
        n_jittered = 30
    elif domain_key == "temporal":
        baselines = [10, 120, 900, 7200, 86400]  # seconds
        ratios = [1.10, 1.20, 1.50, 2.00, 3.00, 5.00]
        n_pairs = 30
        n_jittered = 18
    elif domain_key == "spatial":
        baselines = [10, 200, 5000, 50000, 500000]  # metres
        ratios = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
        n_pairs = 30
        n_jittered = 18
    else:
        raise ValueError(f"Unknown domain: {domain_key}")

    stimuli = []

    for baseline in baselines:
        for ratio in ratios:
            comparison = baseline * ratio

            for pair_idx in range(n_pairs):
                # Jitter (v2.7: "30 of 50 pairs per cell use jittered baselines ±15%")
                if pair_idx < n_jittered:
                    jitter = rng.uniform(-0.15, 0.15)
                    b_actual = baseline * (1 + jitter)
                    c_actual = b_actual * ratio
                else:
                    b_actual = baseline
                    c_actual = comparison

                # Counterbalanced order
                if pair_idx % 2 == 0:
                    a_val, b_val = b_actual, c_actual
                    correct = "B"
                else:
                    a_val, b_val = c_actual, b_actual
                    correct = "A"

                # B1 prompt (cross-format, PRIMARY for H2)
                prompt_b1 = (
                    f"Which represents a larger quantity: "
                    f"{_format_number_word(a_val)} or {_format_number_word(b_val)}? "
                    f"Answer with only A or B."
                )

                stimuli.append({
                    "task_type": "B1",
                    "domain": domain_key,
                    "baseline": float(baseline),
                    "ratio": float(ratio),
                    "value_a": float(a_val),
                    "value_b": float(b_val),
                    "correct_answer": correct,
                    "prompt": prompt_b1,
                    "pair_idx": pair_idx,
                })

    log.info(f"Generated {len(stimuli)} B1 stimuli for {domain_key}")
    return stimuli


def _format_number_word(n: float) -> str:
    """Convert number to word form for cross-format comparison (B1)."""
    # Simple word forms for common numbers
    n_int = int(round(n))
    word_map = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
        15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
        19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
        50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty",
        90: "ninety", 100: "one hundred", 200: "two hundred",
        300: "three hundred", 500: "five hundred", 1000: "one thousand",
    }

    if n_int in word_map:
        return word_map[n_int]

    # For jittered values, use mixed format (e.g., "about thirty-four")
    return f"about {n_int}"


def run_single_comparison(
    model,
    tokenizer,
    prompt: str,
    use_chat_template: bool = True,
) -> dict:
    """
    Run a single comparison prompt. Greedy decoding (T=0).

    v2.7 Section 5.4:
    "Greedy decoding (T=0). Extract logit for each candidate answer token.
    Score as correct if p(correct) > p(incorrect).
    Confidence = log(p(correct)/p(incorrect))."

    For instruct models, wraps the prompt in the model's chat template
    so the model generates a response rather than continuing the prompt.

    Also computes logit-entropy diagnostic.
    """
    import torch

    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits at the last (answer) position
    logits = outputs.logits[0, -1, :].float().cpu()

    # Get token IDs for "A" and "B"
    # Try multiple token representations
    a_candidates = [tokenizer.encode("A", add_special_tokens=False),
                    tokenizer.encode(" A", add_special_tokens=False)]
    b_candidates = [tokenizer.encode("B", add_special_tokens=False),
                    tokenizer.encode(" B", add_special_tokens=False)]

    # Use the first token of each encoding
    a_ids = [ids[0] for ids in a_candidates if ids]
    b_ids = [ids[0] for ids in b_candidates if ids]

    # Take max logit across candidate tokenisations
    a_logit = max(logits[tid].item() for tid in a_ids) if a_ids else float("-inf")
    b_logit = max(logits[bid].item() for bid in b_ids) if b_ids else float("-inf")

    # Softmax for probabilities
    probs = torch.softmax(logits, dim=0)
    a_prob = max(probs[tid].item() for tid in a_ids) if a_ids else 0
    b_prob = max(probs[bid].item() for bid in b_ids) if b_ids else 0

    # Predicted answer
    predicted = "A" if a_logit > b_logit else "B"

    # Confidence (log odds)
    if a_prob > 0 and b_prob > 0:
        confidence = np.log(max(a_prob, b_prob) / min(a_prob, b_prob))
    else:
        confidence = float("inf")

    # Entropy diagnostic (v2.7 Section 5.4)
    # "Shannon entropy of the full logit distribution at the answer position"
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

    # Greedy decoded token (for non-answer detection)
    greedy_token_id = logits.argmax().item()
    greedy_token = tokenizer.decode([greedy_token_id]).strip()

    return {
        "predicted": predicted,
        "a_logit": float(a_logit),
        "b_logit": float(b_logit),
        "a_prob": float(a_prob),
        "b_prob": float(b_prob),
        "confidence": float(confidence),
        "entropy": float(entropy),
        "greedy_token": greedy_token,
        "is_valid_answer": greedy_token in ["A", "B"],
    }


def run_all_comparisons(
    model,
    tokenizer,
    stimuli: list[dict],
    model_key: str,
) -> list[dict]:
    """Run all comparison stimuli and collect results."""
    log.info(f"Running {len(stimuli)} comparisons...")
    results = []
    n_valid = 0
    n_correct = 0

    for idx, stim in enumerate(stimuli):
        t0 = time.time()
        response = run_single_comparison(model, tokenizer, stim["prompt"])
        elapsed = time.time() - t0

        correct = response["predicted"] == stim["correct_answer"]
        if correct:
            n_correct += 1
        if response["is_valid_answer"]:
            n_valid += 1

        result = {**stim, **response, "correct": correct, "elapsed_s": round(elapsed, 4)}
        results.append(result)

        if (idx + 1) % 100 == 0:
            log.info(
                f"  {idx+1}/{len(stimuli)}: "
                f"accuracy={n_correct/(idx+1):.3f}, "
                f"valid={n_valid/(idx+1):.3f}"
            )

    log.info(
        f"Completed: {n_correct}/{len(stimuli)} correct ({n_correct/len(stimuli):.3f}), "
        f"{n_valid}/{len(stimuli)} valid answers"
    )
    return results


# ── Psychometric function fitting (v2.7 Section 7) ──

def fit_psychometric(
    ratios: np.ndarray,
    accuracy: np.ndarray,
    n_trials: np.ndarray,
) -> dict:
    """
    Fit cumulative Gaussian psychometric function via MLE.

    v2.7: "Cumulative Gaussian via MLE. Threshold and slope with 95% BCa
    bootstrap CIs (10,000 iterations, seed 42)."

    P(correct) = λ + (1 - λ - γ) * Φ((x - μ) / σ)
    where λ = lapse rate, γ = guess rate (0.5 for 2AFC), μ = threshold, σ = slope
    """
    def psychometric(x, mu, sigma, lapse):
        gamma = 0.5  # 2AFC guess rate
        return lapse + (1 - lapse - gamma) * ndtr((x - mu) / sigma)

    # Negative log-likelihood for binomial data
    def neg_ll(params, x, k, n):
        mu, sigma, lapse = params
        sigma = max(sigma, 1e-6)
        lapse = np.clip(lapse, 0, 0.1)
        p = psychometric(x, mu, sigma, lapse)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
        return -ll

    log_ratios = np.log(ratios)
    k = np.round(accuracy * n_trials).astype(int)

    # Initial guesses
    mu0 = np.median(log_ratios)
    sigma0 = np.std(log_ratios)
    lapse0 = 0.01

    try:
        result = minimize(
            neg_ll,
            x0=[mu0, sigma0, lapse0],
            args=(log_ratios, k, n_trials),
            method="Nelder-Mead",
            options={"maxiter": 10000},
        )
        mu, sigma, lapse = result.x
        sigma = max(sigma, 1e-6)
        lapse = np.clip(lapse, 0, 0.1)

        # Weber fraction: ratio at 75% correct threshold
        # P(correct) = 0.75 → solve for x
        target = 0.75
        from scipy.optimize import brentq
        try:
            def threshold_eq(x):
                return psychometric(x, mu, sigma, lapse) - target
            x_75 = brentq(threshold_eq, log_ratios.min() - 2, log_ratios.max() + 2)
            weber_fraction = np.exp(x_75) - 1  # Convert log-ratio to Weber fraction
        except Exception:
            weber_fraction = None

        return {
            "mu": float(mu),
            "sigma": float(sigma),
            "lapse": float(lapse),
            "weber_fraction": float(weber_fraction) if weber_fraction is not None else None,
            "converged": result.success,
            "neg_loglik": float(result.fun),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def bootstrap_weber_fraction(
    results_by_cell: dict,
    n_boot: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    BCa bootstrap CIs for Weber fractions.

    v2.7: "95% BCa bootstrap CIs (10,000 iterations, seed 42)."
    """
    rng = np.random.default_rng(seed)
    all_wf = []

    for baseline, ratio_data in results_by_cell.items():
        ratios = np.array(sorted(ratio_data.keys()))
        acc = np.array([ratio_data[r]["accuracy"] for r in ratios])
        n_trials = np.array([ratio_data[r]["n_trials"] for r in ratios])

        boot_wfs = []
        for _ in range(n_boot):
            # Resample trials within each ratio level
            boot_acc = np.zeros_like(acc)
            for i, n in enumerate(n_trials):
                if n > 0:
                    boot_correct = rng.binomial(int(n), acc[i]) / n
                    boot_acc[i] = boot_correct
                else:
                    boot_acc[i] = 0.5

            fit = fit_psychometric(ratios, boot_acc, n_trials)
            if fit.get("weber_fraction") is not None:
                boot_wfs.append(fit["weber_fraction"])

        if boot_wfs:
            boot_wfs = np.array(boot_wfs)
            # BCa: bias-corrected and accelerated
            # Simplified: use percentile method (full BCa requires jackknife)
            ci_lo = float(np.percentile(boot_wfs, 2.5))
            ci_hi = float(np.percentile(boot_wfs, 97.5))
            all_wf.append({
                "baseline": float(baseline),
                "weber_fraction_median": float(np.median(boot_wfs)),
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
            })

    return all_wf


# ── H2 evaluation (v2.7 Section 4.1) ──

def evaluate_h2(results: list[dict], task_type: str = "B1") -> dict:
    """
    H2 (v2.7): "Logistic regression on log(ratio) explains significantly
    more variance than logistic regression on absolute difference
    (∆ deviance test, p < 0.017)."
    """
    # Filter to task type
    items = [r for r in results if r.get("task_type") == task_type and r.get("is_valid_answer", True)]

    if not items:
        return {"status": "no_valid_items"}

    correct = np.array([int(r["correct"]) for r in items])
    log_ratios = np.array([np.log(r["ratio"]) for r in items])
    # Absolute difference computed from baseline × ratio - baseline
    # (value_a/value_b may be string expressions in cross-format task)
    abs_diffs = np.array([r["baseline"] * (r["ratio"] - 1) for r in items])

    # Logistic regression: correct ~ log(ratio)
    from numpy.linalg import lstsq

    def logistic_deviance(X, y):
        """Fit logistic regression and return deviance."""
        from scipy.optimize import minimize

        def neg_ll(beta):
            z = X @ beta
            z = np.clip(z, -20, 20)
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        k = X.shape[1]
        res = minimize(neg_ll, x0=np.zeros(k), method="BFGS")
        return 2 * res.fun, res.x  # deviance = 2 * neg_ll

    n = len(correct)

    # Model 1: correct ~ intercept + log(ratio)
    X_ratio = np.column_stack([np.ones(n), log_ratios])
    dev_ratio, beta_ratio = logistic_deviance(X_ratio, correct)

    # Model 2: correct ~ intercept + abs_diff
    X_diff = np.column_stack([np.ones(n), abs_diffs])
    dev_diff, beta_diff = logistic_deviance(X_diff, correct)

    # Null model: correct ~ intercept
    X_null = np.ones((n, 1))
    dev_null, _ = logistic_deviance(X_null, correct)

    # Delta deviance test: ratio model vs diff model
    # H2: ratio model should have lower deviance
    delta_dev = dev_diff - dev_ratio
    # Under H0 (both models equivalent), Δdev ~ χ²(0) — but we compare nested models
    # More precisely: both have 2 params, so we compare directly
    # Use likelihood ratio test against null for each
    lr_ratio = dev_null - dev_ratio  # χ²(1) under H0
    lr_diff = dev_null - dev_diff

    p_ratio = 1 - chi2.cdf(lr_ratio, df=1) if lr_ratio > 0 else 1.0
    p_diff = 1 - chi2.cdf(lr_diff, df=1) if lr_diff > 0 else 1.0

    # For H2: test that ratio model is significantly better than diff model
    # Vuong-type comparison (non-nested): simplified as deviance difference
    # Positive delta_dev means ratio model fits better
    p_comparison = 1 - chi2.cdf(abs(delta_dev), df=1) if delta_dev != 0 else 1.0

    return {
        "task_type": task_type,
        "n_items": n,
        "accuracy": float(correct.mean()),
        "deviance_ratio_model": float(dev_ratio),
        "deviance_diff_model": float(dev_diff),
        "deviance_null": float(dev_null),
        "delta_deviance": float(delta_dev),
        "p_ratio_vs_null": float(p_ratio),
        "p_diff_vs_null": float(p_diff),
        "p_comparison": float(p_comparison),
        "ratio_model_better": bool(delta_dev > 0),
        "h2_passes": bool(delta_dev > 0 and p_comparison < BONFERRONI_ALPHA),
        "beta_log_ratio": float(beta_ratio[1]),
        "beta_abs_diff": float(beta_diff[1]),
    }


# ── Logit-entropy diagnostic (v2.7 Section 5.4) ──

def entropy_diagnostic(results: list[dict]) -> dict:
    """
    v2.7: "If entropy < 0.1 nats for >50% of items in a task type, that task
    is flagged as engaging exact rather than approximate processing."
    """
    by_task = {}
    for r in results:
        task = r.get("task_type", "unknown")
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(r.get("entropy", 0))

    diagnostics = {}
    for task, entropies in by_task.items():
        ent = np.array(entropies)
        low_entropy = np.sum(ent < 0.1) / len(ent)
        # Spearman: entropy vs ratio
        items = [r for r in results if r.get("task_type") == task]
        ratios = np.array([r["ratio"] for r in items])
        rho, p = spearmanr(ent, ratios)

        diagnostics[task] = {
            "mean_entropy": float(ent.mean()),
            "median_entropy": float(np.median(ent)),
            "fraction_below_0.1": float(low_entropy),
            "flagged_exact": bool(low_entropy > 0.5),
            "entropy_ratio_spearman_rho": float(rho),
            "entropy_ratio_spearman_p": float(p),
        }

    return diagnostics


# ── Aggregate by cell ──

def aggregate_by_cell(results: list[dict], task_type: str = "B1") -> dict:
    """Aggregate accuracy by baseline × ratio cell."""
    cells = {}
    for r in results:
        if r.get("task_type") != task_type:
            continue
        baseline = r["baseline"]
        ratio = r["ratio"]
        key = (baseline, ratio)
        if key not in cells:
            cells[key] = {"correct": 0, "total": 0}
        cells[key]["total"] += 1
        if r["correct"]:
            cells[key]["correct"] += 1

    # Reorganise by baseline
    by_baseline = {}
    for (baseline, ratio), counts in cells.items():
        if baseline not in by_baseline:
            by_baseline[baseline] = {}
        by_baseline[baseline][ratio] = {
            "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0,
            "n_trials": counts["total"],
            "n_correct": counts["correct"],
        }

    return by_baseline


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Paradigm B: Behavioural Discrimination")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--domain", required=True, choices=list(DOMAINS.keys()) + ["all"])
    parser.add_argument("--stimuli-dir", type=Path, default=STIMULI_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    domains = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]

    # Load model once
    model, tokenizer = load_model(args.model)

    for domain_key in domains:
        log.info(f"\n{'='*60}")
        log.info(f"Paradigm B: {args.model} / {domain_key}")
        log.info(f"{'='*60}")

        # Load or generate stimuli
        stimuli = load_comparison_stimuli(domain_key, args.stimuli_dir)

        # Run comparisons
        results = run_all_comparisons(model, tokenizer, stimuli, args.model)

        # Aggregate
        cells = aggregate_by_cell(results, task_type="B1")

        # Fit psychometric functions per baseline
        log.info("\nFitting psychometric functions...")
        psychometric_fits = {}
        for baseline, ratio_data in cells.items():
            ratios = np.array(sorted(ratio_data.keys()))
            acc = np.array([ratio_data[r]["accuracy"] for r in ratios])
            n_trials = np.array([ratio_data[r]["n_trials"] for r in ratios])

            fit = fit_psychometric(ratios, acc, n_trials)
            psychometric_fits[float(baseline)] = fit
            wf = fit.get("weber_fraction")
            log.info(f"  Baseline {baseline}: Weber fraction = {wf:.4f}" if wf else
                     f"  Baseline {baseline}: fit failed")

        # H2 evaluation
        log.info("\nEvaluating H2...")
        h2 = evaluate_h2(results, task_type="B1")
        log.info(
            f"  H2: delta_deviance={h2.get('delta_deviance', 'N/A'):.2f}, "
            f"p={h2.get('p_comparison', 'N/A'):.4f}, "
            f"{'PASS' if h2.get('h2_passes') else 'FAIL'}"
        )

        # Entropy diagnostic
        entropy_diag = entropy_diagnostic(results)
        for task, diag in entropy_diag.items():
            log.info(
                f"  Entropy ({task}): mean={diag['mean_entropy']:.3f}, "
                f"{'FLAGGED EXACT' if diag['flagged_exact'] else 'approximate'}"
            )

        # Bootstrap Weber fractions
        log.info("\nBootstrapping Weber fraction CIs...")
        boot_wf = bootstrap_weber_fraction(cells)
        for wf_info in boot_wf:
            log.info(
                f"  Baseline {wf_info['baseline']}: "
                f"WF={wf_info['weber_fraction_median']:.4f} "
                f"[{wf_info['ci_95_lo']:.4f}, {wf_info['ci_95_hi']:.4f}]"
            )

        # Ceiling check (v2.7: ">10% of cells show accuracy >98% at ratio 1.05-1.20")
        fine_ratios = [r for r in [1.05, 1.10, 1.20] if r in
                       [ratio for bl in cells.values() for ratio in bl.keys()]]
        n_ceiling = 0
        n_fine_cells = 0
        for bl, rd in cells.items():
            for r in fine_ratios:
                if r in rd:
                    n_fine_cells += 1
                    if rd[r]["accuracy"] > 0.98:
                        n_ceiling += 1
        ceiling_limited = n_ceiling / n_fine_cells > 0.10 if n_fine_cells > 0 else False

        # Save results
        out_dir = args.results_dir / "paradigm_b" / args.model / domain_key
        out_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "model_key": args.model,
            "domain_key": domain_key,
            "n_stimuli": len(stimuli),
            "n_results": len(results),
            "overall_accuracy": float(np.mean([r["correct"] for r in results])),
            "cells": {str(k): v for k, v in cells.items()},
            "psychometric_fits": psychometric_fits,
            "h2_evaluation": h2,
            "entropy_diagnostic": entropy_diag,
            "bootstrap_weber_fractions": boot_wf,
            "ceiling_check": {
                "n_ceiling_cells": n_ceiling,
                "n_fine_cells": n_fine_cells,
                "ceiling_limited": ceiling_limited,
            },
        }

        with open(out_dir / "paradigm_b_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)

        # Save raw results for figure generation
        with open(out_dir / "paradigm_b_raw.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        log.info(f"\nResults saved to {out_dir}")

    log.info("\n=== Paradigm B complete ===")


if __name__ == "__main__":
    main()
