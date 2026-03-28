"""
M3 Identification v3 -- Counterbalanced A/B Forced Choice
==========================================================
Fixes position bias by running each trial twice:
  - Order 1: category_a as option A, category_b as option B
  - Order 2: category_b as option A, category_a as option B
Then averaging P(category_b) across both orders.

This is standard practice for forced-choice psychophysics with
position bias, and matches what Weber ended up doing (Appendix A).

Three framings:
  1. small_large: "Is N small or large?"
  2. single_multi: "Is N a single-digit or multi-digit number?"
  3. digit_count: "Does N have one digit or two digits?"
"""

import json
import torch
import numpy as np
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)

from m3_extract import ExtractionConfig, load_model_and_tokenizer


# =============================================================================
# 1. Identification Framings
# =============================================================================

IDENTIFICATION_FRAMINGS = {
    "small_large": {
        "question": "Is {N} a small number or a large number?",
        "category_a_label": "small",
        "category_b_label": "large",
    },
    "single_multi": {
        "question": "Is {N} a single-digit number or a multi-digit number?",
        "category_a_label": "single-digit",
        "category_b_label": "multi-digit",
    },
    "digit_count": {
        "question": "Does the number {N} have one digit or two digits?",
        "category_a_label": "one digit",
        "category_b_label": "two digits",
    },
}


# =============================================================================
# 2. Build counterbalanced prompts
# =============================================================================

def build_prompt(question: str, option_a: str, option_b: str) -> str:
    """Build A/B forced-choice prompt."""
    return (
        f"{question}\n"
        f"A) {option_a}\n"
        f"B) {option_b}\n"
        f"Answer with the letter A or B only."
    )


# =============================================================================
# 3. Extract A/B logits
# =============================================================================

@torch.no_grad()
def extract_ab_logits(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
) -> dict:
    """Forward pass with chat template, extract logits for A and B tokens."""
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]

    # Get best logit for A and B tokens
    def best_logit(candidates):
        best = float("-inf")
        for c in candidates:
            for tid in tokenizer.encode(c, add_special_tokens=False):
                l = logits[tid].item()
                if l > best:
                    best = l
        return best

    logit_a = best_logit(["A", " A"])
    logit_b = best_logit(["B", " B"])

    # P(B) via softmax over A, B
    max_l = max(logit_a, logit_b)
    ea = np.exp(logit_a - max_l)
    eb = np.exp(logit_b - max_l)
    total = ea + eb

    return {
        "logit_a": float(logit_a),
        "logit_b": float(logit_b),
        "prob_option_a": float(ea / total),
        "prob_option_b": float(eb / total),
    }


# =============================================================================
# 4. Run counterbalanced trial
# =============================================================================

def run_counterbalanced_trial(
    model, tokenizer, question: str,
    cat_a_label: str, cat_b_label: str,
    device: str = "cuda",
) -> dict:
    """Run one identification trial with counterbalanced A/B order.

    Order 1: A=cat_a, B=cat_b  ->  P(B) = P(cat_b)
    Order 2: A=cat_b, B=cat_a  ->  P(A) = P(cat_b)
    Average the two estimates.
    """
    # Order 1: cat_a first
    prompt_1 = build_prompt(question, cat_a_label, cat_b_label)
    r1 = extract_ab_logits(model, tokenizer, prompt_1, device)
    p_catb_order1 = r1["prob_option_b"]  # B = cat_b

    # Order 2: cat_b first
    prompt_2 = build_prompt(question, cat_b_label, cat_a_label)
    r2 = extract_ab_logits(model, tokenizer, prompt_2, device)
    p_catb_order2 = r2["prob_option_a"]  # A = cat_b

    # Average across orders
    p_catb_avg = (p_catb_order1 + p_catb_order2) / 2.0

    return {
        "order1_p_catb": float(p_catb_order1),
        "order2_p_catb": float(p_catb_order2),
        "prob_category_b": float(p_catb_avg),
        "prob_category_a": float(1.0 - p_catb_avg),
        "position_bias": float(p_catb_order2 - p_catb_order1),
        "order1_logit_a": r1["logit_a"],
        "order1_logit_b": r1["logit_b"],
        "order2_logit_a": r2["logit_a"],
        "order2_logit_b": r2["logit_b"],
    }


# =============================================================================
# 5. Main
# =============================================================================

def main():
    config = ExtractionConfig()

    print("=" * 70)
    print("M3 Identification v3 -- Counterbalanced A/B Forced Choice")
    print(f"Framings: {list(IDENTIFICATION_FRAMINGS.keys())}")
    print("=" * 70)

    if not torch.cuda.is_available():
        config.device = "cpu"
        print("WARNING: No GPU, using CPU")

    model, tokenizer = load_model_and_tokenizer(config)

    for condition in config.conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        meta_path = (config.output_dir /
                     f"m3_meta_{condition}_{config.model_short}.json")
        with open(meta_path) as f:
            meta = json.load(f)

        values = meta["values"]
        n_framings = len(IDENTIFICATION_FRAMINGS)
        n_total = len(values) * n_framings
        print(f"  Values: {values}")
        print(f"  Running {n_total} trials x 2 orders = {n_total * 2} forward passes")

        results = []
        count = 0

        for framing_key, spec in IDENTIFICATION_FRAMINGS.items():
            print(f"\n  Framing: {framing_key}")
            print(f"  {'Value':>6s} | {'O1 P(B)':>8s} | {'O2 P(B)':>8s} | "
                  f"{'Avg P(B)':>8s} | {'Bias':>7s} | Category")
            print(f"  {'-'*68}")

            for v in values:
                question = spec["question"].format(N=str(v))
                trial = run_counterbalanced_trial(
                    model, tokenizer, question,
                    spec["category_a_label"], spec["category_b_label"],
                    config.device,
                )

                choice = spec["category_b_label"] if trial["prob_category_b"] > 0.5 \
                    else spec["category_a_label"]

                print(f"  {v:6d} | {trial['order1_p_catb']:.4f}   | "
                      f"{trial['order2_p_catb']:.4f}   | "
                      f"{trial['prob_category_b']:.4f}   | "
                      f"{trial['position_bias']:+.4f}  | {choice}")

                results.append({
                    "value": v,
                    "framing": framing_key,
                    "prob_category_a": trial["prob_category_a"],
                    "prob_category_b": trial["prob_category_b"],
                    "order1_p_catb": trial["order1_p_catb"],
                    "order2_p_catb": trial["order2_p_catb"],
                    "position_bias": trial["position_bias"],
                    "raw_logit_a": trial["order1_logit_a"],
                    "raw_logit_b": trial["order1_logit_b"],
                    "option_a_label": spec["category_a_label"],
                    "option_b_label": spec["category_b_label"],
                })
                count += 1

        # Update metadata
        meta["identification_results"] = results
        meta["identification_method"] = "chat_template_ab_counterbalanced"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  Updated: {meta_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done. Rerun m3_pilot_analysis.py to regenerate figures.")
    print("=" * 70)


if __name__ == "__main__":
    main()
