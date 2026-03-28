"""
M3 Paradigm B: Behavioural Discrimination
==========================================
Forced-choice discrimination task: "Which is larger: A) x1 or B) x2?"

For each trial:
  - Present in chat-template format with explicit A/B labels
  - Extract logits for "A" and "B" tokens at the response position
  - Run both orders (counterbalanced) to correct for position bias
  - Record: chosen option, P(A), P(B), Δlogit, confidence (|Δlogit|)

Follows Weber Paradigm B methodology exactly:
  - Greedy decoding (T=0) with logit extraction
  - Chat template with explicit A/B labels
  - Counterbalanced option order

Usage:
  python m3_discrimination.py --model meta-llama/Meta-Llama-3-8B-Instruct
  python m3_discrimination.py --model meta-llama/Meta-Llama-3-8B-Instruct --precision bfloat16
  python m3_discrimination.py --model meta-llama/Meta-Llama-3-8B --skip-chat-template

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# Constants
# ==============================================================================

STIMULI_PATH = Path('stimuli/m3_discrimination_stimuli.json')
OUTPUT_DIR = Path('discrimination_results')

# System prompt for discrimination (minimal, following Weber)
SYSTEM_PROMPT = "Answer with only A or B."


def get_ab_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """
    Get token IDs for 'A' and 'B' response tokens.
    
    Models may tokenise these differently. We check common variants:
    'A', ' A', 'B', ' B', and use whichever the tokenizer produces.
    """
    # Try encoding just 'A' and 'B' — take the last token in each case
    a_ids = set()
    b_ids = set()
    
    for variant in ['A', ' A', 'a', ' a']:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            a_ids.add(ids[-1])
    
    for variant in ['B', ' B', 'b', ' b']:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            b_ids.add(ids[-1])
    
    # Primary: uppercase without space
    a_primary = tokenizer.encode('A', add_special_tokens=False)[-1]
    b_primary = tokenizer.encode('B', add_special_tokens=False)[-1]
    
    return list(a_ids), list(b_ids), a_primary, b_primary


def build_chat_prompt(tokenizer, question: str, use_chat_template: bool) -> str:
    """Build the full prompt string using the model's chat template."""
    if use_chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Some models (e.g. Gemma) don't support system role —
            # fold system prompt into user message
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
    else:
        # For base models: simple prompt format
        prompt = f"{SYSTEM_PROMPT}\n\nQ: {question}\nA:"
    
    return prompt


def run_single_trial(
    model, 
    tokenizer, 
    prompt: str, 
    a_token_ids: list[int],
    b_token_ids: list[int],
    a_primary: int,
    b_primary: int,
    device: str,
) -> dict:
    """
    Run a single discrimination trial.
    
    Returns:
        dict with logit_a, logit_b, prob_a, prob_b, chosen, delta_logit, confidence
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits at the last position (where the model would generate next token)
    last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
    
    # Extract logits for A and B tokens (take max across variants)
    logit_a = max(last_logits[tid].item() for tid in a_token_ids)
    logit_b = max(last_logits[tid].item() for tid in b_token_ids)
    
    # Compute probabilities via softmax over just A and B
    # (This is the 2AFC decision variable)
    logits_ab = torch.tensor([logit_a, logit_b])
    probs_ab = torch.softmax(logits_ab, dim=0)
    prob_a = probs_ab[0].item()
    prob_b = probs_ab[1].item()
    
    # Decision: greedy (higher logit wins)
    chosen = 'A' if logit_a > logit_b else 'B'
    
    # Delta logit (A - B): positive favours A
    delta_logit = logit_a - logit_b
    
    # Confidence: unsigned magnitude of evidence
    confidence = abs(delta_logit)
    
    # Also get NLP of chosen option (for meta-d' Type-2 analysis)
    # NLP = -log(P(chosen))
    if chosen == 'A':
        nlp_chosen = -math.log(max(prob_a, 1e-10))
    else:
        nlp_chosen = -math.log(max(prob_b, 1e-10))
    
    return {
        'logit_a': round(logit_a, 4),
        'logit_b': round(logit_b, 4),
        'prob_a': round(prob_a, 6),
        'prob_b': round(prob_b, 6),
        'chosen': chosen,
        'delta_logit': round(delta_logit, 4),
        'confidence': round(confidence, 4),
        'nlp_chosen': round(nlp_chosen, 4),
    }


def run_discrimination(
    model,
    tokenizer,
    stimuli: dict,
    use_chat_template: bool,
    device: str,
) -> list[dict]:
    """
    Run all discrimination trials with counterbalanced order.
    
    For each trial:
      - Order 1: A = smaller, B = larger
      - Order 2: A = larger, B = smaller
      - Compute counterbalanced P(correct) and confidence
    """
    trials = stimuli['trials']
    a_ids, b_ids, a_primary, b_primary = get_ab_token_ids(tokenizer)
    
    print(f"\nA token IDs: {a_ids} (primary: {a_primary})")
    print(f"B token IDs: {b_ids} (primary: {b_primary})")
    print(f"Running {len(trials)} trials × 2 orders = {len(trials) * 2} forward passes\n")
    
    results = []
    t0 = time.time()
    
    for i, trial in enumerate(trials):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(trials) - i - 1) / rate
            print(f"  Trial {i+1}/{len(trials)} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
        
        prompts = trial['prompts']
        
        # Order 1: A = x1 (smaller), B = x2 (larger) → correct = B
        prompt_o1 = build_chat_prompt(tokenizer, prompts['order1']['prompt'], use_chat_template)
        result_o1 = run_single_trial(model, tokenizer, prompt_o1, a_ids, b_ids, a_primary, b_primary, device)
        
        # Order 2: A = x2 (larger), B = x1 (smaller) → correct = A
        prompt_o2 = build_chat_prompt(tokenizer, prompts['order2']['prompt'], use_chat_template)
        result_o2 = run_single_trial(model, tokenizer, prompt_o2, a_ids, b_ids, a_primary, b_primary, device)
        
        # Counterbalanced scoring:
        # In O1, correct = B. Model chose B → correct.
        # In O2, correct = A. Model chose A → correct.
        correct_o1 = 1 if result_o1['chosen'] == prompts['order1']['correct'] else 0
        correct_o2 = 1 if result_o2['chosen'] == prompts['order2']['correct'] else 0
        
        # P(correct) after counterbalancing: average of both orders
        p_correct_cb = (correct_o1 + correct_o2) / 2.0
        
        # Counterbalanced confidence:
        # O1: P(B) = P(larger) when correct = B
        # O2: P(A) = P(larger) when correct = A
        # Average the probability assigned to the correct/larger option
        p_larger_o1 = result_o1['prob_b']  # P(B) in O1, where B = larger
        p_larger_o2 = result_o2['prob_a']  # P(A) in O2, where A = larger
        p_larger_cb = (p_larger_o1 + p_larger_o2) / 2.0
        
        # Counterbalanced delta_logit (evidence for larger option):
        # O1: logit_b - logit_a (B is correct)
        # O2: logit_a - logit_b (A is correct)
        evidence_o1 = -result_o1['delta_logit']  # flip sign: we want evidence for B
        evidence_o2 = result_o2['delta_logit']    # already positive for A
        evidence_cb = (evidence_o1 + evidence_o2) / 2.0
        
        trial_result = {
            'trial_id': trial['trial_id'],
            'x1': trial['x1'],
            'x2': trial['x2'],
            'log_distance': trial['log_distance'],
            'target_log_distance': trial['target_log_distance'],
            'position': trial['position'],
            'position_collapsed': trial['position_collapsed'],
            # Raw results per order
            'order1': {
                'correct': correct_o1,
                **result_o1,
            },
            'order2': {
                'correct': correct_o2,
                **result_o2,
            },
            # Counterbalanced results
            'p_correct_cb': p_correct_cb,
            'p_larger_cb': round(p_larger_cb, 6),
            'evidence_cb': round(evidence_cb, 4),
            'confidence_cb': round(abs(evidence_cb), 4),
        }
        
        results.append(trial_result)
    
    elapsed = time.time() - t0
    print(f"\nCompleted {len(trials)} trials in {elapsed:.1f}s "
          f"({len(trials) * 2 / elapsed:.1f} forward passes/sec)")
    
    return results


def summarise_results(results: list[dict]) -> dict:
    """Compute summary statistics for quick inspection."""
    
    summary = {
        'n_trials': len(results),
        'overall_accuracy': np.mean([r['p_correct_cb'] for r in results]),
    }
    
    # By position (collapsed)
    for pos in ['cross_boundary', 'within_category']:
        subset = [r for r in results if r['position_collapsed'] == pos]
        if subset:
            summary[f'{pos}_accuracy'] = round(np.mean([r['p_correct_cb'] for r in subset]), 4)
            summary[f'{pos}_n'] = len(subset)
            summary[f'{pos}_mean_evidence'] = round(np.mean([r['evidence_cb'] for r in subset]), 4)
            summary[f'{pos}_mean_confidence'] = round(np.mean([r['confidence_cb'] for r in subset]), 4)
    
    # By log-distance level
    log_dists = sorted(set(r['target_log_distance'] for r in results))
    for ld in log_dists:
        subset = [r for r in results if r['target_log_distance'] == ld]
        summary[f'logdist_{ld:.2f}_accuracy'] = round(np.mean([r['p_correct_cb'] for r in subset]), 4)
    
    # By position × log-distance (the full factorial)
    for pos in ['cross_boundary', 'within_category']:
        for ld in log_dists:
            subset = [r for r in results 
                      if r['position_collapsed'] == pos and r['target_log_distance'] == ld]
            if subset:
                key = f'{pos}_logdist_{ld:.2f}_accuracy'
                summary[key] = round(np.mean([r['p_correct_cb'] for r in subset]), 4)
    
    # Position bias check
    o1_choices = [r['order1']['chosen'] for r in results]
    o2_choices = [r['order2']['chosen'] for r in results]
    summary['order1_pct_A'] = round(o1_choices.count('A') / len(o1_choices), 4)
    summary['order2_pct_A'] = round(o2_choices.count('A') / len(o2_choices), 4)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='M3 Paradigm B: Discrimination')
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model ID')
    parser.add_argument('--output-tag', type=str, default=None, 
                        help='Short name for output files (default: derived from model)')
    parser.add_argument('--precision', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--skip-chat-template', action='store_true',
                        help='Use raw prompt format (for base models)')
    parser.add_argument('--stimuli', type=str, default=str(STIMULI_PATH),
                        help='Path to discrimination stimuli JSON')
    args = parser.parse_args()
    
    # Set HSA override for AMD GPUs
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    
    # Derive output tag
    if args.output_tag:
        tag = args.output_tag
    else:
        tag = args.model.split('/')[-1].lower().replace('.', '').replace('-', '_')
    
    print(f"M3 Paradigm B: Discrimination")
    print(f"Model: {args.model}")
    print(f"Output tag: {tag}")
    print(f"Precision: {args.precision}")
    print(f"Chat template: {not args.skip_chat_template}")
    
    # Load stimuli
    with open(args.stimuli) as f:
        stimuli = json.load(f)
    print(f"Loaded {len(stimuli['trials'])} trials")
    
    # Load model
    print(f"\nLoading model...")
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    
    trust_remote = 'phi' in args.model.lower() or 'lexius' in args.model.lower()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_map[args.precision],
        trust_remote_code=trust_remote,
    ).to('cuda')
    model.eval()
    
    device = 'cuda'
    print(f"Model loaded on {device}")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM allocated: {mem:.1f} GB")
    
    # Run discrimination
    results = run_discrimination(
        model, tokenizer, stimuli,
        use_chat_template=not args.skip_chat_template,
        device=str(device),
    )
    
    # Summarise
    summary = summarise_results(results)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY: {tag}")
    print(f"{'='*60}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.4f}")
    print(f"Cross-boundary:   {summary.get('cross_boundary_accuracy', 'N/A')}")
    print(f"Within-category:  {summary.get('within_category_accuracy', 'N/A')}")
    print(f"Position bias O1 %A: {summary['order1_pct_A']:.4f}")
    print(f"Position bias O2 %A: {summary['order2_pct_A']:.4f}")
    
    print(f"\nAccuracy by log-distance:")
    log_dists = sorted(set(r['target_log_distance'] for r in results))
    for ld in log_dists:
        cross_key = f'cross_boundary_logdist_{ld:.2f}_accuracy'
        within_key = f'within_category_logdist_{ld:.2f}_accuracy'
        print(f"  log-dist {ld:.2f}: cross={summary.get(cross_key, 'N/A')}, "
              f"within={summary.get(within_key, 'N/A')}")
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f'm3_discrimination_{tag}.json'
    
    output = {
        'model': args.model,
        'tag': tag,
        'precision': args.precision,
        'chat_template': not args.skip_chat_template,
        'stimuli_path': str(args.stimuli),
        'summary': summary,
        'trials': results,
    }
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
