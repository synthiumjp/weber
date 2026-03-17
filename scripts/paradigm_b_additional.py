#!/usr/bin/env python3
"""
Paradigm B: Run B2, B3, and Symbolic Control Tasks
Weber's Law in Transformer Magnitude Representations (Project 4.2)

B1 (cross-format, primary for H2) was run in the initial session.
This script runs the three remaining task types:
- B2: Approximate arithmetic comparison
- B3: Contextual comparison  
- Symbolic control: Exact symbolic comparison (ceiling prediction)

Pre-reg: "Tasks B2 and B3 are secondary replications."
Pre-reg: "Symbolic comparison control... predicted at ceiling with no ratio effect."

Author: JP Cacioli
Date: March 2026
"""

import json
import logging
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


LLAMA_CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
MISTRAL_CHAT_TEMPLATE = "<s>[INST] {prompt} [/INST]"


def get_chat_template(model_key):
    if "llama" in model_key and "base" not in model_key:
        return LLAMA_CHAT_TEMPLATE
    elif "mistral" in model_key:
        return MISTRAL_CHAT_TEMPLATE
    elif "base" in model_key:
        return "{prompt}"  # No template for base models
    else:
        raise ValueError(f"Unknown model: {model_key}")


def load_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODELS[model_key]
    hf_id = cfg["hf_id"]
    log.info(f"Loading {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16).to("cuda")
    model.eval()
    return model, tokenizer


def score_ab_prompt(model, tokenizer, prompt_text, correct_answer, model_key, device="cuda"):
    """Score a prompt with A/B answer format.
    
    Returns dict with predicted, a_prob, b_prob, correct, entropy, etc.
    """
    template = get_chat_template(model_key)
    formatted = template.format(prompt=prompt_text)
    
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]  # (vocab_size,)
    
    id_a = tokenizer.encode("A", add_special_tokens=False)[0]
    id_b = tokenizer.encode("B", add_special_tokens=False)[0]
    
    a_logit = logits[id_a].float().item()
    b_logit = logits[id_b].float().item()
    
    # Softmax over A/B only
    logits_ab = torch.tensor([a_logit, b_logit])
    probs_ab = torch.softmax(logits_ab, dim=0)
    a_prob = probs_ab[0].item()
    b_prob = probs_ab[1].item()
    
    predicted = "A" if a_prob > b_prob else "B"
    correct_bool = (predicted == correct_answer)
    confidence = abs(a_logit - b_logit)
    
    # Full vocabulary entropy
    full_probs = torch.softmax(logits.float(), dim=0)
    entropy = -torch.sum(full_probs * torch.log(full_probs + 1e-10)).item()
    
    # Greedy token (for checking if model produces A/B)
    greedy_id = logits.argmax().item()
    greedy_token = tokenizer.decode([greedy_id]).strip()
    is_valid = greedy_token in ["A", "B"]
    
    return {
        "predicted": predicted,
        "a_logit": a_logit,
        "b_logit": b_logit,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "confidence": confidence,
        "entropy": entropy,
        "greedy_token": greedy_token,
        "is_valid_answer": is_valid,
        "correct": correct_bool,
    }


def score_symbolic_prompt(model, tokenizer, prompt_data, model_key, device="cuda"):
    """Score a symbolic comparison prompt ("Which is larger, A or B? Answer with only the larger number.")
    
    The correct answer is a number, not A/B. We check which number token gets higher probability.
    """
    template = get_chat_template(model_key)
    formatted = template.format(prompt=prompt_data["prompt"])
    
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]
    
    first = str(prompt_data["first_presented"])
    second = str(prompt_data["second_presented"])
    correct_str = str(prompt_data["correct_answer"])
    
    # Get token IDs for both numbers (first token of each)
    id_first = tokenizer.encode(first, add_special_tokens=False)[0]
    id_second = tokenizer.encode(second, add_special_tokens=False)[0]
    
    logit_first = logits[id_first].float().item()
    logit_second = logits[id_second].float().item()
    
    probs = torch.softmax(torch.tensor([logit_first, logit_second]), dim=0)
    
    if logit_first > logit_second:
        predicted = first
    else:
        predicted = second
    
    correct_bool = (predicted == correct_str)
    
    # Entropy
    full_probs = torch.softmax(logits.float(), dim=0)
    entropy = -torch.sum(full_probs * torch.log(full_probs + 1e-10)).item()
    
    # Greedy token
    greedy_id = logits.argmax().item()
    greedy_token = tokenizer.decode([greedy_id]).strip()
    
    return {
        "predicted": predicted,
        "first_logit": logit_first,
        "second_logit": logit_second,
        "first_prob": probs[0].item(),
        "second_prob": probs[1].item(),
        "confidence": abs(logit_first - logit_second),
        "entropy": entropy,
        "greedy_token": greedy_token,
        "correct": correct_bool,
    }


def run_task(model, tokenizer, task_name, prompts, model_key, device="cuda"):
    """Run all prompts for one task type."""
    
    results = []
    t0 = time.time()
    
    for i, prompt_data in enumerate(prompts):
        if task_name in ("symbolic_control", "symbolic"):
            score = score_symbolic_prompt(model, tokenizer, prompt_data, model_key, device)
        else:
            score = score_ab_prompt(
                model, tokenizer, prompt_data["prompt"],
                prompt_data["correct_answer"], model_key, device
            )
        
        # Merge prompt metadata with score
        result = {
            "task_type": "SYMBOLIC" if task_name in ("symbolic_control", "symbolic") else task_name.upper(),
            "domain": "numerical",
            "baseline": prompt_data.get("nominal_baseline", 0),
            "ratio": prompt_data.get("nominal_ratio", 0),
            "correct_answer": prompt_data["correct_answer"],
            "prompt": prompt_data["prompt"],
            "pair_id": prompt_data.get("pair_id", ""),
        }
        result.update(score)
        results.append(result)
        
        if (i + 1) % 250 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(prompts) - i - 1) / rate
            log.info(f"  {task_name}: {i+1}/{len(prompts)} ({elapsed:.0f}s, ~{remaining:.0f}s remaining)")
    
    elapsed = time.time() - t0
    acc = np.mean([r["correct"] for r in results])
    log.info(f"  {task_name}: {len(results)} items, accuracy = {acc:.3f} ({elapsed:.1f}s)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run B2, B3, symbolic control")
    parser.add_argument("--model", choices=["llama_instruct", "mistral_instruct", "llama_base"], required=True)
    parser.add_argument("--project-root", type=str, default=r"C:\weber")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks", type=str, default="b2,b3,symbolic",
                        help="Comma-separated task list (default: b2,b3,symbolic)")
    args = parser.parse_args()
    
    import os
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    
    stimuli_dir = Path(args.project_root) / "stimuli"
    results_dir = Path(args.project_root) / "results" / "paradigm_b" / args.model / "numerical"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_to_run = [t.strip() for t in args.tasks.split(",")]
    
    # Load stimuli
    task_files = {
        "b1": "prompts_b1.json",
        "b2": "prompts_b2.json",
        "b3": "prompts_b3.json",
        "symbolic": "prompts_symbolic_control.json",
    }
    
    task_prompts = {}
    for task in tasks_to_run:
        path = stimuli_dir / task_files[task]
        with open(path) as f:
            task_prompts[task] = json.load(f)
        log.info(f"Loaded {task}: {len(task_prompts[task])} prompts")
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    log.info(f"\n{'='*70}")
    log.info(f"PARADIGM B: B2/B3/Symbolic — {args.model}")
    log.info(f"{'='*70}")
    
    all_raw = []
    task_summaries = {}
    
    for task in tasks_to_run:
        log.info(f"\n--- Task: {task} ---")
        results = run_task(model, tokenizer, task, task_prompts[task], args.model, args.device)
        all_raw.extend(results)
        
        # Summary by ratio
        acc = np.mean([r["correct"] for r in results])
        ratios = sorted(set(r["ratio"] for r in results))
        
        ratio_accs = {}
        for ratio in ratios:
            ratio_items = [r for r in results if r["ratio"] == ratio]
            ratio_accs[str(ratio)] = float(np.mean([r["correct"] for r in ratio_items]))
        
        mean_entropy = float(np.mean([r["entropy"] for r in results]))
        
        task_summaries[task] = {
            "n_items": len(results),
            "accuracy": float(acc),
            "mean_entropy": mean_entropy,
            "accuracy_by_ratio": ratio_accs,
        }
        
        log.info(f"  Overall accuracy: {acc:.3f}")
        log.info(f"  Mean entropy: {mean_entropy:.3f}")
        log.info(f"  By ratio: {' | '.join(f'{r}:{a:.3f}' for r, a in ratio_accs.items())}")
    
    # Save raw results
    raw_path = results_dir / "paradigm_b_b2b3_raw.json"
    with open(raw_path, 'w') as f:
        json.dump(all_raw, f, indent=2, cls=NumpyEncoder)
    log.info(f"\nSaved raw: {raw_path}")
    
    # Save summary
    summary_path = results_dir / "paradigm_b_b2b3_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "model": args.model,
            "tasks": task_summaries,
        }, f, indent=2, cls=NumpyEncoder)
    log.info(f"Saved summary: {summary_path}")
    
    # Print comparison table
    log.info(f"\n{'='*70}")
    log.info(f"TASK COMPARISON: {args.model}")
    log.info(f"{'='*70}")
    log.info(f"  {'Task':>10s} | Accuracy | Entropy | Ratio effect?")
    for task, s in task_summaries.items():
        ratios = s["accuracy_by_ratio"]
        ratio_vals = list(ratios.values())
        has_ratio = ratio_vals[-1] - ratio_vals[0] > 0.05 if len(ratio_vals) > 1 else False
        log.info(f"  {task:>10s} | {s['accuracy']:.3f}    | {s['mean_entropy']:.3f}   | {'yes' if has_ratio else 'no'}")


if __name__ == "__main__":
    main()
