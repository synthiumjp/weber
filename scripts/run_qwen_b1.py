"""
run_qwen_b1.py — Run existing B1 cross-format prompts through Qwen
Uses the exact same prompts that Llama and Mistral were tested on.
"""

import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Load B1 prompts from Llama's raw results (prompts are model-independent)
print("Loading B1 prompts...")
with open(r"C:\weber\results\paradigm_b\llama_instruct\numerical\paradigm_b_raw.json") as f:
    all_items = json.load(f)

b1_items = [x for x in all_items if x["task_type"] == "B1"]
print(f"  {len(b1_items)} B1 items loaded")

# Load model
print(f"\nLoading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
)
model = model.to(DEVICE)
model.config.output_hidden_states = False  # don't need hidden states, save memory
model.eval()
print("  Model loaded.")

# Run B1
print("\nRunning B1 cross-format comparison...")
results = []
correct_count = 0
t_start = time.time()

for idx, item in enumerate(b1_items):
    # Use the exact pre-built prompt, wrap in chat template
    prompt_text = item["prompt"]
    
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]
    
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    
    logit_a = logits[a_ids[0]].item()
    logit_b = logits[b_ids[0]].item()
    
    predicted = "A" if logit_a > logit_b else "B"
    is_correct = predicted == item["correct_answer"]
    
    probs = torch.softmax(torch.tensor([logit_a, logit_b]), dim=0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    
    results.append({
        "task_type": "B1",
        "baseline": item["baseline"],
        "ratio": item["ratio"],
        "value_a": item["value_a"],
        "value_b": item["value_b"],
        "correct_answer": item["correct_answer"],
        "predicted": predicted,
        "correct": is_correct,
        "logit_a": logit_a,
        "logit_b": logit_b,
        "entropy": entropy,
        "pair_id": item["pair_id"],
    })
    
    if is_correct:
        correct_count += 1
    
    if (idx + 1) % 250 == 0:
        acc = correct_count / (idx + 1)
        print(f"  {idx+1}/{len(b1_items)}: {acc:.3f} accuracy")

elapsed = time.time() - t_start
total = len(results)
overall_acc = correct_count / total

print(f"\n{'='*50}")
print(f"QWEN B1 RESULTS")
print(f"{'='*50}")
print(f"Overall accuracy: {overall_acc:.3f} ({correct_count}/{total})")
print(f"Mean entropy: {np.mean([r['entropy'] for r in results]):.3f}")
print(f"Runtime: {elapsed:.0f}s")

# Accuracy by nominal ratio
print(f"\nAccuracy by ratio:")
from collections import defaultdict
ratio_bins = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results:
    # Bin to nearest nominal ratio
    raw = r["ratio"]
    nominals = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
    nearest = min(nominals, key=lambda x: abs(x - raw))
    ratio_bins[nearest]["total"] += 1
    if r["correct"]:
        ratio_bins[nearest]["correct"] += 1

for ratio in sorted(ratio_bins.keys()):
    d = ratio_bins[ratio]
    acc = d["correct"] / d["total"] if d["total"] > 0 else 0
    print(f"  Ratio {ratio:.2f}: {acc:.3f} ({d['correct']}/{d['total']})")

# Accuracy by baseline
print(f"\nAccuracy by baseline:")
base_bins = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results:
    base_bins[r["baseline"]]["total"] += 1
    if r["correct"]:
        base_bins[r["baseline"]]["correct"] += 1

for base in sorted(base_bins.keys()):
    d = base_bins[base]
    acc = d["correct"] / d["total"] if d["total"] > 0 else 0
    print(f"  Baseline {base:.0f}: {acc:.3f} ({d['correct']}/{d['total']})")

# Save
out_path = r"C:\weber\results\exploratory\qwen25_7b\paradigm_b1_crossformat.json"
summary = {
    "model": MODEL_ID,
    "task": "B1_cross_format",
    "overall_accuracy": overall_acc,
    "n_items": total,
    "mean_entropy": float(np.mean([r["entropy"] for r in results])),
    "accuracy_by_ratio": {str(k): {"accuracy": v["correct"]/v["total"], **v} 
                          for k, v in sorted(ratio_bins.items())},
    "accuracy_by_baseline": {str(k): {"accuracy": v["correct"]/v["total"], **v}
                             for k, v in sorted(base_bins.items())},
    "items": results,
}

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out_path}")
