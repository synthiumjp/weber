"""
Paradigm B diagnostic: position bias analysis on 50 B1 pairs.

Tests whether the model has ANY magnitude sensitivity beneath the A-bias.
Computes:
  1. Raw accuracy (will be ~50% due to bias)
  2. Position-corrected accuracy: for each pair, compare logit(correct_option)
     across the two orderings
  3. Confidence modulation: does the A-B logit gap vary with magnitude ratio?
"""
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B-Instruct',
    torch_dtype=torch.float16,
).to('cuda')
model.eval()

a_ids = tok.encode("A", add_special_tokens=False) + tok.encode(" A", add_special_tokens=False)
b_ids = tok.encode("B", add_special_tokens=False) + tok.encode(" B", add_special_tokens=False)


def get_ab_logits(prompt):
    messages = [{"role": "user", "content": prompt}]
    formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(formatted, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float().cpu()
    a_logit = max(logits[t].item() for t in a_ids)
    b_logit = max(logits[t].item() for t in b_ids)
    return a_logit, b_logit


# Load B1 stimuli
with open('C:/weber/stimuli/prompts_b1.json') as f:
    b1_all = json.load(f)

# Take first 50 pairs, test with labelled prompts
results = []
n_correct_raw = 0
n_total = 0

print("Testing 50 B1 prompts with chat template + A/B labels...")
print()

for item in b1_all[:50]:
    a_expr = item['a_expression']
    b_expr = item['b_expression']
    correct = item['correct_answer']
    ratio = item['nominal_ratio']
    baseline = item['nominal_baseline']

    prompt = "Which represents a larger quantity: A) %s or B) %s? Answer with only A or B." % (a_expr, b_expr)
    a_logit, b_logit = get_ab_logits(prompt)
    pred = "A" if a_logit > b_logit else "B"
    is_correct = pred == correct

    if is_correct:
        n_correct_raw += 1
    n_total += 1

    # logit assigned to the CORRECT option (regardless of position)
    correct_logit = a_logit if correct == "A" else b_logit
    incorrect_logit = b_logit if correct == "A" else a_logit
    # Positive = model favours correct answer
    correct_advantage = correct_logit - incorrect_logit

    results.append({
        'ratio': ratio,
        'baseline': baseline,
        'correct': correct,
        'pred': pred,
        'a_logit': a_logit,
        'b_logit': b_logit,
        'correct_advantage': correct_advantage,
        'is_correct': is_correct,
    })

    if n_total <= 10 or n_total % 10 == 0:
        print("  %d/%d acc=%.3f | ratio=%.2f correct=%s pred=%s A=%.1f B=%.1f adv=%.1f" % (
            n_total, 50, n_correct_raw/n_total, ratio, correct, pred,
            a_logit, b_logit, correct_advantage))

print()
print("=" * 60)
print("SUMMARY (n=%d)" % n_total)
print("=" * 60)

# Raw accuracy
print("Raw accuracy: %.3f" % (n_correct_raw / n_total))

# Position bias
n_pred_a = sum(1 for r in results if r['pred'] == 'A')
print("Always-A rate: %.3f" % (n_pred_a / n_total))

# Does correct_advantage correlate with ratio?
ratios = np.array([r['ratio'] for r in results])
advantages = np.array([r['correct_advantage'] for r in results])
from scipy.stats import spearmanr
rho, p = spearmanr(ratios, advantages)
print("Correct-advantage vs ratio: Spearman rho=%.3f p=%.4f" % (rho, p))

# Mean correct_advantage by whether correct=A vs correct=B
adv_when_a = [r['correct_advantage'] for r in results if r['correct'] == 'A']
adv_when_b = [r['correct_advantage'] for r in results if r['correct'] == 'B']
print("Mean advantage when correct=A: %.2f (model biased TOWARD correct)" % np.mean(adv_when_a))
print("Mean advantage when correct=B: %.2f (model biased AWAY from correct)" % np.mean(adv_when_b))

# Does the model at least show RELATIVE sensitivity?
# Group by ratio and see if accuracy varies
for r in sorted(set(ratios)):
    subset = [x for x in results if x['ratio'] == r]
    acc = np.mean([x['is_correct'] for x in subset])
    mean_adv = np.mean([x['correct_advantage'] for x in subset])
    print("  Ratio %.2f: n=%d acc=%.3f mean_advantage=%.2f" % (r, len(subset), acc, mean_adv))

print()
print("DONE")
