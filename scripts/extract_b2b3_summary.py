#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — Extract B2/B3/Symbolic Summaries
============================================================
- B2/B3: scored from paradigm_b_raw.json 'correct' field (reliable for A/B tasks)
- Symbolic: locked from confirmed values (paradigm_b_results.json entropy diagnostic
  + session log). The 'correct' field is broken for symbolic items in both raw files
  because greedy_token captures only the first BPE token for multi-token numbers.

  Confirmed symbolic accuracies:
    Llama:   99.87% (1498/1500 correct, from original paradigm_b_additional.py)
    Mistral: 50.00% (750/1500, pure position bias)

No GPU needed. Usage: python scripts/extract_b2b3_summary.py
"""

import json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

MODELS = ['llama_instruct', 'mistral_instruct']
RATIOS = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]

# Locked symbolic accuracies from confirmed session log + original analysis
SYMBOLIC_LOCKED = {
    'llama_instruct': {
        'n_items': 1500,
        'accuracy': 0.9987,
        'accuracy_by_ratio': {'1.05': 1.0, '1.1': 1.0, '1.2': 1.0, '1.5': 1.0, '2.0': 0.992, '3.0': 1.0},
        'mean_entropy': 0.0009,
    },
    'mistral_instruct': {
        'n_items': 1500,
        'accuracy': 0.5000,
        'accuracy_by_ratio': {'1.05': 0.5, '1.1': 0.5, '1.2': 0.5, '1.5': 0.5, '2.0': 0.5, '3.0': 0.5},
        'mean_entropy': 0.2101,
    },
}


def summarise_items(items):
    n = len(items)
    if n == 0: return None
    n_correct = sum(1 for it in items if bool(it.get('correct', False)))
    acc = n_correct / n
    by_ratio = defaultdict(list)
    for it in items:
        rt = it.get('ratio', it.get('nominal_ratio'))
        if rt is not None:
            rt_snap = min(RATIOS, key=lambda r: abs(r - rt))
            by_ratio[rt_snap].append(1 if bool(it.get('correct', False)) else 0)
    acc_by_ratio = {str(rt): round(np.mean(v), 4) for rt, v in sorted(by_ratio.items()) if v}
    entropies = [it.get('entropy') for it in items if it.get('entropy') is not None]
    return {'n_items': n, 'accuracy': round(acc, 4), 'accuracy_by_ratio': acc_by_ratio,
            'mean_entropy': float(np.mean(entropies)) if entropies else None}


def process_model(results_dir, model):
    base = Path(results_dir) / 'paradigm_b' / model / 'numerical'
    raw_path = base / 'paradigm_b_raw.json'
    summary_path = base / 'paradigm_b_b2b3_summary.json'

    if not raw_path.exists():
        print(f"  [SKIP] {raw_path} not found"); return

    with open(raw_path) as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get('items', [])

    by_task = defaultdict(list)
    for it in items:
        by_task[it.get('task_type', 'unknown')].append(it)
    print(f"  {model}: {len(items)} items, tasks: {dict((k,len(v)) for k,v in by_task.items())}")

    summary = {'model': model, 'tasks': {}}

    # B1, B2, B3: 'correct' field is reliable
    for raw_key, sum_key in [('B1','b1'), ('B2','b2'), ('B3','b3')]:
        if raw_key in by_task:
            r = summarise_items(by_task[raw_key])
            if r:
                summary['tasks'][sum_key] = r
                print(f"    {sum_key}: {r['accuracy']:.1%} (n={r['n_items']})")

    # Symbolic: locked values
    summary['tasks']['symbolic'] = SYMBOLIC_LOCKED[model]
    print(f"    symbolic: {SYMBOLIC_LOCKED[model]['accuracy']:.1%} (LOCKED from confirmed analysis)")

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default=r'C:\weber\results')
    a = p.parse_args()
    print("="*60); print("Extracting B task summaries"); print("="*60)
    for model in MODELS:
        print(f"\n--- {model} ---")
        process_model(a.results_dir, model)
    print("\n"+"="*60); print("DONE."); print("="*60)

if __name__ == '__main__':
    main()
