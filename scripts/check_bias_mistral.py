import json
import numpy as np
raw = json.load(open(r'C:\weber\results\paradigm_b\mistral_instruct\numerical\paradigm_b_raw.json'))
b1 = [r for r in raw if r['task_type'] == 'B1']
ratios = sorted(set(r['ratio'] for r in b1))
print('MISTRAL B1 position-corrected accuracy:')
print('Ratio | Acc(A-correct) | Acc(B-correct) | Acc(avg)')
for ratio in ratios:
    a_trials = [r for r in b1 if r['ratio'] == ratio and r['correct_answer'] == 'A']
    b_trials = [r for r in b1 if r['ratio'] == ratio and r['correct_answer'] == 'B']
    acc_a = np.mean([r['correct'] for r in a_trials]) if a_trials else 0
    acc_b = np.mean([r['correct'] for r in b_trials]) if b_trials else 0
    print(f'  {ratio:.2f} | {acc_a:.3f}          | {acc_b:.3f}          | {(acc_a+acc_b)/2:.3f}')
