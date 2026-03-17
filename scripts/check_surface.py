import json
import numpy as np
raw = json.load(open(r'C:\weber\results\paradigm_b\llama_instruct\numerical\paradigm_b_raw.json'))
b1 = [r for r in raw if r['task_type'] == 'B1']
baselines = sorted(set(r['baseline'] for r in b1))
ratios = sorted(set(r['ratio'] for r in b1))
print('         |', '  |  '.join(f'{r:.2f}' for r in ratios))
for bl in baselines:
    accs = []
    for ratio in ratios:
        trials_a = [r for r in b1 if r['baseline']==bl and r['ratio']==ratio and r['correct_answer']=='A']
        trials_b = [r for r in b1 if r['baseline']==bl and r['ratio']==ratio and r['correct_answer']=='B']
        acc_a = np.mean([r['correct'] for r in trials_a]) if trials_a else 0.5
        acc_b = np.mean([r['correct'] for r in trials_b]) if trials_b else 0.5
        accs.append(f'{(acc_a+acc_b)/2:.3f}')
    print(f'  bl={bl:5.0f} | {"  |  ".join(accs)}')
