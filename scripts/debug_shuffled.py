import json, numpy as np
d = json.load(open(r'C:\weber\results\robustness\llama_instruct_shuffled_magnitude.json'))
layers = d['layers']
rhos = [l['rho_shuffled'] for l in layers]
unique = len(set([f"{r:.6f}" for r in rhos]))
print(f'Unique rho values: {unique}/{len(rhos)}')
print(f'All identical: {unique == 1}')
