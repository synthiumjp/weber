import json, numpy as np
d = json.load(open(r'C:\weber\results\paradigm_d\mistral_numerical_paradigm_d_results.json'))
prompts = d['prompts']
for bl in [10, 30, 100, 300, 1000]:
    bp = [p for p in prompts if p.get('nominal_baseline') == bl and 'error' not in p]
    mag = [abs(p['magnitude_direction']['1.0']['delta_p']) for p in bp]
    base = [p['p_correct_baseline'] for p in bp]
    print(f'  baseline {bl:4d}: n={len(bp)}, mean|dp|={np.mean(mag):.4f}, mean_p_baseline={np.mean(base):.3f}')
