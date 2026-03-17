import json
import numpy as np
d = json.load(open(r'C:\weber\results\paradigm_d\llama_numerical_paradigm_d_results.json'))
h7 = d['h7']
details = h7['prompt_details']
mag_deltas = [p['mag_abs_delta_p'] for p in details]
thresholds = [p['random_97_5_pct'] for p in details]
exceeds = [p['exceeds'] for p in details]
print(f"Mag |dp| distribution: mean={np.mean(mag_deltas):.4f}, median={np.median(mag_deltas):.4f}, max={np.max(mag_deltas):.4f}")
print(f"Random 97.5pct distribution: mean={np.mean(thresholds):.4f}, median={np.median(thresholds):.4f}")
print(f"Prompts exceeding: {sum(exceeds)}/200")
print(f"\nBy baseline:")
prompts = d['prompts']
for bl in [10, 30, 100, 300, 1000]:
    bl_prompts = [p for p in prompts if p.get('nominal_baseline') == bl and 'error' not in p]
    bl_mag = [abs(p['magnitude_direction']['1.0']['delta_p']) for p in bl_prompts]
    bl_base = [p['p_correct_baseline'] for p in bl_prompts]
    print(f"  baseline {bl:4d}: n={len(bl_prompts)}, mean|dp|={np.mean(bl_mag):.4f}, mean_p_baseline={np.mean(bl_base):.3f}")
