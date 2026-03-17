import json
d = json.load(open(r'C:\weber\results\paradigm_d\llama_numerical_gate_report.json'))
cv = d['cv_robustness']
print('Layer | Primary R2 | CV LOO-R2 | CV alpha')
for i in range(33):
    r2 = d['all_r2'][i]
    loo = cv['cv_all_loo_r2'][i]
    a = cv['cv_all_alphas'][i]
    print(f'  {i:2d}  | {r2:.4f}     | {loo:+.4f}    | {a:.1f}')
