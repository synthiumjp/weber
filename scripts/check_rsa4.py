import json
d = json.load(open(r'C:\weber\results\paradigm_a\llama_instruct\numerical\paradigm_a_analysis.json'))
best_rho = -1
best_layer = -1
print('Layer | Weber rho | Weber R2 | Best AIC')
for lname, ldata in sorted(d['layers'].items()):
    layer_num = int(lname.split('_')[1])
    cos = ldata.get('cosine', {})
    rsa = cos.get('rsa', {})
    mf = cos.get('model_fits', {})
    rho = rsa.get('weber', {}).get('rho', -1)
    r2 = mf.get('weber', {}).get('r2', -1)
    baic = mf.get('best_aic', '?')
    if layer_num >= 16 and rho > best_rho:
        best_rho = rho
        best_layer = layer_num
    if layer_num >= 16:
        print(f'  {layer_num:2d}  | {rho:.4f}   | {r2:.4f}  | {baic}')
print(f'\nPeak RSA Weber rho (layers 16-32): layer {best_layer} (rho={best_rho:.4f})')
