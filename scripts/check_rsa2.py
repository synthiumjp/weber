import json
d = json.load(open(r'C:\weber\results\paradigm_a\llama_instruct\numerical\paradigm_a_analysis.json'))
best_rsa = -1
best_layer = -1
for lname, ldata in d['layers'].items():
    layer_num = int(lname.split('_')[1])
    if isinstance(ldata, dict) and 'cosine' in ldata:
        cos = ldata['cosine']
        if isinstance(cos, dict) and 'rsa' in cos:
            rsa = cos['rsa']
            rho = rsa.get('weber_rho', rsa.get('mantel_rho', -1))
            r2w = cos.get('model_fits', {}).get('weber', {}).get('r2', -1)
            if layer_num >= 16:
                if rho > best_rsa:
                    best_rsa = rho
                    best_layer = layer_num
            if layer_num in [5, 16, 20, 24, 28, 31]:
                print(f'  Layer {layer_num:2d}: Weber rho={rho:.4f}, Weber R2={r2w:.4f}')
print(f'\nPeak RSA layer (>=16): {best_layer} (rho={best_rsa:.4f})')
