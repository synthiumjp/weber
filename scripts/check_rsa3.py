import json
d = json.load(open(r'C:\weber\results\paradigm_a\llama_instruct\numerical\paradigm_a_analysis.json'))
layer20 = d['layers']['layer_20']
cos = layer20['cosine']
print('cosine keys:', list(cos.keys()))
if 'rsa' in cos:
    print('rsa:', cos['rsa'])
if 'model_fits' in cos:
    mf = cos['model_fits']
    print('model_fits keys:', list(mf.keys()))
    if 'weber' in mf:
        print('weber:', mf['weber'])
for k, v in cos.items():
    if isinstance(v, dict):
        print(f'{k}: {list(v.keys())[:8]}')
    elif isinstance(v, (int, float, str, bool)):
        print(f'{k}: {v}')
