import json
d = json.load(open(r'C:\weber\results\paradigm_a\llama_instruct\numerical\paradigm_a_analysis.json'))
if 'rsa_results' in d:
    rsa = d['rsa_results']
    print(type(rsa))
    if isinstance(rsa, dict):
        print(list(rsa.keys())[:10])
    elif isinstance(rsa, list):
        print(f'{len(rsa)} entries')
        print(rsa[0] if rsa else 'empty')
else:
    print('Keys:', list(d.keys()))
    for k in list(d.keys())[:5]:
        v = d[k]
        if isinstance(v, dict):
            print(f'  {k}: dict with keys {list(v.keys())[:5]}')
        elif isinstance(v, list):
            print(f'  {k}: list len {len(v)}')
        else:
            print(f'  {k}: {type(v).__name__} = {v}')
