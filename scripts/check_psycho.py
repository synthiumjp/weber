import json
d = json.load(open(r'C:\weber\results\paradigm_b\llama_instruct\numerical\paradigm_b_results.json'))
pf = d['psychometric_fits']
print(f'Type: {type(pf)}')
if isinstance(pf, dict):
    print(f'Keys: {list(pf.keys())[:10]}')
    for k in list(pf.keys())[:3]:
        print(f'  {k}: {pf[k]}')
elif isinstance(pf, list):
    print(f'Len: {len(pf)}')
    for item in pf[:3]:
        print(f'  {item}')
print(f'\nH2: {d["h2_evaluation"]}')
print(f'\nWeber fracs: {d.get("bootstrap_weber_fractions", "missing")}')
