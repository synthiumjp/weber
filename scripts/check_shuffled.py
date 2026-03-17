import json
d = json.load(open(r'C:\weber\stimuli\shuffled_magnitudes.json'))
num = d['numerical']
print(type(num), list(num.keys()) if isinstance(num, dict) else len(num))
if isinstance(num, dict):
    for k in num.keys():
        v = num[k]
        if isinstance(v, list):
            print(f'  {k}: list len {len(v)}')
            if v and isinstance(v[0], dict):
                print(f'    [0]: {v[0]}')
            elif v:
                print(f'    [0]: {v[0]}')
        else:
            print(f'  {k}: {v}')
