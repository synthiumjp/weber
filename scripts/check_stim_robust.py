import json
for f in ['shuffled_magnitudes.json', 'unit_boundary_check.json']:
    d = json.load(open(rf'C:\weber\stimuli\{f}'))
    if isinstance(d, list):
        print(f'{f}: {len(d)} items')
        if d:
            print(f'  keys: {list(d[0].keys()) if isinstance(d[0], dict) else type(d[0])}')
            print(f'  [0]: {d[0]}')
    elif isinstance(d, dict):
        print(f'{f}: dict with keys {list(d.keys())[:8]}')
        for k in list(d.keys())[:2]:
            v = d[k]
            print(f'  {k}: {type(v).__name__} = {v if not isinstance(v, (list,dict)) else f"len {len(v)}"}')
