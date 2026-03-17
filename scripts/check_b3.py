import json, os
bdir = r'C:\weber\results\paradigm_b\llama_instruct\numerical'
print('Contents:', os.listdir(bdir))
for f in sorted(os.listdir(bdir)):
    if f.endswith('.json'):
        path = os.path.join(bdir, f)
        d = json.load(open(path))
        if isinstance(d, dict):
            print(f'\n{f}: keys={list(d.keys())[:12]}')
        elif isinstance(d, list):
            print(f'\n{f}: list len {len(d)}')
            if d and isinstance(d[0], dict):
                print(f'  [0] keys: {list(d[0].keys())}')
                print(f'  [0]: {d[0]}')
