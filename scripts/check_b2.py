import json, os
bdir = r'C:\weber\results\paradigm_b\llama_instruct'
print('Contents:', os.listdir(bdir))
for f in os.listdir(bdir):
    if f.endswith('.json'):
        path = os.path.join(bdir, f)
        d = json.load(open(path))
        if isinstance(d, dict):
            print(f'\n{f}: keys={list(d.keys())[:10]}')
            for k in list(d.keys())[:5]:
                v = d[k]
                if isinstance(v, list):
                    print(f'  {k}: list len {len(v)}')
                    if v and isinstance(v[0], dict):
                        print(f'    [0] keys: {list(v[0].keys())[:10]}')
                        print(f'    [0]: {v[0]}')
                elif isinstance(v, dict):
                    print(f'  {k}: dict keys {list(v.keys())[:8]}')
                else:
                    print(f'  {k}: {v}')
        elif isinstance(d, list):
            print(f'\n{f}: list len {len(d)}')
            if d and isinstance(d[0], dict):
                print(f'  [0] keys: {list(d[0].keys())[:10]}')
                print(f'  [0]: {d[0]}')
        break
