import json, os
bdir = r'C:\weber\results\paradigm_b'
print('Files:', os.listdir(bdir))
for f in os.listdir(bdir):
    if f.endswith('.json') and 'llama' in f:
        d = json.load(open(os.path.join(bdir, f)))
        if isinstance(d, dict):
            print(f'\n{f}: keys={list(d.keys())[:10]}')
            for k in list(d.keys())[:5]:
                v = d[k]
                if isinstance(v, list):
                    print(f'  {k}: list len {len(v)}')
                    if v and isinstance(v[0], dict):
                        print(f'    [0] keys: {list(v[0].keys())[:8]}')
                elif isinstance(v, dict):
                    print(f'  {k}: dict keys {list(v.keys())[:8]}')
                else:
                    print(f'  {k}: {v}')
        break
