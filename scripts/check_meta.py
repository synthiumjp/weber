import json
tk = json.load(open(r'C:\weber\stimuli\metadata.json'))
if 'single_token' in str(tk).lower() or 'tokenisation' in str(tk).lower():
    print(type(tk))
    if isinstance(tk, dict):
        for k in tk.keys():
            if 'token' in k.lower():
                print(f'{k}: {tk[k]}')
    print(list(tk.keys())[:15])
else:
    print('Keys:', list(tk.keys())[:15])
    for k in list(tk.keys())[:5]:
        v = tk[k]
        if isinstance(v, (str, int, float, bool)):
            print(f'  {k}: {v}')
        elif isinstance(v, (list, dict)):
            print(f'  {k}: {type(v).__name__} len {len(v)}')
