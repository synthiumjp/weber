import json
d = json.load(open(r'C:\weber\results\sanity_checks\token_positions_llama_instruct.json'))
print(type(d))
if isinstance(d, list):
    print(f'len: {len(d)}')
    print(d[0])
elif isinstance(d, dict):
    print(list(d.keys())[:10])
    for k in list(d.keys())[:3]:
        print(f'  {k}: {d[k]}')
