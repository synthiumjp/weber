import json
d = json.load(open(r'C:\weber\results\paradigm_b\llama_instruct\numerical\paradigm_b_b2b3_raw.json'))
types = set(r['task_type'] for r in d)
print(f'Task types: {types}')
sym = [r for r in d if 'symbolic' in r['task_type'].lower() or 'SYMBOLIC' in r['task_type']]
print(f'Symbolic items: {len(sym)}')
if sym:
    print(f'  [0]: predicted={sym[0]["predicted"]}, correct_answer={sym[0]["correct_answer"]}, correct={sym[0]["correct"]}')
    print(f'  [0] full: { {k:v for k,v in sym[0].items() if k != "prompt"} }')
