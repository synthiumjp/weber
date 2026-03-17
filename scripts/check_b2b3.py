import json
for task in ['b2', 'b3', 'symbolic_control']:
    f = f'prompts_{task}.json'
    try:
        d = json.load(open(rf'C:\weber\stimuli\{f}'))
        print(f'{f}: {len(d)} items')
        print(f'  keys: {list(d[0].keys())}')
        print(f'  [0]: {d[0]}')
        print()
    except FileNotFoundError:
        print(f'{f}: NOT FOUND')
