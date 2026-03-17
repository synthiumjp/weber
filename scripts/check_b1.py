import json
d = json.load(open(r'C:\weber\stimuli\prompts_b1.json'))
print(f'Total: {len(d)} items')
print(f'Keys: {d[0].keys()}')
print(f'Example 0: {d[0]}')
print(f'Example 1: {d[1]}')
print(f'Example -1: {d[-1]}')
