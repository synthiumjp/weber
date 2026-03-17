import json
d = json.load(open(r'C:\weber\stimuli\unit_boundary_check.json'))
print('Keys:', list(d.keys()))
for domain in d:
    items = d[domain]
    print(f'\n{domain}: {len(items)} items')
    for item in items[:3]:
        print(f'  {item}')
