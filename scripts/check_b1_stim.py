import json, os
stim_dir = r'C:\weber\stimuli'
files = [f for f in os.listdir(stim_dir) if 'b1' in f.lower() or 'cross_format' in f.lower() or 'comparison' in f.lower()]
print('B1-related files:', files)
for f in files[:3]:
    d = json.load(open(os.path.join(stim_dir, f)))
    if isinstance(d, list) and len(d) > 0:
        print(f'\n{f}: {len(d)} items')
        print(f'  Keys: {d[0].keys()}')
        print(f'  Example: {d[0]}')
