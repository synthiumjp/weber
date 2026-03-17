import os
for root, dirs, files in os.walk(r'C:\weber\results'):
    for f in files:
        if any(x in f.lower() for x in ['shuffle', 'unit_bound', 'single_token', 'corpus', 'robustness']):
            print(os.path.join(root, f))
