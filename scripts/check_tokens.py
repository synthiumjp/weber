import json, os
# Check for tokenisation files
for f in os.listdir(r'C:\weber\results\sanity_checks'):
    print(f)
print()
# Also check config for single-token magnitudes
import sys
sys.path.insert(0, r'C:\weber\scripts')
import config
for attr in dir(config):
    if 'token' in attr.lower() or 'single' in attr.lower():
        print(f'{attr}: {getattr(config, attr)}')
