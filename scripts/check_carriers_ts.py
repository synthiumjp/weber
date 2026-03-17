import sys
sys.path.insert(0, r'C:\weber\scripts')
from config import DOMAINS
for domain in ['temporal', 'spatial']:
    carriers = DOMAINS[domain]['carriers']
    print(f'{domain}:')
    for c in carriers:
        print(f'  {c.format(N="60 seconds")}')
