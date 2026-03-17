import sys, json
sys.path.insert(0, r'C:\weber\scripts')
from config import DOMAINS
num = DOMAINS.get('numerical', {})
carriers = num.get('carriers', num.get('carrier_sentences', None))
if carriers:
    for c in carriers:
        print(f'  "{c}"')
else:
    print('Carriers not in config. Check probing stimuli:')
    d = json.load(open(r'C:\weber\stimuli\probing_numerical.json'))
    if isinstance(d, list) and d:
        print(f'  keys: {list(d[0].keys())}')
        sents = set()
        for item in d[:10]:
            s = item.get('sentence', item.get('text', ''))
            print(f'  {s}')
