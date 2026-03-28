import json

with open('extractions/m3_meta_temp_hotcold_llama3-8b-instruct.json') as f:
    meta = json.load(f)

print('Keys:', list(meta.keys()))

# Try both possible key names
for key in ['identification_results', 'identification']:
    if key in meta:
        print(f'\nFound: {key} ({len(meta[key])} entries)')
        for r in meta[key]:
            framing = r.get('framing', '')
            val = r.get('value', '?')
            pb = r.get('prob_category_b', None)
            if pb is not None:
                print(f'  {val:>6} C  [{framing:<15}]  P(hot/warm/comf) = {pb:.4f}')
            else:
                # Raw logit format
                pa = r.get('prob_category_a', '?')
                pb2 = r.get('prob_category_b', '?')
                print(f'  {val:>6} C  [{framing:<15}]  P(a)={pa}  P(b)={pb2}')
        break
else:
    print('No identification results found')
    print('Available keys:', list(meta.keys()))
