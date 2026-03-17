import json
d = json.load(open(r'C:\weber\results\paradigm_b\mistral_instruct\numerical\paradigm_b_b2b3_raw.json'))
sym = [r for r in d if r['task_type'] == 'SYMBOLIC']
print(f'Symbolic items: {len(sym)}')
for r in sym[:5]:
    print(f"  prompt: {r['prompt'][:60]}...")
    print(f"  correct_answer={r['correct_answer']}, predicted={r['predicted']}, greedy={r.get('greedy_token','?')}, correct={r['correct']}")
    print()
