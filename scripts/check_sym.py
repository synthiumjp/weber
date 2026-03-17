import json
d = json.load(open(r'C:\weber\results\paradigm_b\llama_instruct\numerical\paradigm_b_b2b3_raw.json'))
sym = [r for r in d if r['task_type'] == 'symbolic_control']
for r in sym[:5]:
    print(f"  correct={r['correct_answer']}, predicted={r['predicted']}, correct={r['correct']}")
