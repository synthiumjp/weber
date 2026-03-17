import json
d = json.load(open(r'C:\weber\stimuli\prompts_b1.json'))
first_correct_a = sum(1 for p in d if p['correct_answer'] == 'A')
first_correct_b = sum(1 for p in d if p['correct_answer'] == 'B')
print(f'Correct=A: {first_correct_a}, Correct=B: {first_correct_b}')
print(f'Total: {len(d)}')
