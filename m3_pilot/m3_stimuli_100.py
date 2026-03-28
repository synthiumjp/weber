"""
M3 Stimulus Generation: 100-Boundary Replication
=================================================
Generates stimulus files for decade_100 and control_150 conditions,
matching the EXACT JSON format produced by m3_stimuli.py so that
m3_extract.py works unchanged.

Required output format:
  probing_sentences: list of {text, magnitude_token, value, sentence_idx, is_rsa, is_b0}
  identification_stimuli: list of {value, framing, prompt, targets}
  theoretical_rdms: {Continuous, CP-Additive, Categorical, Linear, lambda}
  probing_values, boundary, condition, pairs, ...

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import json
import math
import itertools
import numpy as np
from pathlib import Path


CARRIER_SENTENCES = [
    "The number {N} is a quantity.",
    "There are {N} items in the collection.",
    "A total of {N} units were recorded.",
    "The measurement showed {N}.",
    "Approximately {N} cases were observed.",
    "The count reached {N} in total.",
    "A value of {N} was reported.",
    "The survey found {N} instances.",
]

B0_INDICES = [0, 1, 2, 3]
RSA_INDICES = [4, 5, 6, 7]


FRAMINGS_100 = [
    {
        'name': 'two_three_digit',
        'template': "Does the number {N} have two digits or three digits?",
        'category_a': 'two digits',
        'category_b': 'three digits',
        'targets': {'category_a': ['two', 'Two', '2'], 'category_b': ['three', 'Three', '3']},
    },
    {
        'name': 'tens_hundreds',
        'template': "Is {N} in the tens or in the hundreds?",
        'category_a': 'tens',
        'category_b': 'hundreds',
        'targets': {'category_a': ['tens', 'Tens', 'ten', 'Ten'], 'category_b': ['hundreds', 'Hundreds', 'hundred', 'Hundred']},
    },
    {
        'name': 'less_more_100',
        'template': "Is {N} less than one hundred or one hundred or more?",
        'category_a': 'less than one hundred',
        'category_b': 'one hundred or more',
        'targets': {'category_a': ['less', 'Less', 'under', 'Under'], 'category_b': ['one', 'One', 'hundred', 'Hundred']},
    },
]

FRAMINGS_150 = [
    {
        'name': 'small_large',
        'template': "Is {N} a small number or a large number?",
        'category_a': 'small',
        'category_b': 'large',
        'targets': {'category_a': ['small', 'Small'], 'category_b': ['large', 'Large']},
    },
]


def build_probing_sentences(values):
    sentences = []
    for val in values:
        for idx, template in enumerate(CARRIER_SENTENCES):
            text = template.format(N=val)
            sentences.append({
                'text': text,
                'magnitude_token': str(val),
                'value': val,
                'sentence_idx': idx,
                'is_rsa': idx in RSA_INDICES,
                'is_b0': idx in B0_INDICES,
            })
    return sentences


def build_identification_stimuli(values, framings):
    id_stims = []
    for val in values:
        for framing in framings:
            prompt = framing['template'].format(N=val)
            id_stims.append({
                'value': val,
                'framing': framing['name'],
                'prompt': prompt,
                'targets': framing['targets'],
                'category_a': framing['category_a'],
                'category_b': framing['category_b'],
            })
    return id_stims


def compute_theoretical_rdms(values, boundary):
    n = len(values)
    continuous = np.zeros((n, n))
    categorical = np.zeros((n, n))
    linear = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        log_dist = abs(math.log(values[i]) - math.log(values[j]))
        continuous[i, j] = continuous[j, i] = log_dist
        linear[i, j] = linear[j, i] = abs(values[i] - values[j])
        if (values[i] < boundary) != (values[j] < boundary):
            categorical[i, j] = categorical[j, i] = 1.0
    lam = float(np.mean(continuous[np.triu_indices(n, k=1)]))
    cp_additive = continuous + lam * categorical
    rdms = {
        'Continuous': continuous.tolist(),
        'CP-Additive': cp_additive.tolist(),
        'Categorical': categorical.tolist(),
        'Linear': linear.tolist(),
        'lambda': lam,
    }
    for name, rdm_list in rdms.items():
        if name == 'lambda':
            continue
        rdm = np.array(rdm_list)
        assert rdm.shape == (n, n) and np.allclose(rdm, rdm.T) and np.allclose(np.diag(rdm), 0)
    return rdms


def build_pairs(values, boundary):
    pairs = []
    for i, j in itertools.combinations(range(len(values)), 2):
        vi, vj = values[i], values[j]
        pairs.append({
            'i': i, 'j': j,
            'value_i': vi, 'value_j': vj,
            'log_distance': round(abs(math.log(vi) - math.log(vj)), 6),
            'linear_distance': abs(vi - vj),
            'crosses_boundary': (vi < boundary) != (vj < boundary),
        })
    return pairs


def generate_condition(name, values, boundary, framings, is_control=False):
    probing_sentences = build_probing_sentences(values)
    identification_stimuli = build_identification_stimuli(values, framings)
    rdms = compute_theoretical_rdms(values, boundary)
    pairs = build_pairs(values, boundary)
    return {
        'condition': name,
        'boundary': boundary,
        'is_control': is_control,
        'probing_values': values,
        'n_values': len(values),
        'probing_sentences': probing_sentences,
        'identification_stimuli': identification_stimuli,
        'n_pairs': len(pairs),
        'pairs': pairs,
        'theoretical_rdms': rdms,
    }


def main():
    out_dir = Path('stimuli')
    out_dir.mkdir(exist_ok=True)

    print("Generating decade_100...")
    d100 = generate_condition('decade_100',
        [70, 80, 90, 95, 98, 99, 100, 101, 102, 105, 110, 120, 130],
        100, FRAMINGS_100)
    path = out_dir / 'm3_stimuli_decade_100.json'
    with open(path, 'w') as f:
        json.dump(d100, f, indent=2)
    print(f"  {path}: {path.stat().st_size/1024:.1f} KB")
    print(f"  {len(d100['probing_sentences'])} probing, {len(d100['identification_stimuli'])} identification")

    print("\nGenerating control_150...")
    c150 = generate_condition('control_150',
        [130, 135, 140, 145, 150, 155, 160, 165, 170],
        150, FRAMINGS_150, is_control=True)
    path = out_dir / 'm3_stimuli_control_150.json'
    with open(path, 'w') as f:
        json.dump(c150, f, indent=2)
    print(f"  {path}: {path.stat().st_size/1024:.1f} KB")
    print(f"  {len(c150['probing_sentences'])} probing, {len(c150['identification_stimuli'])} identification")

    # Format check
    s = d100['probing_sentences'][0]
    assert all(k in s for k in ['text', 'magnitude_token', 'value', 'sentence_idx'])
    ids = d100['identification_stimuli'][0]
    assert all(k in ids for k in ['value', 'framing', 'prompt', 'targets'])
    print("\nFormat verified ✓")


if __name__ == '__main__':
    main()
