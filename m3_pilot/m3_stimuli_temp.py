"""
M3 Stimulus Generation: Temperature Domain
===========================================
Generates stimulus files for temperature categorical perception.

Domain: outdoor air temperature in degrees Celsius.
Boundary: hot/cold (~20-22°C in English-language contexts).
Pre-registered as secondary domain for H6 (cross-domain generality).

Conditions:
  - temp_hotcold: probing values spanning cold→hot, boundary ~20-22°C
  - temp_control: probing values in a region with no categorical boundary

Key design differences from numerical domain:
  - Functional form unknown (not necessarily log-compressive)
  - Pre-reg: fit linear/log/power to RDM BEFORE testing CP
  - Boundary empirically located via "hot or cold?" identification
  - No tokenisation confound (all temperatures are multi-token)
  - Context fixed: outdoor air temperature

Carrier sentences (5 from pre-reg spec + 3 added for consistency with numerical):
  Following Weber: 8 carrier sentences, B0/RSA split (0-3 / 4-7).

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import json
import math
import itertools
import numpy as np
from pathlib import Path


# ==============================================================================
# Temperature probing values (from pre-registration)
# ==============================================================================

# Primary condition: full cold-to-hot continuum
# Dense around expected boundary (18-25°C), sparser at extremes
TEMP_VALUES = [-20, -10, 0, 5, 10, 15, 18, 20, 22, 25, 28, 30, 35, 40, 50, 60, 80, 100]

# Control: values within a single category (all clearly hot, no boundary)
# 35-55°C range — all unambiguously "hot", no categorical transition
TEMP_CONTROL_VALUES = [35, 37, 39, 41, 43, 45, 47, 49, 51]

# Structural boundary estimate (will be refined per model via identification)
TEMP_BOUNDARY = 22  # ~22°C is typical warm/cool crossover in English


# ==============================================================================
# Carrier sentences — outdoor air temperature context
# ==============================================================================

CARRIER_SENTENCES = [
    # B0 identification (indices 0-3)
    "The temperature outside was {T} degrees.",
    "The forecast predicted {T} degrees for the afternoon.",
    "At {T} degrees, the conditions were notable.",
    "The thermometer read {T} degrees.",
    # RSA centroid computation (indices 4-7)
    "The weather report showed {T} degrees Celsius.",
    "It was {T} degrees outside that day.",
    "The outdoor temperature reached {T} degrees.",
    "Conditions were measured at {T} degrees Celsius.",
]

B0_INDICES = [0, 1, 2, 3]
RSA_INDICES = [4, 5, 6, 7]


# ==============================================================================
# Identification framings
# ==============================================================================

FRAMINGS_HOTCOLD = [
    {
        'name': 'hot_cold',
        'template': "Is {T} degrees Celsius hot or cold?",
        'category_a_label': 'cold',
        'category_b_label': 'hot',
        'targets': {'category_a': ['cold', 'Cold'], 'category_b': ['hot', 'Hot']},
    },
    {
        'name': 'warm_cool',
        'template': "Is {T} degrees Celsius warm or cool?",
        'category_a_label': 'cool',
        'category_b_label': 'warm',
        'targets': {'category_a': ['cool', 'Cool'], 'category_b': ['warm', 'Warm']},
    },
    {
        'name': 'comfortable',
        'template': "Is {T} degrees Celsius comfortable or uncomfortable?",
        'category_a_label': 'uncomfortable (cold)',
        'category_b_label': 'comfortable',
        'targets': {'category_a': ['uncomfortable', 'Uncomfortable', 'un'], 'category_b': ['comfortable', 'Comfortable', 'com']},
    },
]

FRAMINGS_CONTROL = [
    {
        'name': 'hot_cold',
        'template': "Is {T} degrees Celsius hot or cold?",
        'category_a_label': 'cold',
        'category_b_label': 'hot',
        'targets': {'category_a': ['cold', 'Cold'], 'category_b': ['hot', 'Hot']},
    },
]


def format_temp(t):
    """Format temperature for insertion into sentences.
    Use numeric form directly — matches how models encounter temperatures.
    """
    return str(t)


def get_magnitude_token(t):
    """Get the magnitude token that m3_extract.py will search for.
    For negative numbers, the token is the number part (e.g., '20' from '-20').
    For positive numbers, it's the number itself.
    For multi-token numbers, we need the last token of the number.
    """
    # Use the absolute value string — the sign is a separate token
    return str(abs(t))


def build_probing_sentences(values):
    """Build flat list matching m3_extract.py format."""
    sentences = []
    for val in values:
        t_str = format_temp(val)
        mag_tok = get_magnitude_token(val)
        for idx, template in enumerate(CARRIER_SENTENCES):
            text = template.format(T=t_str)
            sentences.append({
                'text': text,
                'magnitude_token': mag_tok,
                'value': val,
                'sentence_idx': idx,
                'is_rsa': idx in RSA_INDICES,
                'is_b0': idx in B0_INDICES,
            })
    return sentences


def build_identification_stimuli(values, framings):
    """Build flat list matching m3_extract.py format."""
    id_stims = []
    for val in values:
        for framing in framings:
            prompt = framing['template'].format(T=val)
            id_stims.append({
                'value': val,
                'framing': framing['name'],
                'prompt': prompt,
                'targets': framing['targets'],
                'category_a': framing['category_a_label'],
                'category_b': framing['category_b_label'],
            })
    return id_stims


def compute_theoretical_rdms(values, boundary):
    """
    Compute theoretical RDMs for temperature.
    
    Unlike numerical domain, we don't know the functional form.
    Pre-reg says: fit linear/log/power BEFORE testing CP.
    
    For now, build both linear and log versions of continuous,
    plus categorical and CP-additive with both bases.
    
    Note: negative temperatures break log transform. Use absolute
    temperature offset: T_eff = T - min(T) + 1 for log models.
    """
    n = len(values)
    vals = np.array(values, dtype=float)
    
    # Linear continuous: |Ti - Tj|
    linear = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = abs(vals[i] - vals[j])
        linear[i, j] = linear[j, i] = d
    
    # Log continuous: |log(Ti_eff) - log(Tj_eff)| where T_eff = T - min + 1
    t_min = vals.min()
    t_eff = vals - t_min + 1  # shift to positive
    log_cont = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = abs(math.log(t_eff[i]) - math.log(t_eff[j]))
        log_cont[i, j] = log_cont[j, i] = d
    
    # Categorical: 0 same side of boundary, 1 different
    categorical = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        cat_i = 0 if vals[i] < boundary else 1
        cat_j = 0 if vals[j] < boundary else 1
        if cat_i != cat_j:
            categorical[i, j] = categorical[j, i] = 1.0
    
    # CP-Additive (linear base): linear + λ * categorical
    lam_lin = float(np.mean(linear[np.triu_indices(n, k=1)]))
    cp_additive_linear = linear + lam_lin * categorical
    
    # CP-Additive (log base): log + λ * categorical
    lam_log = float(np.mean(log_cont[np.triu_indices(n, k=1)]))
    cp_additive_log = log_cont + lam_log * categorical
    
    rdms = {
        'continuous': linear.tolist(),  # primary: linear (unknown form)
        'continuous_log': log_cont.tolist(),  # alternative: log
        'categorical': categorical.tolist(),
        'cp_additive': cp_additive_linear.tolist(),  # CP with linear base
        'cp_additive_log': cp_additive_log.tolist(),  # CP with log base
        'linear': linear.tolist(),  # same as continuous for temperature
        'lambda': lam_lin,
        'lambda_log': lam_log,
    }
    
    # Validate
    for name, rdm_list in rdms.items():
        if isinstance(rdm_list, (float, int)):
            continue
        rdm = np.array(rdm_list)
        assert rdm.shape == (n, n) and np.allclose(rdm, rdm.T) and np.allclose(np.diag(rdm), 0), \
            f"{name} validation failed"
    
    return rdms


def build_pairs(values, boundary):
    pairs = []
    for i, j in itertools.combinations(range(len(values)), 2):
        vi, vj = values[i], values[j]
        pairs.append({
            'i': i, 'j': j,
            'value_i': vi, 'value_j': vj,
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
        'domain': 'temperature',
        'boundary': boundary,
        'boundary_unit': 'degrees_celsius',
        'is_control': is_control,
        'probing_values': values,
        'n_values': len(values),
        'probing_sentences': probing_sentences,
        'identification_stimuli': identification_stimuli,
        'n_pairs': len(pairs),
        'pairs': pairs,
        'theoretical_rdms': rdms,
        'notes': 'Temperature domain. Boundary is approximate (~20-22C). '
                 'Empirical boundary from identification task per model. '
                 'Functional form (linear vs log) determined from RSA fit.',
    }


def main():
    out_dir = Path('stimuli')
    out_dir.mkdir(exist_ok=True)

    # Primary: hot/cold continuum
    print("Generating temp_hotcold...")
    hotcold = generate_condition('temp_hotcold', TEMP_VALUES, TEMP_BOUNDARY, FRAMINGS_HOTCOLD)
    path = out_dir / 'm3_stimuli_temp_hotcold.json'
    with open(path, 'w') as f:
        json.dump(hotcold, f, indent=2)
    print(f"  {path}: {path.stat().st_size/1024:.1f} KB")
    print(f"  {len(hotcold['probing_sentences'])} probing, "
          f"{len(hotcold['identification_stimuli'])} identification")
    print(f"  Values: {TEMP_VALUES}")
    print(f"  Boundary: {TEMP_BOUNDARY}°C")

    # Control: all-hot region
    print("\nGenerating temp_control...")
    control = generate_condition('temp_control', TEMP_CONTROL_VALUES, 43,
                                 FRAMINGS_CONTROL, is_control=True)
    path = out_dir / 'm3_stimuli_temp_control.json'
    with open(path, 'w') as f:
        json.dump(control, f, indent=2)
    print(f"  {path}: {path.stat().st_size/1024:.1f} KB")
    print(f"  {len(control['probing_sentences'])} probing, "
          f"{len(control['identification_stimuli'])} identification")
    print(f"  Values: {TEMP_CONTROL_VALUES}")

    # Format check
    s = hotcold['probing_sentences'][0]
    assert all(k in s for k in ['text', 'magnitude_token', 'value', 'sentence_idx'])
    ids = hotcold['identification_stimuli'][0]
    assert all(k in ids for k in ['value', 'framing', 'prompt', 'targets'])
    print("\nFormat verified ✓")

    # Show example sentences
    print("\nExample sentences:")
    for v in [-20, 0, 20, 22, 40, 100]:
        sents = [s for s in hotcold['probing_sentences'] if s['value'] == v]
        if sents:
            print(f"  {v}°C: \"{sents[0]['text']}\" (token: '{sents[0]['magnitude_token']}')")

    print(f"\nExample identification:")
    for f in hotcold['identification_stimuli'][:3]:
        print(f"  {f['value']}°C [{f['framing']}]: \"{f['prompt']}\"")


if __name__ == '__main__':
    main()
