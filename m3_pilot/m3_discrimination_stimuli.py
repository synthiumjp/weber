"""
M3 Paradigm B: Discrimination Stimulus Generation
===================================================
Generates log-distance-matched discrimination pairs for the CP behavioural test.

Design: 2 (position: cross-boundary vs within-category) × 6 (log-distance) factorial.
Boundary: structurally defined at 10 (digit count transition).
Per cell: 50 pairs with jittered base values (±10%, seed 42).
Total: 600 trials per model.

Pair types:
  - Cross-boundary: one value < 10, one value ≥ 10
  - Within-category-below: both values < 10
  - Within-category-above: both values ≥ 10

Log-distances chosen to span the range achievable in both cross-boundary
and within-category conditions given the probing range (4–20).

Each trial specifies:
  - pair_id, x1, x2 (x1 < x2 always, presentation order counterbalanced at runtime)
  - log_distance: |log(x2) - log(x1)|
  - target_log_distance: the factorial cell target
  - position: 'cross_boundary', 'within_below', or 'within_above'
  - correct_answer: always 'B' when presented as A=x1, B=x2 (x1 < x2)

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import json
import math
import numpy as np
from pathlib import Path


# ==============================================================================
# Design parameters
# ==============================================================================

BOUNDARY = 10
SEED = 42
N_PAIRS_PER_CELL = 50
JITTER_FRACTION = 0.10  # ±10%

# Probing range: integers 4–20 (matching Paradigm A decade_10 range)
PROBE_MIN = 4
PROBE_MAX = 20

# Log-distance levels for the factorial design.
# These must be achievable in BOTH cross-boundary and within-category conditions.
# Range constraint:
#   - Within-below: max log-distance = |log(9) - log(4)| = 0.811
#   - Within-above: max log-distance = |log(20) - log(10)| = 0.693
#   - Cross-boundary: min ≈ |log(10) - log(9)| = 0.105, max ≈ |log(20) - log(4)| = 1.609
# The binding constraint is within-above (max 0.693).
# Choose 6 levels from 0.10 to 0.60 in steps of 0.10.
LOG_DISTANCE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

# Tolerance for log-distance matching (how close a jittered pair must be to target)
LOG_DISTANCE_TOLERANCE = 0.02


def generate_base_pairs(position: str, target_log_dist: float, rng: np.random.Generator) -> list[dict]:
    """
    Generate N_PAIRS_PER_CELL pairs for a given position × log-distance cell.
    
    Strategy: sample a base value, compute the partner that gives the target
    log-distance, jitter both by ±JITTER_FRACTION, verify the pair still
    satisfies boundary-crossing constraints and distance tolerance.
    
    For cross-boundary pairs: base is below boundary, partner is above.
    For within-below: both below boundary (< 10).
    For within-above: both above or at boundary (≥ 10).
    """
    pairs = []
    max_attempts = N_PAIRS_PER_CELL * 500  # generous retry budget
    attempts = 0
    
    while len(pairs) < N_PAIRS_PER_CELL and attempts < max_attempts:
        attempts += 1
        
        if position == 'cross_boundary':
            # Base below boundary, partner above
            # Sample base from [4, 9] range
            base = rng.uniform(PROBE_MIN, BOUNDARY - 0.5)
            # Partner: base * exp(target_log_dist)
            partner = base * math.exp(target_log_dist)
            # Partner must be ≥ 10 and ≤ 20
            if partner < BOUNDARY or partner > PROBE_MAX:
                continue
                
        elif position == 'within_below':
            # Both below boundary (< 10)
            # Sample base, partner = base * exp(target_log_dist), both < 10
            base = rng.uniform(PROBE_MIN, BOUNDARY - 0.5)
            partner = base * math.exp(target_log_dist)
            if partner >= BOUNDARY or partner > PROBE_MAX:
                continue
            if base < PROBE_MIN:
                continue
                
        elif position == 'within_above':
            # Both ≥ 10
            base = rng.uniform(BOUNDARY, PROBE_MAX - 1)
            partner = base * math.exp(target_log_dist)
            if partner > PROBE_MAX or partner < BOUNDARY:
                continue
        else:
            raise ValueError(f"Unknown position: {position}")
        
        # Apply jitter to both values
        base_jittered = base * (1 + rng.uniform(-JITTER_FRACTION, JITTER_FRACTION))
        partner_jittered = partner * (1 + rng.uniform(-JITTER_FRACTION, JITTER_FRACTION))
        
        # Clamp to probing range
        base_jittered = max(PROBE_MIN, min(base_jittered, PROBE_MAX))
        partner_jittered = max(PROBE_MIN, min(partner_jittered, PROBE_MAX))
        
        # Ensure x1 < x2
        x1, x2 = sorted([base_jittered, partner_jittered])
        
        # Verify log-distance is within tolerance
        actual_log_dist = abs(math.log(x2) - math.log(x1))
        if abs(actual_log_dist - target_log_dist) > LOG_DISTANCE_TOLERANCE:
            continue
        
        # Verify boundary-crossing constraint still holds after jitter
        if position == 'cross_boundary':
            if not (x1 < BOUNDARY and x2 >= BOUNDARY):
                continue
        elif position == 'within_below':
            if not (x1 < BOUNDARY and x2 < BOUNDARY):
                continue
        elif position == 'within_above':
            if not (x1 >= BOUNDARY and x2 >= BOUNDARY):
                continue
        
        # Ensure values are distinguishable (not too close after rounding)
        if abs(x1 - x2) < 0.01:
            continue
        
        pairs.append({
            'x1': round(float(x1), 2),
            'x2': round(float(x2), 2),
            'log_distance': round(actual_log_dist, 4),
            'target_log_distance': target_log_dist,
            'position': position,
        })
    
    if len(pairs) < N_PAIRS_PER_CELL:
        print(f"  WARNING: Only generated {len(pairs)}/{N_PAIRS_PER_CELL} pairs "
              f"for {position} at log-dist {target_log_dist}")
    
    return pairs


def generate_all_pairs() -> dict:
    """
    Generate the full 2 × 6 factorial stimulus set.
    
    Returns dict with:
      - design: metadata
      - trials: list of trial dicts
      - summary: cell counts
    """
    rng = np.random.default_rng(SEED)
    
    # Position conditions:
    # For the primary analysis (H2), we collapse within_below and within_above
    # into a single "within-category" condition. But we generate them separately
    # to ensure balanced sampling from both sides of the boundary.
    positions = ['cross_boundary', 'within_below', 'within_above']
    
    all_trials = []
    trial_id = 0
    summary = {}
    
    for position in positions:
        for log_dist in LOG_DISTANCE_LEVELS:
            pairs = generate_base_pairs(position, log_dist, rng)
            
            for pair in pairs:
                pair['trial_id'] = trial_id
                # Correct answer when presented as A=x1, B=x2 (x1 < x2): always B
                pair['correct_larger'] = 'x2'
                all_trials.append(pair)
                trial_id += 1
            
            cell_key = f"{position}_logdist{log_dist:.2f}"
            summary[cell_key] = len(pairs)
    
    # Build prompt templates
    # Following Weber Paradigm B: "Which is larger: A) [x1] or B) [x2]?"
    # At runtime, counterbalance A/B assignment (x1 as A or x1 as B)
    prompt_template = "Which is larger: A) {val_a} or B) {val_b}?"
    
    # For the analysis, within_below and within_above are collapsed
    # into "within_category" for the primary H2 test
    for trial in all_trials:
        if trial['position'] in ('within_below', 'within_above'):
            trial['position_collapsed'] = 'within_category'
        else:
            trial['position_collapsed'] = 'cross_boundary'
    
    design = {
        'paradigm': 'B',
        'description': 'Behavioural discrimination — log-distance-matched pairs',
        'boundary': BOUNDARY,
        'boundary_type': 'structural',
        'seed': SEED,
        'n_pairs_per_cell': N_PAIRS_PER_CELL,
        'jitter_fraction': JITTER_FRACTION,
        'log_distance_levels': LOG_DISTANCE_LEVELS,
        'log_distance_tolerance': LOG_DISTANCE_TOLERANCE,
        'positions': positions,
        'probe_range': [PROBE_MIN, PROBE_MAX],
        'prompt_template': prompt_template,
        'counterbalanced': True,
        'n_total_trials': len(all_trials),
        'scoring': 'greedy_decoding_T0_logit_extraction',
    }
    
    return {
        'design': design,
        'trials': all_trials,
        'summary': summary,
    }


def format_value_for_prompt(x: float) -> str:
    """
    Format a numerical value for the discrimination prompt.
    
    Use integers when possible (cleaner tokenisation), otherwise 1 decimal.
    Values in our range (4–20) will mostly be near-integers after jitter.
    """
    # Round to integer if within 0.05
    if abs(x - round(x)) < 0.05:
        return str(int(round(x)))
    else:
        return f"{x:.1f}"


def build_trial_prompts(trial: dict) -> dict:
    """
    Build the two counterbalanced prompt versions for a trial.
    
    Order 1: A = x1 (smaller), B = x2 (larger) → correct = B
    Order 2: A = x2 (larger), B = x1 (smaller) → correct = A
    """
    x1_str = format_value_for_prompt(trial['x1'])
    x2_str = format_value_for_prompt(trial['x2'])
    
    return {
        'order1': {
            'prompt': f"Which is larger: A) {x1_str} or B) {x2_str}?",
            'val_a': trial['x1'],
            'val_b': trial['x2'],
            'correct': 'B',
        },
        'order2': {
            'prompt': f"Which is larger: A) {x2_str} or B) {x1_str}?",
            'val_a': trial['x2'],
            'val_b': trial['x1'],
            'correct': 'A',
        },
    }


def validate_design(stim: dict) -> None:
    """Validation checks on the generated stimulus set."""
    trials = stim['trials']
    design = stim['design']
    
    print(f"\n{'='*60}")
    print("PARADIGM B STIMULUS VALIDATION")
    print(f"{'='*60}")
    
    # 1. Total trial count
    expected_total = len(design['positions']) * len(design['log_distance_levels']) * design['n_pairs_per_cell']
    print(f"\nTotal trials: {len(trials)} (expected: {expected_total})")
    
    # 2. Cell counts
    print(f"\nCell counts:")
    for cell, count in sorted(stim['summary'].items()):
        status = "✓" if count == N_PAIRS_PER_CELL else "✗"
        print(f"  {status} {cell}: {count}")
    
    # 3. Log-distance accuracy
    print(f"\nLog-distance accuracy (tolerance = {LOG_DISTANCE_TOLERANCE}):")
    for log_dist in LOG_DISTANCE_LEVELS:
        subset = [t for t in trials if t['target_log_distance'] == log_dist]
        actual_dists = [t['log_distance'] for t in subset]
        if actual_dists:
            mean_err = np.mean([abs(d - log_dist) for d in actual_dists])
            max_err = max(abs(d - log_dist) for d in actual_dists)
            print(f"  Target {log_dist:.2f}: n={len(subset)}, "
                  f"mean |error|={mean_err:.4f}, max |error|={max_err:.4f}")
    
    # 4. Boundary crossing verification
    print(f"\nBoundary crossing verification:")
    for pos in design['positions']:
        subset = [t for t in trials if t['position'] == pos]
        if pos == 'cross_boundary':
            violations = sum(1 for t in subset if not (t['x1'] < BOUNDARY and t['x2'] >= BOUNDARY))
        elif pos == 'within_below':
            violations = sum(1 for t in subset if not (t['x1'] < BOUNDARY and t['x2'] < BOUNDARY))
        elif pos == 'within_above':
            violations = sum(1 for t in subset if not (t['x1'] >= BOUNDARY and t['x2'] >= BOUNDARY))
        status = "✓" if violations == 0 else "✗"
        print(f"  {status} {pos}: {len(subset)} pairs, {violations} violations")
    
    # 5. Value range
    all_x1 = [t['x1'] for t in trials]
    all_x2 = [t['x2'] for t in trials]
    all_vals = all_x1 + all_x2
    print(f"\nValue range: [{min(all_vals):.2f}, {max(all_vals):.2f}] "
          f"(expected: [{PROBE_MIN}, {PROBE_MAX}])")
    
    # 6. Collapsed position balance
    cross = sum(1 for t in trials if t['position_collapsed'] == 'cross_boundary')
    within = sum(1 for t in trials if t['position_collapsed'] == 'within_category')
    print(f"\nCollapsed position counts:")
    print(f"  Cross-boundary: {cross}")
    print(f"  Within-category: {within} (below + above)")
    
    # 7. Log-distance distribution per collapsed position
    print(f"\nLog-distance × position (collapsed) cell counts:")
    for log_dist in LOG_DISTANCE_LEVELS:
        cross_n = sum(1 for t in trials 
                      if t['position_collapsed'] == 'cross_boundary' 
                      and t['target_log_distance'] == log_dist)
        within_n = sum(1 for t in trials 
                       if t['position_collapsed'] == 'within_category' 
                       and t['target_log_distance'] == log_dist)
        print(f"  log-dist {log_dist:.2f}: cross={cross_n}, within={within_n}")
    
    print(f"\n{'='*60}")


def main():
    print("M3 Paradigm B: Discrimination Stimulus Generation")
    print("=" * 50)
    
    stim = generate_all_pairs()
    
    # Add prompt versions to each trial
    for trial in stim['trials']:
        trial['prompts'] = build_trial_prompts(trial)
    
    # Validate
    validate_design(stim)
    
    # Save
    out_dir = Path('stimuli')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'm3_discrimination_stimuli.json'
    
    with open(out_path, 'w') as f:
        json.dump(stim, f, indent=2)
    
    print(f"\nSaved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")
    
    # Print example trials
    print(f"\nExample trials:")
    for pos in ['cross_boundary', 'within_below', 'within_above']:
        subset = [t for t in stim['trials'] if t['position'] == pos]
        if subset:
            t = subset[0]
            print(f"\n  [{pos}] trial {t['trial_id']}:")
            print(f"    x1={t['x1']}, x2={t['x2']}, log_dist={t['log_distance']:.4f} "
                  f"(target={t['target_log_distance']:.2f})")
            print(f"    O1: {t['prompts']['order1']['prompt']} → correct={t['prompts']['order1']['correct']}")
            print(f"    O2: {t['prompts']['order2']['prompt']} → correct={t['prompts']['order2']['correct']}")


if __name__ == '__main__':
    main()
