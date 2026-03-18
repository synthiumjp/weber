#!/usr/bin/env python3
"""
Patch paradigm_b_behaviour.py to fix temporal/spatial domain stimulus routing.

Bug: load_comparison_stimuli always loads numerical prompt files regardless of domain.
Fix: For temporal/spatial, load from comparison_temporal.json / comparison_spatial.json
     and generate domain-appropriate B1 prompts.

Usage: python scripts/patch_paradigm_b_domains.py
"""

from pathlib import Path
import re

SCRIPT = Path(r"C:\weber\scripts\paradigm_b_behaviour.py")

# Read the original
with open(SCRIPT, 'r', encoding='utf-8') as f:
    content = f.read()

# Back up
backup = SCRIPT.with_suffix('.py.bak')
with open(backup, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Backup saved to {backup}")

# === PATCH 1: Replace load_comparison_stimuli ===
# Find the function and replace it entirely

old_load_fn = '''def load_comparison_stimuli(domain_key: str, stimuli_dir: Path) -> list[dict]:
    """
    Load pre-generated comparison stimuli from archived JSON files.

    Handles the actual stimulus file format from stimuli_generation.py:
        prompts_b1.json \u2014 B1 cross-format comparison (PRIMARY for H2)
        prompts_b2.json \u2014 B2 approximate arithmetic
        prompts_b3.json \u2014 B3 contextual comparison
        prompts_symbolic_control.json \u2014 symbolic comparison control

    Fields in the archive:
        pair_id, task, prompt, a_expression, b_expression,
        correct_answer, nominal_ratio, nominal_baseline

    We normalise field names to what the analysis pipeline expects.
    """
    # Map task types to filenames
    task_files = {
        "B1": stimuli_dir / "prompts_b1.json",
        "B2": stimuli_dir / "prompts_b2.json",
        "B3": stimuli_dir / "prompts_b3.json",
        "symbolic": stimuli_dir / "prompts_symbolic_control.json",
    }'''

# Since the file has encoding issues with em-dashes, match more flexibly
# Find the function start
fn_start = content.find('def load_comparison_stimuli(domain_key: str, stimuli_dir: Path)')
if fn_start == -1:
    print("ERROR: Could not find load_comparison_stimuli function")
    exit(1)

# Find where task_files dict ends (look for the closing brace after "symbolic")
task_files_end = content.find('"symbolic": stimuli_dir / "prompts_symbolic_control.json",', fn_start)
if task_files_end == -1:
    print("ERROR: Could not find task_files dict")
    exit(1)
# Find the closing brace of the dict
closing_brace = content.find('}', task_files_end)
if closing_brace == -1:
    print("ERROR: Could not find closing brace")
    exit(1)

# We need to insert domain routing AFTER the task_files dict but BEFORE the all_stimuli loop
# Find "all_stimuli = []" after the dict
all_stimuli_line = content.find('all_stimuli = []', closing_brace)
if all_stimuli_line == -1:
    print("ERROR: Could not find all_stimuli = []")
    exit(1)

# Insert domain routing code between the closing brace and all_stimuli
domain_routing = '''
    # === DOMAIN ROUTING (patched) ===
    # For temporal/spatial, use domain-specific comparison files instead of
    # numerical prompt files. B2/B3/symbolic are numerical-only tasks.
    if domain_key in ("temporal", "spatial"):
        comp_file = stimuli_dir / f"comparison_{domain_key}.json"
        if comp_file.exists():
            log.info(f"Loading domain-specific stimuli from {comp_file.name}")
            return _load_domain_comparison(domain_key, comp_file)
        else:
            log.warning(f"Domain comparison file not found: {comp_file}")
            log.warning("Falling through to generate_comparison_stimuli")
            return generate_comparison_stimuli(domain_key)

'''

# Insert the routing code
content = content[:all_stimuli_line] + domain_routing + '    ' + content[all_stimuli_line:]

# === PATCH 2: Add _load_domain_comparison helper function ===
# Insert it right before generate_comparison_stimuli

gen_fn_start = content.find('def generate_comparison_stimuli(domain_key: str)')
if gen_fn_start == -1:
    print("ERROR: Could not find generate_comparison_stimuli")
    exit(1)

helper_fn = '''
def _load_domain_comparison(domain_key: str, comp_file: Path) -> list[dict]:
    """
    Load temporal/spatial comparison stimuli and generate B1 prompts.

    comparison_temporal.json fields: pair_id, domain, nominal_baseline,
        baseline_seconds, comparison_seconds, first_presented, second_presented,
        correct_answer, nominal_ratio, actual_ratio, pair_index_in_cell

    comparison_spatial.json has equivalent structure with metres.

    Only B1 is generated for temporal/spatial (B2/B3 are numerical-specific).
    Symbolic control is also included using the raw numbers.
    """
    with open(comp_file) as f:
        pairs = json.load(f)

    log.info(f"  Loaded {len(pairs)} comparison pairs for {domain_key}")

    # Domain-specific prompt templates
    if domain_key == "temporal":
        b1_template = "Which duration is longer: A) {a} or B) {b}? Answer with only A or B."
        sym_template = "Which is larger: A) {a} or B) {b}? Answer with only A or B."
    elif domain_key == "spatial":
        b1_template = "Which distance is greater: A) {a} or B) {b}? Answer with only A or B."
        sym_template = "Which is larger: A) {a} or B) {b}? Answer with only A or B."
    else:
        raise ValueError(f"Unsupported domain for comparison loading: {domain_key}")

    stimuli = []

    for pair in pairs:
        first = pair.get("first_presented", "")
        second = pair.get("second_presented", "")
        correct = pair.get("correct_answer", "B")
        ratio = pair.get("nominal_ratio", pair.get("actual_ratio", 1.0))

        # Extract nominal baseline (may be string like "10 seconds")
        nb = pair.get("nominal_baseline", "")
        if isinstance(nb, str):
            # Parse number from string like "10 seconds" or "200 metres"
            import re as _re
            m = _re.search(r"[\\d.]+", nb)
            baseline = float(m.group()) if m else 0.0
        else:
            baseline = float(nb)

        # B1: cross-format (uses presented strings directly)
        b1_prompt = b1_template.format(a=first, b=second)
        stimuli.append({
            "task_type": "B1",
            "domain": domain_key,
            "baseline": baseline,
            "ratio": float(ratio),
            "value_a": first,
            "value_b": second,
            "correct_answer": correct,
            "prompt": b1_prompt,
            "pair_id": pair.get("pair_id", ""),
        })

        # Symbolic control: use raw numbers
        if domain_key == "temporal":
            raw_a = pair.get("baseline_seconds", 0)
            raw_b = pair.get("comparison_seconds", 0)
        elif domain_key == "spatial":
            # spatial files may use baseline_metres / comparison_metres
            raw_a = pair.get("baseline_metres", pair.get("baseline_seconds", 0))
            raw_b = pair.get("comparison_metres", pair.get("comparison_seconds", 0))

        # Position matches B1
        if correct == "B":
            sym_a, sym_b, sym_correct = raw_a, raw_b, "B"
        else:
            sym_a, sym_b, sym_correct = raw_b, raw_a, "A"

        sym_prompt = sym_template.format(a=sym_a, b=sym_b)
        stimuli.append({
            "task_type": "symbolic_control",
            "domain": domain_key,
            "baseline": baseline,
            "ratio": float(ratio),
            "value_a": str(sym_a),
            "value_b": str(sym_b),
            "correct_answer": str(max(sym_a, sym_b)),
            "prompt": sym_prompt,
            "pair_id": pair.get("pair_id", ""),
        })

    log.info(f"  Generated {len(stimuli)} stimuli ({len(pairs)} B1 + {len(pairs)} symbolic)")
    return stimuli


'''

content = content[:gen_fn_start] + helper_fn + content[gen_fn_start:]

# Write patched file
with open(SCRIPT, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Patched {SCRIPT}")
print("Changes:")
print("  1. load_comparison_stimuli now routes temporal/spatial to comparison_*.json")
print("  2. Added _load_domain_comparison() to generate domain-appropriate prompts")
print("  3. Temporal: 'Which duration is longer: A) X or B) Y?'")
print("  4. Spatial: 'Which distance is greater: A) X or B) Y?'")
print("  5. B2/B3 skipped for temporal/spatial (numerical-specific tasks)")
print(f"\nBackup at: {backup}")
print("\nRe-run:")
print("  python scripts/paradigm_b_behaviour.py --model llama_instruct --domain temporal")
print("  python scripts/paradigm_b_behaviour.py --model llama_instruct --domain spatial")
print("  python scripts/paradigm_b_behaviour.py --model mistral_instruct --domain temporal")
print("  python scripts/paradigm_b_behaviour.py --model mistral_instruct --domain spatial")
