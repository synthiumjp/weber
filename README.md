# Weber's Law in Transformer Magnitude Representations

Code, stimuli, and results for:

> **Weber's Law in Transformer Magnitude Representations: Efficient Coding, Representational Geometry, and Psychophysical Laws in Language Models**
>
> JP Cacioli · *Classical Minds, Modern Machines*

**Pre-registration:** [OSF (v2.7 + v2.8 amendment)](https://osf.io/u4wp5/overview?view_only=516e8b0c44964c688f6c3161f4d16da4)
---

## Overview

Four converging psychophysics paradigms test whether transformer LLMs develop log-compressed magnitude representations consistent with efficient coding theory.

| Paradigm | Question | Method |
|----------|----------|--------|
| **A** | Is distance structure log-compressive? | RSA + AIC model comparison |
| **B** | Do models show Weber's Law behaviourally? | Forced-choice discrimination, psychometric functions |
| **C** | Does precision decrease with magnitude? | Local precision gradient analysis |
| **D** | Which layers are functionally implicated? | Activation patching along probe direction |

Three magnitude domains (numerical, temporal, spatial) across three models:

- **Llama-3-8B-Instruct** (Meta) — primary
- **Mistral-7B-Instruct-v0.3** (Mistral AI) — primary
- **Qwen-2.5-7B-Instruct** (Alibaba) — exploratory

## Repository Structure

```
weber/
├── config.json                     # Model paths, layer ranges, HF commit hashes
├── model_checksums.json            # SHA256 checksums for reproducibility
│
├── scripts/
│   ├── config.py                   # Shared configuration (imported by all scripts)
│   ├── stimuli_generation.py       # Generate all stimuli (seed 42)
│   │
│   ├── paradigm_a_extract.py       # Extract hidden states
│   ├── paradigm_a_analyse.py       # RSA, AIC, Stevens exponent
│   ├── paradigm_b_behaviour.py     # Behavioural discrimination (B1)
│   ├── paradigm_b_additional.py    # B2/B3 tasks
│   ├── paradigm_c_supplement.py    # Precision gradient analysis
│   ├── paradigm_c_robustness.py    # Normalised precision
│   ├── paradigm_d_causal.py        # Activation patching (H7)
│   ├── exploratory_e5_b1_patching.py  # E5: patching with B1-format prompts
│   │
│   ├── corpus_distribution.py      # OpenWebText magnitude frequency analysis
│   ├── shuffled_magnitude_check.py # Shuffled-magnitude control
│   ├── unit_boundary_check.py      # Unit-boundary control
│   ├── psychometric_corrected.py   # Position-corrected Weber fractions
│   ├── compute_e3_dprime.py        # E3: SDT bridge (d-prime)
│   ├── evaluate_hypotheses.py      # Formal hypothesis evaluation
│   │
│   ├── run_exploratory_models.py   # Qwen replication
│   ├── run_qwen_b1.py             # Qwen B1 cross-format
│   ├── qwen_h2_analysis.py        # Qwen H2 analysis
│   │
│   ├── generate_all_figures.py     # Individual figure generation
│   ├── generate_paper_figures.py   # Combined manuscript figures
│   ├── generate_qwen_figures.py    # Qwen-specific figures
│   │
│   ├── phase0_verify.py           # Phase 0 infrastructure verification
│   ├── phase1_compliance.py       # Pre-registration compliance audit
│   ├── prereg_finalise.py         # Pre-registration finalisation checks
│   ├── power_simulation.py        # Monte Carlo power analysis
│   └── check_*.py / debug_*.py    # Diagnostic and debugging scripts
│
├── stimuli/                        # Generated stimulus files (deterministic, seed 42)
│   ├── probing_numerical.json      # Paradigm A probing sentences
│   ├── probing_temporal.json
│   ├── probing_spatial.json
│   ├── comparison_*.json           # Paradigm A pairwise comparisons
│   ├── prompts_b1.json             # Paradigm B cross-format prompts
│   ├── prompts_b2.json             # Paradigm B approximate arithmetic
│   ├── prompts_b3.json             # Paradigm B approximate estimation
│   ├── prompts_symbolic_control.json
│   ├── paradigm_d_prompts.json     # Activation patching prompts
│   ├── digit_boundary_pairs.json   # Control: digit-boundary diagnostic
│   ├── shuffled_magnitudes.json    # Control: shuffled magnitude
│   ├── unit_boundary_check.json    # Control: unit boundary
│   └── CHECKSUMS.json              # Stimulus file checksums
│
├── results/
│   ├── paradigm_a/                 # RSA, distances, hidden states per model × domain
│   ├── paradigm_b/                 # Behavioural results per model × domain
│   ├── paradigm_c/                 # (empty — results in paradigm_a as supplements)
│   ├── paradigm_d/                 # Patching results + gate reports
│   ├── robustness/                 # Shuffled-magnitude + unit-boundary controls
│   ├── figures/                    # Individual figures (PNG)
│   ├── paper_figures/              # Combined manuscript figures (PNG)
│   ├── exploratory/                # Qwen replication results
│   ├── appendix_e/                 # Corpus distribution analysis
│   ├── power_analysis/             # Pre-registered power simulations
│   ├── prereg_finalisation/        # Model hashes, frequency matching, cross-precision
│   ├── sanity_checks/              # Token position verification
│   └── compliance/                 # Pre-registration compliance audit
│
└── models/                         # NOT in git — download from HuggingFace
```

## Requirements

- Python 3.12
- PyTorch 2.8+ with ROCm 6.4 (AMD) or CUDA
- HuggingFace Transformers
- ~16 GB VRAM

Key packages: `torch`, `transformers`, `numpy`, `scipy`, `matplotlib`, `statsmodels`

Models download automatically from HuggingFace Hub on first run. Commit hashes are recorded in `config.json` and `results/prereg_finalisation/`.

## Reproducing the Results

The complete experiment runs in under 30 minutes per model on a single GPU.

```bash
# 1. Generate stimuli (deterministic, seed 42)
cd scripts
python stimuli_generation.py

# 2. Paradigm A: extract hidden states + analyse geometry
python paradigm_a_extract.py        # all models × domains
python paradigm_a_analyse.py        # RSA, AIC, Stevens

# 3. Paradigm B: behavioural discrimination
python paradigm_b_behaviour.py      # B1 cross-format
python paradigm_b_additional.py     # B2, B3
python psychometric_corrected.py    # Weber fractions

# 4. Paradigm C: precision gradients
python paradigm_c_supplement.py
python paradigm_c_robustness.py

# 5. Paradigm D: causal intervention
python paradigm_d_causal.py
python exploratory_e5_b1_patching.py

# 6. Controls
python shuffled_magnitude_check.py
python unit_boundary_check.py
python corpus_distribution.py

# 7. Figures
python generate_all_figures.py
python generate_paper_figures.py

# 8. Formal evaluation
python evaluate_hypotheses.py
python phase1_compliance.py
```

## Hardware

All experiments ran on:

- AMD Radeon RX 7900 GRE (16 GB VRAM)
- Windows, `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- PyTorch 2.8 with ROCm 6.4

CUDA users can ignore the ROCm environment variable. Any GPU supporting HuggingFace `output_hidden_states=True` should work.

## Key Findings

1. **Log-compressive geometry is universal.** RSA ρ = .68–.96 across all 96 model × domain × layer cells.
2. **Geometry dissociates from behaviour.** Llama and Qwen show human-range Weber fractions (WF ≈ 0.20); Mistral does not. Temporal/spatial: chance performance despite strong geometry.
3. **Causal layer inversion.** Early layers (weak geometry) are functionally active (4.1× specificity); late layers (strong geometry) are not causally engaged (1.2×).
4. **Geometry is a pretraining property; behaviour is an instruction-tuning property.** Base Llama has equal geometry but zero behavioural competence.

## Citation

```bibtex
@article{cacioli2026weber,
  title={Weber's Law in Transformer Magnitude Representations: Efficient Coding, Representational Geometry, and Psychophysical Laws in Language Models},
  author={Cacioli, JP},
  year={2026},
  note={Pre-registered: \url{https://osf.io/u4wp5}}
}
```

## Licence

MIT
