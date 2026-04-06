# Classical Minds, Modern Machines — Psychophysics of LLM Representations

This repository contains code, stimuli, and results for three papers in the *Classical Minds, Modern Machines* programme, which applies formal cognitive science tools to large language model representations.

**Anonymous repository:** [anonymous.4open.science/r/weber-B02C](https://anonymous.4open.science/r/weber-B02C/README.md)

---

## Paper 1: Weber's Law in Transformer Magnitude Representations

> **Weber's Law in Transformer Magnitude Representations: Efficient Coding, Representational Geometry, and Psychophysical Laws in Language Models**

**Pre-registration:** [OSF (v2.7 + v2.8 amendment)](https://osf.io/u4wp5/overview?view_only=516e8b0c44964c688f6c3161f4d16da4)

**Status:** Under review at *Computational Brain & Behavior*

**Code:** Root directory (`scripts/`, `stimuli/`, `results/`)

### Overview

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

### Key Findings

1. **Log-compressive geometry is universal.** RSA ρ = .68–.96 across all 96 model × domain × layer cells.
2. **Geometry dissociates from behaviour.** Llama and Qwen show human-range Weber fractions (WF ≈ 0.20); Mistral does not. Temporal/spatial: chance performance despite strong geometry.
3. **Causal layer inversion.** Early layers (weak geometry) are functionally active (4.1× specificity); late layers (strong geometry) are not causally engaged (1.2×).
4. **Geometry is a pretraining property; behaviour is an instruction-tuning property.** Base Llama has equal geometry but zero behavioural competence.

---

## Paper 2: Categorical Perception in LLM Hidden States

> **Categorical Perception in Large Language Model Hidden States: Structural Warping at Digit-Count Boundaries**

**Pre-registration:** [OSF](https://osf.io/qrxf3/overview?view_only=4afe7fbd28764087a65a4222578ad625)

**Status:** Targeting *Neural Computation*

**Code:** `m3_pilot/` directory

### Overview

Five paradigms test whether LLM hidden states exhibit categorical perception — geometric warping at digit-count boundaries — using the Weber paper's log-compressive geometry as the continuous baseline.

| Paradigm | Question | Method |
|----------|----------|--------|
| **A** | Does representational geometry warp at boundaries? | RSA + Mantel permutation tests |
| **B0** | Can models explicitly identify the category? | Counterbalanced forced-choice identification |
| **B** | Does discrimination confidence track boundaries? | Forced-choice with |Δlogit| confidence |
| **C** | Does precision spike at the boundary? | Local precision gradient |
| **E** | Are boundary representations causally implicated? | Activation patching along category direction |

Six models from five architecture families:

- **Llama-3-8B-Instruct** (Meta) — primary
- **Mistral-7B-Instruct-v0.3** (Mistral AI) — primary
- **Gemma-2-9B-IT** (Google) — primary
- **Qwen2.5-7B-Instruct** (Alibaba) — primary
- **Phi-3.5-mini-instruct** (Microsoft) — primary (scale probe)
- **Llama-3-8B-Base** (Meta) — exploratory (instruction-tuning control)

Three control conditions: temperature domain (linguistic boundary, no tokenisation discontinuity), non-boundary positions (15, 150), nonce-token remapping (ordinal context without numerical surface form).

### Key Findings

1. **Universal geometric CP.** CP-Additive > Continuous at 100% of primary layers for all 6 models at both decade boundaries (10 and 100).
2. **Structural, not semantic.** Temperature domain shows no CP; nonce tokens show 3–10× weaker effects than real numbers. Tokenisation discontinuity is the dominant driver.
3. **Classic vs structural CP dissociation.** Gemma/Qwen show both identification and geometry ("classic CP"); Llama/Mistral/Phi show geometry without identification ("structural CP"). The dissociation is architecture-dependent, not stimulus-dependent.
4. **Geometry-function dissociation.** Early layers are causally implicated (specificity 44–70×); late layers where geometry peaks are causally inert. Replicates the Weber finding for magnitude.

---

## Paper 3: Scalar Variability in Transformer Magnitude Representations

> **Same Geometry, Opposite Noise: Transformer Magnitude Representations Lack Scalar Variability**

**Pre-registration:** [OSF](https://osf.io/w4892/overview)

**Status:** arXiv preprint

**Code:** `scripts/scalar_variability_v2.py`, `scripts/scalar_variability_exploratory.py`

### Overview

Pre-registered companion analysis to Paper 1 (Weber). Tests whether transformer magnitude representations exhibit scalar variability — the biological signature that representational noise scales proportionally with magnitude (constant CV; Gibbon, 1977). Uses existing Paradigm A hidden states (no new model inference).

| Hypothesis | Prediction | Result |
|---|---|---|
| **H1** (α > 0) | Variability increases with magnitude | **Not supported** (α ≈ −0.04, all layers, all models) |
| **H2** (α ≈ 1) | Constant CV (scalar property) | **Not supported** |
| **H3** (α < 1) | Sub-scalar (no metabolic constraint) | **Supported** (3/3 models, 48/48 cells) |
| **H4** (layerwise) | Non-flat α profile | **Not supported** (stable across depth) |

### Key Findings

1. **Anti-scalar variability.** Representational variability *decreases* with magnitude (α ≈ −0.04 raw, −0.19 on the magnitude axis). The opposite of biological scalar variability.
2. **Corpus frequency mechanism.** Per-magnitude variability correlates with corpus frequency (ρ = .84): frequent numbers appear in more diverse contexts, producing wider representational dispersion.
3. **Magnitude-specific.** The anti-scalar pattern is 3–5× stronger along the magnitude axis (PC1) than orthogonal dimensions.
4. **Instruction tuning amplifies.** Llama-Instruct is more anti-scalar than Llama-Base (p < .001).

---

## Repository Structure

```
weber/
├── README.md
├── config.json                     # Model paths, layer ranges, HF commit hashes
├── model_checksums.json            # SHA256 checksums for reproducibility
│
├── scripts/                        # Weber paper code
│   ├── config.py                   # Shared configuration
│   ├── stimuli_generation.py       # Generate all stimuli (seed 42)
│   ├── paradigm_a_extract.py       # Extract hidden states
│   ├── paradigm_a_analyse.py       # RSA, AIC, Stevens exponent
│   ├── paradigm_b_behaviour.py     # Behavioural discrimination (B1)
│   ├── paradigm_b_additional.py    # B2/B3 tasks
│   ├── paradigm_c_supplement.py    # Precision gradient analysis
│   ├── paradigm_c_robustness.py    # Normalised precision
│   ├── paradigm_d_causal.py        # Activation patching (H7)
│   ├── exploratory_e5_b1_patching.py  # E5: patching with B1-format prompts
│   ├── corpus_distribution.py      # OpenWebText magnitude frequency analysis
│   ├── shuffled_magnitude_check.py # Shuffled-magnitude control
│   ├── unit_boundary_check.py      # Unit-boundary control
│   ├── psychometric_corrected.py   # Position-corrected Weber fractions
│   ├── compute_e3_dprime.py        # E3: SDT bridge (d-prime)
│   ├── evaluate_hypotheses.py      # Formal hypothesis evaluation
│   ├── run_exploratory_models.py   # Qwen replication
│   ├── generate_all_figures.py     # Figure generation
│   ├── generate_paper_figures.py   # Combined manuscript figures
│   ├── phase0_verify.py            # Phase 0 infrastructure verification
│   ├── phase1_compliance.py        # Pre-registration compliance audit
│   ├── scalar_variability_v2.py    # Paper 3: scalar variability analysis (H1-H4)
│   ├── scalar_variability_exploratory.py  # Paper 3: exploratory analyses (E4-E6)
│   └── check_*.py / debug_*.py     # Diagnostic scripts
│
├── stimuli/                        # Weber stimuli (deterministic, seed 42)
├── results/                        # Weber + scalar variability results
│
├── m3_pilot/                       # Categorical perception paper code
│   ├── m3_stimuli.py               # Stimulus generation (decade_10, control_15)
│   ├── m3_stimuli_100.py           # 100-boundary stimuli
│   ├── m3_stimuli_temp.py          # Temperature domain stimuli
│   ├── m3_stimuli_nonce.py         # Nonce-token remapping stimuli (E10)
│   ├── m3_extract.py               # Hidden-state extraction (CLI-enabled)
│   ├── m3_pilot_analysis.py        # RSA + precision + identification analysis
│   ├── m3_batch_run.py             # Batch extraction runner (all models)
│   ├── m3_run_analysis.py          # Multi-model RSA analysis runner
│   ├── m3_run_identification.py    # Multi-model identification runner
│   ├── m3_rerun_identification.py  # Counterbalanced identification
│   ├── m3_discrimination_stimuli.py # Paradigm B stimulus generation
│   ├── m3_discrimination.py        # Single-model discrimination runner
│   ├── m3_run_discrimination.py    # Batch discrimination runner
│   ├── m3_discrimination_analysis.py # Discrimination analysis pipeline
│   ├── m3_run_nonce_analysis.py    # E10 RSA analysis runner
│   ├── m3_causal.py                # E5 causal intervention
│   ├── m3_supplementary.py         # Pre-registration compliance analyses
│   │
│   ├── stimuli/                    # M3 stimulus files
│   │   ├── m3_stimuli_decade_10.json
│   │   ├── m3_stimuli_control_15.json
│   │   ├── m3_stimuli_decade_100.json
│   │   ├── m3_stimuli_control_150.json
│   │   ├── m3_stimuli_temp_hotcold.json
│   │   ├── m3_stimuli_temp_control.json
│   │   ├── m3_stimuli_nonce_no_order.json
│   │   ├── m3_stimuli_nonce_ordered.json
│   │   └── m3_discrimination_stimuli.json
│   │
│   ├── extractions/                # Hidden states (gitignored — regenerate via m3_extract.py)
│   ├── results/                    # Per-model RSA results + cross-model summaries
│   ├── discrimination_results/     # Paradigm B results
│   ├── causal_results/             # E5 activation patching results
│   └── logs/                       # Extraction logs
│
└── models/                         # NOT in git — download from HuggingFace
```

## Requirements

- Python 3.12
- PyTorch 2.8+ with ROCm 6.4 (AMD) or CUDA
- HuggingFace Transformers 5.0.0
- ~16 GB VRAM

Key packages: `torch`, `transformers`, `numpy`, `scipy`, `matplotlib`, `statsmodels`, `scikit-learn`

Models download automatically from HuggingFace Hub on first run. Commit hashes are recorded in `config.json`.

## Reproducing the Results

### Weber paper

The complete experiment runs in under 30 minutes per model on a single GPU.

```bash
cd scripts
python stimuli_generation.py
python paradigm_a_extract.py
python paradigm_a_analyse.py
python paradigm_b_behaviour.py
python paradigm_b_additional.py
python paradigm_c_supplement.py
python paradigm_d_causal.py
python shuffled_magnitude_check.py
python unit_boundary_check.py
python corpus_distribution.py
python generate_paper_figures.py
python evaluate_hypotheses.py
```

### Scalar variability paper (Paper 3)

Requires existing Paradigm A hidden states in `results/paradigm_a/` (generated above). No GPU needed — pure numpy on saved tensors.

```bash
cd scripts
python scalar_variability_v2.py          # Confirmatory analysis (H1-H4)
python scalar_variability_exploratory.py  # Exploratory analyses (E4-E6)
```

### M3 (categorical perception) paper

```bash
cd m3_pilot

# 1. Generate stimuli
python m3_stimuli.py                    # decade_10, control_15
python m3_stimuli_100.py                # decade_100, control_150
python m3_stimuli_temp.py               # temperature domain
python m3_stimuli_nonce.py              # nonce-token remapping (E10)
python m3_discrimination_stimuli.py     # Paradigm B pairs

# 2. Extract hidden states (all models × conditions)
python m3_extract.py --model llama3-8b-instruct --condition decade_10
# ... repeat for all model × condition combinations, or use:
python m3_batch_run.py

# 3. Run analyses
python m3_run_analysis.py               # RSA (all models, 10K Mantel permutations)
python m3_run_identification.py         # Counterbalanced identification
python m3_run_discrimination.py         # Paradigm B discrimination
python m3_run_nonce_analysis.py         # E10 nonce-token RSA
python m3_causal.py                     # E5 activation patching
python m3_supplementary.py              # Pre-registration compliance analyses
```

## Hardware

All experiments ran on:

- AMD Radeon RX 7900 GRE (16 GB VRAM)
- Windows 11, `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- PyTorch 2.8.0a0 with ROCm 6.4

CUDA users can ignore the ROCm environment variable. Any GPU with ≥16 GB VRAM supporting HuggingFace `output_hidden_states=True` should work.

### M3-specific notes

- Phi-3.5-mini loaded via `Lexius/Phi-3.5-mini-instruct` (community fork, same weights) with local DynamicCache compatibility patch for Transformers 5.0.0.
- Gemma-2-9B loaded in BF16 (all others FP16).
- All models loaded with `.to('cuda')` directly — `device_map='auto'` causes ROCm RoPE kernel failures.

## Citation

```bibtex
@article{cacioli2026weber,
  title={Weber's Law in Transformer Magnitude Representations: Efficient Coding, 
         Representational Geometry, and Psychophysical Laws in Language Models},
  author={Cacioli, JP},
  year={2026},
  note={Pre-registered: \url{https://osf.io/u4wp5}}
}

@article{cacioli2026cp,
  title={Categorical Perception in Large Language Model Hidden States: 
         Structural Warping at Digit-Count Boundaries},
  author={Cacioli, JP},
  year={2026},
  note={Pre-registered: \url{https://osf.io/qrxf3}}
}

@article{cacioli2026scalar,
  title={Same Geometry, Opposite Noise: Transformer Magnitude Representations 
         Lack Scalar Variability},
  author={Cacioli, JP},
  year={2026},
  note={Pre-registered: \url{https://osf.io/w4892}}
}
```

## Licence

MIT
