#!/usr/bin/env python3
"""
M3 Batch Runner — Full Experiment Paradigm A
=============================================
Runs m3_extract.py on all pre-registered models × both conditions,
then m3_rerun_identification.py for counterbalanced identification.

Models (pre-registered):
  1. Mistral-7B-Instruct-v0.3    (FP16, ~13.5GB)
  2. Gemma-2-9B-IT               (BF16, ~18GB — requires 8-bit fallback)
  3. Qwen2.5-7B-Instruct         (FP16, ~14GB)
  4. Phi-3.5-mini-instruct       (FP16, ~7.1GB)
  5. Llama-3-8B (base)           (FP16, ~16GB — exploratory)

Conditions: decade_10, control_15
Working directory: C:\\weber\\m3_pilot\\

Usage:
  # Run all models sequentially:
  python m3_batch_run.py

  # Run a single model (by index or short name):
  python m3_batch_run.py --model mistral
  python m3_batch_run.py --model 0

  # Skip identification (extraction only):
  python m3_batch_run.py --extract-only

  # Resume from a specific model (skip already-completed):
  python m3_batch_run.py --resume-from qwen

  # Dry run (print what would execute without running):
  python m3_batch_run.py --dry-run

Author: JP Cacioli
Programme: Classical Minds, Modern Machines
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "mistral",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "short": "mistral-7b-instruct",
        "precision": "float16",
        "layers": 32,
        "d_model": 4096,
        "est_vram_gb": 13.5,
        "role": "primary",
        "notes": "Sliding-window attention. Different tokeniser from Llama family.",
    },
    {
        "name": "gemma",
        "hf_id": "google/gemma-2-9b-it",
        "short": "gemma2-9b-it",
        "precision": "bfloat16",  # Gemma2 native precision
        "layers": 42,
        "d_model": 3584,
        "est_vram_gb": 18.0,  # FP16 ~18GB — needs CPU offload
        "role": "primary",
        "notes": "42 layers, 3584 d_model. BF16 ~18GB, uses device_map='auto' for GPU+CPU split.",
        "cpu_offload": True,
    },
    {
        "name": "qwen",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "short": "qwen25-7b-instruct",
        "precision": "float16",
        "layers": 28,
        "d_model": 3584,
        "est_vram_gb": 14.0,
        "role": "primary",
        "notes": "28 layers, 3584 d_model. Different tokeniser.",
    },
    {
        "name": "phi",
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "short": "phi35-mini-instruct",
        "precision": "float16",
        "layers": 32,
        "d_model": 3072,
        "est_vram_gb": 7.1,
        "role": "primary",
        "notes": "3.82B params — scale probe. Curriculum-learned on synthetic data.",
    },
    {
        "name": "llama-base",
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "short": "llama3-8b-base",
        "precision": "float16",
        "layers": 32,
        "d_model": 4096,
        "est_vram_gb": 16.1,
        "role": "exploratory",
        "notes": "Instruction-tuning control. No chat template — B0 uses raw logits.",
    },
]

CONDITIONS = ["decade_10", "control_15"]

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────

WORK_DIR = Path(r"C:\weber\m3_pilot")
EXTRACT_SCRIPT = WORK_DIR / "m3_extract.py"
STIMULI_DIR = WORK_DIR / "stimuli"
EXTRACT_DIR = WORK_DIR / "extractions"
LOG_DIR = WORK_DIR / "logs"


def check_prerequisites():
    """Verify all required files exist before starting."""
    missing = []
    if not EXTRACT_SCRIPT.exists():
        missing.append(str(EXTRACT_SCRIPT))
    for cond in CONDITIONS:
        stim_file = STIMULI_DIR / f"m3_stimuli_{cond}.json"
        if not stim_file.exists():
            missing.append(str(stim_file))
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)


def check_existing_outputs(model_short, condition):
    """Check if extraction outputs already exist for this model × condition."""
    centroid_file = EXTRACT_DIR / f"m3_centroids_{condition}_{model_short}.npz"
    meta_file = EXTRACT_DIR / f"m3_meta_{condition}_{model_short}.json"
    return centroid_file.exists() and meta_file.exists()


def format_duration(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def run_extraction(model, condition, dry_run=False, skip_identification=False):
    """
    Run m3_extract.py for a given model × condition.

    m3_extract.py includes both hidden-state extraction (Paradigm A)
    and raw-logit identification (Paradigm B0). Counterbalanced
    identification (m3_rerun_identification.py) runs as a separate
    step after all extractions complete.
    """

    cmd = [
        sys.executable, str(EXTRACT_SCRIPT),
        "--model", model["hf_id"],
        "--condition", condition,
        "--precision", model["precision"],
        "--output-tag", model["short"],
    ]
    if model.get("cpu_offload"):
        cmd.append("--cpu-offload")
    if skip_identification:
        cmd.append("--skip-identification")

    print(f"\n{'='*60}")
    print(f"EXTRACTION: {model['short']} × {condition}")
    print(f"  HF ID:      {model['hf_id']}")
    print(f"  Precision:  {model['precision']}")
    print(f"  Est. VRAM:  {model['est_vram_gb']:.1f} GB")
    print(f"  Layers:     {model['layers']}")
    print(f"  d_model:    {model['d_model']}")
    if model.get("cpu_offload"):
        print(f"  Offload:    GPU+CPU (device_map='auto')")
    print(f"  Role:       {model['role']}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return True, 0.0

    # Check for existing outputs
    if check_existing_outputs(model["short"], condition):
        print(f"  [SKIP] Outputs already exist for {model['short']} × {condition}")
        return True, 0.0

    t0 = time.time()
    timeout = 1800 if model.get("cpu_offload") else 600  # 30 min for offload, 10 min otherwise
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORK_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0

        # Log stdout/stderr
        LOG_DIR.mkdir(exist_ok=True)
        log_file = LOG_DIR / f"extract_{model['short']}_{condition}.log"
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Duration: {format_duration(elapsed)}\n")
            f.write(f"\n{'='*40} STDOUT {'='*40}\n")
            f.write(result.stdout)
            f.write(f"\n{'='*40} STDERR {'='*40}\n")
            f.write(result.stderr)

        if result.returncode != 0:
            print(f"  [FAIL] Exit code {result.returncode} ({format_duration(elapsed)})")
            print(f"  Last stderr: {result.stderr[-500:] if result.stderr else '(empty)'}")
            print(f"  Full log: {log_file}")
            return False, elapsed
        else:
            print(f"  [OK] Completed in {format_duration(elapsed)}")
            return True, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] Exceeded {timeout}s limit")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERROR] {e}")
        return False, elapsed


def resolve_model(model_arg):
    """Resolve a model argument (name or index) to a model dict."""
    # Try as integer index
    try:
        idx = int(model_arg)
        if 0 <= idx < len(MODELS):
            return [MODELS[idx]]
        else:
            print(f"ERROR: Model index {idx} out of range (0–{len(MODELS)-1})")
            sys.exit(1)
    except ValueError:
        pass

    # Try as name
    matches = [m for m in MODELS if m["name"] == model_arg.lower()]
    if matches:
        return matches
    # Try as short name
    matches = [m for m in MODELS if m["short"] == model_arg.lower()]
    if matches:
        return matches

    print(f"ERROR: Unknown model '{model_arg}'")
    print(f"Available: {', '.join(m['name'] for m in MODELS)}")
    sys.exit(1)


def print_summary(results):
    """Print a summary table of all runs."""
    print(f"\n\n{'='*70}")
    print("BATCH RUN SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Condition':<12} {'Status':<10} {'Time':<10}")
    print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*10}")

    total_time = 0
    failures = []
    for r in results:
        status = "OK" if r["extract_ok"] else "FAIL"
        time_str = format_duration(r["total_time"])
        total_time += r["total_time"]
        print(f"{r['model']:<25} {r['condition']:<12} {status:<10} {time_str:<10}")
        if not r["extract_ok"]:
            failures.append(f"{r['model']} × {r['condition']}")

    print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*10}")
    print(f"{'TOTAL':<25} {'':<12} {'':<10} {format_duration(total_time):<10}")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        print("\nCheck logs/ directory for details.")
    else:
        print("\nAll runs completed successfully.")

    # Save summary JSON
    summary_file = WORK_DIR / "logs" / "batch_summary.json"
    LOG_DIR.mkdir(exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_s": total_time,
        "n_runs": len(results),
        "n_failures": len(failures),
        "results": results,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="M3 Batch Runner — Paradigm A full experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  0 / mistral     Mistral-7B-Instruct-v0.3   (primary)
  1 / gemma       Gemma-2-9B-IT              (primary, 8-bit fallback)
  2 / qwen        Qwen2.5-7B-Instruct        (primary)
  3 / phi         Phi-3.5-mini-instruct       (primary, scale probe)
  4 / llama-base  Llama-3-8B (base)           (exploratory)
        """,
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run a single model (name or index). Default: run all.",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Run extraction only, skip identification.",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from a specific model (skip earlier models).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would execute without running.",
    )
    parser.add_argument(
        "--condition", type=str, default=None, choices=CONDITIONS,
        help="Run a single condition. Default: both.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("M3 BATCH RUNNER — Paradigm A Full Experiment")
    print("Classical Minds, Modern Machines")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Verify prerequisites
    check_prerequisites()

    # Determine which models to run
    if args.model:
        models_to_run = resolve_model(args.model)
    else:
        models_to_run = MODELS.copy()

    # Handle --resume-from
    if args.resume_from:
        resume_models = resolve_model(args.resume_from)
        resume_name = resume_models[0]["name"]
        start_idx = next(
            (i for i, m in enumerate(models_to_run) if m["name"] == resume_name),
            None,
        )
        if start_idx is None:
            print(f"ERROR: --resume-from model '{args.resume_from}' not in run list")
            sys.exit(1)
        skipped = [m["name"] for m in models_to_run[:start_idx]]
        models_to_run = models_to_run[start_idx:]
        if skipped:
            print(f"Resuming from {resume_name}, skipping: {', '.join(skipped)}")

    # Determine conditions
    conditions = [args.condition] if args.condition else CONDITIONS

    # Print run plan
    print(f"\nModels ({len(models_to_run)}):")
    for m in models_to_run:
        flag = " [CPU offload]" if m.get("cpu_offload") else ""
        print(f"  {m['name']:<15} {m['hf_id']:<45} {m['precision']}{flag}")
    print(f"\nConditions: {', '.join(conditions)}")
    print(f"Identification: {'SKIP' if args.extract_only else 'included in extraction'}")
    total_runs = len(models_to_run) * len(conditions)
    print(f"Total runs: {total_runs} (extraction + identification per run)")

    # Confirm (unless dry run)
    if not args.dry_run:
        print(f"\nEstimated time: ~{total_runs * 1}–{total_runs * 3} minutes")
        print("(Based on pilot: ~40s extraction + ~60s identification per condition)")

    # ── IMPORTANT: Set ROCm env var ──
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    # Run
    results = []
    for mi, model in enumerate(models_to_run):
        print(f"\n\n{'#'*70}")
        print(f"MODEL {mi+1}/{len(models_to_run)}: {model['short']}")
        print(f"{'#'*70}")

        for condition in conditions:
            run_result = {
                "model": model["short"],
                "hf_id": model["hf_id"],
                "condition": condition,
                "role": model["role"],
                "extract_ok": None,
                "total_time": 0,
            }

            # Extraction (includes built-in B0 identification unless --extract-only)
            if args.extract_only:
                # Pass --skip-identification to m3_extract.py
                ext_ok, ext_time = run_extraction(
                    model, condition, dry_run=args.dry_run,
                    skip_identification=True,
                )
            else:
                ext_ok, ext_time = run_extraction(model, condition, dry_run=args.dry_run)
            
            run_result["extract_ok"] = ext_ok
            run_result["total_time"] += ext_time

            results.append(run_result)

        # GPU memory cleanup between models
        if not args.dry_run and mi < len(models_to_run) - 1:
            print(f"\n  Clearing GPU memory before next model...")
            # The subprocess approach means each extraction loads/unloads
            # its own model, so explicit cleanup isn't strictly needed.
            # But if VRAM leaks occur, uncomment:
            # subprocess.run([sys.executable, "-c",
            #     "import torch; torch.cuda.empty_cache()"])
            time.sleep(2)

    # Summary
    if not args.dry_run:
        print_summary(results)
    else:
        print(f"\n[DRY RUN] Would have executed {total_runs} runs")


if __name__ == "__main__":
    main()
