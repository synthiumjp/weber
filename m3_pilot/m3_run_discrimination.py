"""
M3 Paradigm B: Batch Discrimination Runner
===========================================
Runs m3_discrimination.py across all pre-registered models sequentially.

Uses subprocess execution with per-model GPU cleanup (same pattern as m3_batch_run.py).

Usage:
  python m3_run_discrimination.py                   # run all models
  python m3_run_discrimination.py --models llama3 mistral  # run specific models
  python m3_run_discrimination.py --dry-run          # show commands without running

Author: JP Cacioli
Research Assistant: Claude (Anthropic)
Date: 28 March 2026
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ==============================================================================
# Model registry (same as m3_batch_run.py)
# ==============================================================================

MODELS = {
    'llama3-8b-instruct': {
        'hf_id': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'precision': 'float16',
        'chat_template': True,
        'vram_est_gb': 16.1,
    },
    'mistral-7b-instruct': {
        'hf_id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'precision': 'float16',
        'chat_template': True,
        'vram_est_gb': 14.5,
    },
    'gemma2-9b-it': {
        'hf_id': 'google/gemma-2-9b-it',
        'precision': 'bfloat16',
        'chat_template': True,
        'vram_est_gb': 18.5,
    },
    'qwen25-7b-instruct': {
        'hf_id': 'Qwen/Qwen2.5-7B-Instruct',
        'precision': 'float16',
        'chat_template': True,
        'vram_est_gb': 15.3,
    },
    'phi35-mini-instruct': {
        'hf_id': 'Lexius/Phi-3.5-mini-instruct',
        'precision': 'float16',
        'chat_template': True,
        'vram_est_gb': 7.7,
    },
    'llama3-8b-base': {
        'hf_id': 'meta-llama/Meta-Llama-3-8B',
        'precision': 'float16',
        'chat_template': False,
        'vram_est_gb': 16.1,
    },
}

DISC_SCRIPT = Path('m3_discrimination.py')
OUTPUT_DIR = Path('discrimination_results')
LOG_DIR = Path('logs')


def build_command(model_key: str, model_cfg: dict) -> list[str]:
    """Build the subprocess command for a model."""
    cmd = [
        sys.executable, str(DISC_SCRIPT),
        '--model', model_cfg['hf_id'],
        '--output-tag', model_key.replace('-', '_'),
        '--precision', model_cfg['precision'],
    ]
    if not model_cfg['chat_template']:
        cmd.append('--skip-chat-template')
    return cmd


def run_model(model_key: str, model_cfg: dict, dry_run: bool = False) -> dict:
    """Run discrimination for a single model."""
    cmd = build_command(model_key, model_cfg)
    
    result_path = OUTPUT_DIR / f'm3_discrimination_{model_key.replace("-", "_")}.json'
    
    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"HF ID: {model_cfg['hf_id']}")
    print(f"Precision: {model_cfg['precision']}")
    print(f"VRAM est: {model_cfg['vram_est_gb']} GB")
    print(f"Command: {' '.join(cmd)}")
    
    if result_path.exists():
        print(f"  → Output already exists: {result_path}")
        print(f"  → SKIPPING (delete file to re-run)")
        return {'status': 'skipped', 'model': model_key}
    
    if dry_run:
        print(f"  → DRY RUN — would execute above command")
        return {'status': 'dry_run', 'model': model_key}
    
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f'discrimination_{model_key}.log'
    
    t0 = time.time()
    
    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    
    try:
        with open(log_path, 'w') as logf:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=1800,  # 30 min timeout
            )
        
        elapsed = time.time() - t0
        
        if proc.returncode == 0:
            print(f"  ✓ Completed in {elapsed:.0f}s")
            
            # Print summary from results
            if result_path.exists():
                with open(result_path) as f:
                    data = json.load(f)
                s = data.get('summary', {})
                print(f"  Overall accuracy: {s.get('overall_accuracy', 'N/A')}")
                print(f"  Cross-boundary:   {s.get('cross_boundary_accuracy', 'N/A')}")
                print(f"  Within-category:  {s.get('within_category_accuracy', 'N/A')}")
            
            return {'status': 'success', 'model': model_key, 'time': elapsed}
        else:
            print(f"  ✗ Failed (exit code {proc.returncode}) in {elapsed:.0f}s")
            print(f"  Check log: {log_path}")
            return {'status': 'failed', 'model': model_key, 'exit_code': proc.returncode}
    
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timed out after 1800s")
        return {'status': 'timeout', 'model': model_key}
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {'status': 'error', 'model': model_key, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='M3 Paradigm B: Batch Discrimination Runner')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model keys to run (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without running')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip models with existing results (default: True)')
    args = parser.parse_args()
    
    # Verify script exists
    if not DISC_SCRIPT.exists():
        print(f"ERROR: {DISC_SCRIPT} not found")
        sys.exit(1)
    
    # Select models
    if args.models:
        model_keys = []
        for key in args.models:
            # Allow partial matching
            matches = [k for k in MODELS if key in k]
            if matches:
                model_keys.extend(matches)
            else:
                print(f"WARNING: No model matching '{key}'")
        model_keys = list(dict.fromkeys(model_keys))  # deduplicate preserving order
    else:
        model_keys = list(MODELS.keys())
    
    print(f"M3 Paradigm B: Batch Discrimination Runner")
    print(f"Models to run: {model_keys}")
    print(f"Dry run: {args.dry_run}")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run all models
    results = []
    t0 = time.time()
    
    for model_key in model_keys:
        model_cfg = MODELS[model_key]
        result = run_model(model_key, model_cfg, dry_run=args.dry_run)
        results.append(result)
    
    total_time = time.time() - t0
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = r['status']
        model = r['model']
        if status == 'success':
            print(f"  ✓ {model}: {r['time']:.0f}s")
        elif status == 'skipped':
            print(f"  ⊘ {model}: skipped (exists)")
        else:
            print(f"  ✗ {model}: {status}")
    
    print(f"\nTotal time: {total_time:.0f}s")
    
    # Save batch summary
    LOG_DIR.mkdir(exist_ok=True)
    summary_path = LOG_DIR / 'discrimination_batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time': total_time,
            'results': results,
        }, f, indent=2)
    print(f"Batch summary: {summary_path}")


if __name__ == '__main__':
    main()
