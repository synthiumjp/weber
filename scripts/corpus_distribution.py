#!/usr/bin/env python3
"""
Appendix E: Corpus Magnitude Distribution Analysis
Weber's Law in Transformer Magnitude Representations (Project 4.2)

Pre-registration: "Extract integer frequencies 1–1000 from OpenWebText or
RedPajama. Fit power-law vs exponential. Pre-registered as mandatory preliminary."

Tests whether the distribution of integers in natural language text follows
a 1/s (power-law) prior, which is the theoretical basis for expecting
efficient coding to produce logarithmic representations.

Uses HuggingFace datasets to stream OpenWebText without full download.

Author: JP Cacioli
Date: March 2026
"""

import json
import re
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# Regex to find standalone integers (not part of decimals, dates, etc.)
# Matches integers 1-9999 that are word-bounded
INTEGER_PATTERN = re.compile(r'\b(\d{1,4})\b')

# More restrictive: avoid dates, IPs, version numbers
# Matches integers preceded and followed by space/punctuation, not digits/dots
STRICT_INTEGER_PATTERN = re.compile(r'(?<![.\d])(\b\d{1,4}\b)(?![.\d])')


def count_integers_in_text(text, max_val=1000):
    """Extract and count integers 1–max_val from text."""
    counts = Counter()
    for match in STRICT_INTEGER_PATTERN.finditer(text):
        val = int(match.group(1))
        if 1 <= val <= max_val:
            counts[val] += 1
    return counts


def stream_openwebtext(n_docs=None, max_docs=500000):
    """Stream OpenWebText from HuggingFace datasets.
    
    Uses streaming mode to avoid downloading the full ~12GB dataset.
    """
    from datasets import load_dataset
    
    log.info("Loading OpenWebText (streaming mode)...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    target = n_docs if n_docs else max_docs
    log.info(f"Will process up to {target:,} documents")
    
    return ds, target


def run_corpus_analysis(project_root, n_docs=100000, corpus="openwebtext"):
    """Run the corpus magnitude distribution analysis."""
    
    output_dir = Path(project_root) / "results" / "appendix_e"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"\n{'='*70}")
    log.info(f"APPENDIX E: Corpus Magnitude Distribution")
    log.info(f"{'='*70}")
    log.info(f"Corpus: {corpus}")
    log.info(f"Target documents: {n_docs:,}")
    
    # Stream and count
    ds, target = stream_openwebtext(n_docs=n_docs)
    
    total_counts = Counter()
    n_processed = 0
    total_integers = 0
    t_start = time.time()
    
    for doc in ds:
        text = doc.get("text", "")
        counts = count_integers_in_text(text, max_val=1000)
        total_counts.update(counts)
        total_integers += sum(counts.values())
        n_processed += 1
        
        if n_processed % 10000 == 0:
            elapsed = time.time() - t_start
            rate = n_processed / elapsed
            remaining = (target - n_processed) / rate if rate > 0 else 0
            log.info(f"  Processed {n_processed:,}/{target:,} docs "
                     f"({total_integers:,} integers found, "
                     f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
        
        if n_processed >= target:
            break
    
    elapsed_total = time.time() - t_start
    log.info(f"\nProcessed {n_processed:,} documents in {elapsed_total:.1f}s")
    log.info(f"Total integers found (1-1000): {total_integers:,}")
    
    # Build frequency array for integers 1-1000
    magnitudes = np.arange(1, 1001)
    frequencies = np.array([total_counts.get(n, 0) for n in magnitudes], dtype=np.float64)
    
    # Report basic stats
    n_nonzero = np.sum(frequencies > 0)
    log.info(f"Integers with nonzero count: {n_nonzero}/1000")
    log.info(f"Top 10 most frequent: {[int(n) for n, _ in total_counts.most_common(10)]}")
    
    # Fit models
    log.info(f"\n--- Model Fitting ---")
    
    # Only fit on magnitudes with nonzero counts
    mask = frequencies > 0
    x_fit = magnitudes[mask].astype(np.float64)
    y_fit = frequencies[mask]
    
    # Normalise to proportions for fitting
    y_norm = y_fit / y_fit.sum()
    
    # Model 1: Power law — f(n) = a * n^(-α)
    # The 1/s prior corresponds to α = 1
    def power_law(n, a, alpha):
        return a * np.power(n, -alpha)
    
    try:
        popt_pl, pcov_pl = curve_fit(power_law, x_fit, y_norm, p0=[1.0, 1.0],
                                      bounds=([0, 0], [np.inf, 5.0]), maxfev=10000)
        y_pred_pl = power_law(x_fit, *popt_pl)
        ss_res_pl = np.sum((y_norm - y_pred_pl)**2)
        ss_tot = np.sum((y_norm - y_norm.mean())**2)
        r2_pl = 1 - ss_res_pl / ss_tot
        aic_pl = len(x_fit) * np.log(ss_res_pl / len(x_fit)) + 2 * 2
        
        log.info(f"  Power law: f(n) = {popt_pl[0]:.6f} × n^(-{popt_pl[1]:.3f})")
        log.info(f"    α = {popt_pl[1]:.3f} (1/s prior predicts α = 1.0)")
        log.info(f"    R² = {r2_pl:.4f}, AIC = {aic_pl:.1f}")
        
        pl_fit = {
            "a": float(popt_pl[0]),
            "alpha": float(popt_pl[1]),
            "r2": float(r2_pl),
            "aic": float(aic_pl),
        }
    except Exception as e:
        log.warning(f"  Power law fit failed: {e}")
        pl_fit = {"error": str(e)}
        r2_pl = -1
        aic_pl = np.inf
    
    # Model 2: Exponential — f(n) = a * exp(-λn)
    def exponential(n, a, lam):
        return a * np.exp(-lam * n)
    
    try:
        popt_exp, pcov_exp = curve_fit(exponential, x_fit, y_norm, p0=[0.01, 0.01],
                                        bounds=([0, 0], [np.inf, 1.0]), maxfev=10000)
        y_pred_exp = exponential(x_fit, *popt_exp)
        ss_res_exp = np.sum((y_norm - y_pred_exp)**2)
        r2_exp = 1 - ss_res_exp / ss_tot
        aic_exp = len(x_fit) * np.log(ss_res_exp / len(x_fit)) + 2 * 2
        
        log.info(f"  Exponential: f(n) = {popt_exp[0]:.6f} × exp(-{popt_exp[1]:.5f}n)")
        log.info(f"    R² = {r2_exp:.4f}, AIC = {aic_exp:.1f}")
        
        exp_fit = {
            "a": float(popt_exp[0]),
            "lambda": float(popt_exp[1]),
            "r2": float(r2_exp),
            "aic": float(aic_exp),
        }
    except Exception as e:
        log.warning(f"  Exponential fit failed: {e}")
        exp_fit = {"error": str(e)}
        r2_exp = -1
        aic_exp = np.inf
    
    # Model 3: Log-normal — fit in log-log space
    # log(f) = a - α*log(n) is a straight line in log-log if power law
    log_x = np.log(x_fit)
    log_y = np.log(y_fit)  # raw counts (not normalised) for log-log
    
    from numpy.polynomial import polynomial as P
    # Linear fit in log-log space
    coeffs = np.polyfit(log_x, log_y, 1)
    alpha_loglog = -coeffs[0]  # slope is -α
    
    log_y_pred = np.polyval(coeffs, log_x)
    ss_res_loglog = np.sum((log_y - log_y_pred)**2)
    ss_tot_loglog = np.sum((log_y - log_y.mean())**2)
    r2_loglog = 1 - ss_res_loglog / ss_tot_loglog
    
    log.info(f"  Log-log linear fit: slope = -{alpha_loglog:.3f}")
    log.info(f"    α (log-log) = {alpha_loglog:.3f}, R² (log-log) = {r2_loglog:.4f}")
    
    # Summary
    log.info(f"\n--- Summary ---")
    best_model = "power_law" if aic_pl < aic_exp else "exponential"
    log.info(f"  Best model (AIC): {best_model}")
    log.info(f"  Power law α = {pl_fit.get('alpha', 'N/A')}")
    log.info(f"  1/s prior prediction: α = 1.0")
    
    if 'alpha' in pl_fit:
        alpha = pl_fit['alpha']
        if 0.8 <= alpha <= 1.2:
            log.info(f"  α is CONSISTENT with 1/s prior (within ±0.2)")
        elif alpha < 0.8:
            log.info(f"  α < 1: distribution is LESS skewed than 1/s")
        else:
            log.info(f"  α > 1: distribution is MORE skewed than 1/s (e.g., Zipfian)")
    
    # Benford's Law check — is digit 1 more common as leading digit?
    leading_digit_counts = Counter()
    for n in range(1, 1001):
        if total_counts.get(n, 0) > 0:
            leading = str(n)[0]
            leading_digit_counts[leading] += total_counts[n]
    
    total_leading = sum(leading_digit_counts.values())
    if total_leading > 0:
        log.info(f"\n  Leading digit distribution (Benford's Law check):")
        for d in '123456789':
            observed = leading_digit_counts.get(d, 0) / total_leading
            benford = np.log10(1 + 1/int(d))
            log.info(f"    Digit {d}: observed={observed:.3f}, Benford={benford:.3f}")
    
    # Save results
    output = {
        "corpus": corpus,
        "n_documents": n_processed,
        "total_integers_found": total_integers,
        "n_nonzero_magnitudes": int(n_nonzero),
        "elapsed_seconds": elapsed_total,
        "frequencies": {int(n): int(c) for n, c in zip(magnitudes, frequencies)},
        "top_20": [(int(n), int(c)) for n, c in total_counts.most_common(20)],
        "power_law_fit": pl_fit,
        "exponential_fit": exp_fit,
        "loglog_fit": {
            "alpha": float(alpha_loglog),
            "r2": float(r2_loglog),
        },
        "best_model_aic": best_model,
        "leading_digit_distribution": {d: int(leading_digit_counts.get(d, 0)) 
                                        for d in '123456789'},
    }
    
    out_path = output_dir / "corpus_magnitude_distribution.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    log.info(f"\nSaved: {out_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Appendix E: Corpus magnitude distribution")
    parser.add_argument("--n-docs", type=int, default=100000,
                        help="Number of documents to process (default: 100,000)")
    parser.add_argument("--project-root", type=str, default=r"C:\weber")
    args = parser.parse_args()
    
    run_corpus_analysis(args.project_root, n_docs=args.n_docs)


if __name__ == "__main__":
    main()
