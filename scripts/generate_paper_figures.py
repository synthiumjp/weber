#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — Combined Paper Figures
==================================================
6 manuscript figures combining pre-registered F1–F10 into publication composites.
Individual figures remain in generate_all_figures.py for supplement/OSF.

Usage: python scripts/generate_paper_figures.py [--results-dir C:\\weber\\results]

Figure 1: H1 — RSA heatmap (cosine, both models × 3 domains)
Figure 2: H2 — Psychometric curves (Llama + Mistral) + Weber fraction constancy
Figure 3: Geometry — Stevens β (Llama) + Precision gradient (numerical raw, both models)
Figure 4: Causal — E5 dose-response (Llama) + E5 bar chart (both models)
Figure 5: Controls — Digit boundary Cohen's d + Variance partitioning
Figure 6: Corpus — Log-log distribution + Benford's

Author: JP Cacioli | March 2026
"""

import json, sys, argparse, warnings, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore', category=RuntimeWarning)

# === HELPERS (shared with generate_all_figures.py) ===
def parse_layer_int(k):
    m = re.search(r'(\d+)', str(k)); return int(m.group(1)) if m else 0
def sorted_layer_keys(d): return sorted(d.keys(), key=parse_layer_int)
def find_layer_key(d, n):
    n = int(n) if not isinstance(n, int) else n
    for c in [str(n), f'layer_{n:02d}', f'layer_{n}']:
        if c in d: return c
    for k in d:
        if parse_layer_int(k) == n: return k
    return None
def load_json(path):
    p = Path(path)
    if not p.exists(): return None
    with open(p) as f: return json.load(f)
def safe_mkdir(p): Path(p).mkdir(parents=True, exist_ok=True)
def save_fig(fig, d, name):
    fig.savefig(Path(d)/f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(Path(d)/f"{name}.pdf", bbox_inches='tight')
    plt.close(fig); print(f"  Saved: {name}")
def get_pairwise(data, layer_idx, metric='cosine'):
    pw = data.get(f'pairwise_{metric}')
    if pw is not None and isinstance(pw, list) and layer_idx < len(pw): return pw[layer_idx]
    return None

# === CONFIG ===
MODELS = ['llama_instruct', 'mistral_instruct']
ML = {'llama_instruct': 'Llama-3-8B-Instruct', 'mistral_instruct': 'Mistral-7B-Instruct-v0.3'}
DOMAINS = ['numerical', 'temporal', 'spatial']
DL = {'numerical': 'Numerical', 'temporal': 'Temporal', 'spatial': 'Spatial'}
PRIMARY = list(range(16, 32))
MAGS = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,150,200,300,500,700,1000]
BL = [10, 30, 100, 300, 1000]
RATIOS = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
DC = {'numerical': '#2166AC', 'temporal': '#B2182B', 'spatial': '#4DAF4A'}
MC = {'llama_instruct': '#2166AC', 'mistral_instruct': '#D6604D'}

plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':13,'xtick.labelsize':10,
    'ytick.labelsize':10,'legend.fontsize':10,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'font.family':'sans-serif','axes.spines.top':False,'axes.spines.right':False})


# ======================================================================
# FIGURE 1: H1 — RSA Heatmap (Cosine, both models × 3 domains)
# ======================================================================
def figure_1(R, O):
    print("\n=== Figure 1: H1 RSA Heatmap ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    fig.text(0.5, 0.97, 'Figure 1. Representational geometry is unanimously logarithmic across models, domains, and layers.',
             ha='center', fontsize=11, style='italic')
    for ai, model in enumerate(MODELS):
        ax = axes[ai]; hm = np.full((len(PRIMARY), len(DOMAINS)), np.nan)
        for di, dom in enumerate(DOMAINS):
            d = load_json(Path(R)/'paradigm_a'/model/dom/'paradigm_a_analysis.json')
            if d is None: continue
            layers = d.get('layers', {})
            for li, layer in enumerate(PRIMARY):
                lk = find_layer_key(layers, layer)
                if lk is None: continue
                w = layers[lk].get('cosine',{}).get('rsa',{}).get('weber',{})
                rho = w.get('rho') if isinstance(w, dict) else w
                if rho is not None: hm[li, di] = rho
        im = ax.imshow(hm, aspect='auto', cmap='RdYlBu_r', vmin=0.5, vmax=1.0, interpolation='nearest')
        ax.set_title(ML[model], fontsize=12, fontweight='bold')
        ax.set_xlabel('Domain'); ax.set_xticks(range(3)); ax.set_xticklabels([DL[d] for d in DOMAINS])
        if ai==0: ax.set_ylabel('Layer'); ax.set_yticks(range(len(PRIMARY))); ax.set_yticklabels(PRIMARY)
        else: ax.set_yticks(range(len(PRIMARY)))
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                v=hm[i,j]
                if not np.isnan(v): ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,
                                           color='white' if v>0.8 else 'black')
    fig.subplots_adjust(right=0.88, top=0.93)
    cax=fig.add_axes([0.90,0.12,0.02,0.75])
    fig.colorbar(im, cax=cax, label='RSA Spearman \u03c1 (Weber theoretical RDM)')
    save_fig(fig, O, 'Figure1_H1_rsa_heatmap')


# ======================================================================
# FIGURE 2: H2 — Psychometric + Weber Fraction Constancy
# ======================================================================
def figure_2(R, O):
    print("\n=== Figure 2: H2 Behavioural ===")
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)
    fig.text(0.5, 0.97, 'Figure 2. Behavioural Weber\'s Law: Llama shows ratio-dependent discrimination in the human range; Mistral does not.',
             ha='center', fontsize=11, style='italic')
    cm = plt.cm.viridis; cols = cm(np.linspace(0.1, 0.9, len(BL)))

    # Panels A & B: Psychometric curves
    for pi, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[pi])
        raw = load_json(Path(R)/'paradigm_b'/model/'numerical'/'paradigm_b_raw.json')
        if raw is None: continue
        items = raw if isinstance(raw, list) else raw.get('items', [])
        acc = {}
        for it in items:
            bl=it.get('baseline',it.get('nominal_baseline')); rt=it.get('ratio',it.get('nominal_ratio'))
            c=it.get('correct',it.get('is_correct'))
            if bl is None or rt is None or c is None: continue
            acc.setdefault((min(BL,key=lambda b:abs(b-bl)),min(RATIOS,key=lambda r:abs(r-rt))),[]).append(1 if c else 0)
        for bi, bl in enumerate(BL):
            xs,ys=[],[]
            for rt in RATIOS:
                if (bl,rt) in acc: xs.append(rt); ys.append(np.mean(acc[(bl,rt)]))
            if xs: ax.plot(xs,ys,'o-',color=cols[bi],label=f'{bl}',lw=2,ms=5)
        ax.set_xlabel('Magnitude Ratio'); ax.set_xscale('log')
        ax.set_xticks(RATIOS); ax.set_xticklabels([f'{r:.2f}' for r in RATIOS], fontsize=8)
        ax.axhline(0.5,color='gray',ls='--',alpha=0.5)
        ax.set_ylim(0.4,1.05); ax.set_title(f'{"A" if pi==0 else "B"}. {ML[model].split("-")[0]}', fontweight='bold')
        if pi==0: ax.set_ylabel('Proportion Correct'); ax.legend(title='Baseline', fontsize=8, title_fontsize=9, loc='lower right')

    # Panel C: Weber fraction constancy
    ax = fig.add_subplot(gs[2])
    ax.axhspan(0.10, 0.25, alpha=0.15, color='green', label='Human range')
    for mi, model in enumerate(MODELS):
        ps = load_json(Path(R)/'paradigm_b'/model/'numerical'/'psychometric_corrected.json')
        if ps is None: continue
        bwf = ps.get('bootstrap_weber_fractions', {})
        xp, wv, cl, ch = [], [], [], []
        for bk in ['10.0','30.0','100.0','300.0','1000.0']:
            bd = bwf.get(bk, {}); wf = bd.get('median', bd.get('mean'))
            if wf is None or not np.isfinite(wf): continue
            xp.append(float(bk)); wv.append(wf); cl.append(bd.get('ci_low',wf)); ch.append(bd.get('ci_high',wf))
        if wv:
            ye = [[max(0,w-c) for w,c in zip(wv,cl)], [max(0,c-w) for w,c in zip(wv,ch)]]
            offset = -0.1 + mi*0.2
            ax.errorbar(np.arange(len(xp))+offset, wv, yerr=ye, fmt='o-', color=MC[model],
                       capsize=4, lw=1.5, ms=6, label=ML[model].split('-')[0])
        agg = bwf.get('aggregate', {}); aw = agg.get('median')
        if aw is not None and np.isfinite(aw):
            ax.axhline(aw, color=MC[model], ls=':', alpha=0.4)
    ax.set_xticks(range(len(BL))); ax.set_xticklabels([str(b) for b in BL])
    ax.set_xlabel('Baseline Magnitude'); ax.set_ylabel('Weber Fraction')
    ax.set_title('C. Weber Fraction Constancy', fontweight='bold')
    ax.set_ylim(-0.1, 2.5); ax.legend(fontsize=9, loc='upper right')

    fig.subplots_adjust(top=0.90)
    save_fig(fig, O, 'Figure2_H2_behavioural')


# ======================================================================
# FIGURE 3: Geometry — Stevens β + Precision Gradient
# ======================================================================
def figure_3(R, O):
    print("\n=== Figure 3: Geometry Characterisation ===")
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)
    fig.text(0.5, 0.97, 'Figure 3. Compression is maximal (\u03b2\u22480) and precision declines with magnitude, consistent with efficient coding.',
             ha='center', fontsize=11, style='italic')

    # Panel A: Stevens β across layers (Llama, all domains)
    ax = fig.add_subplot(gs[0])
    for dom in DOMAINS:
        d = load_json(Path(R)/'paradigm_a'/'llama_instruct'/dom/'paradigm_a_analysis.json')
        if d is None: continue
        ld = d.get('layers', {}); ls_, bs_ = [], []
        for lk in sorted_layer_keys(ld):
            b = None
            for met in ['cosine','euclidean']:
                b = ld[lk].get(met,{}).get('model_fits',{}).get('stevens',{}).get('params',{}).get('beta')
                if b is not None: break
            if b is not None: ls_.append(parse_layer_int(lk)); bs_.append(b)
        if ls_: ax.plot(ls_, bs_, 'o-', color=DC[dom], label=DL[dom], lw=2, ms=4)
    ax.axhline(1, color='black', ls=':', alpha=0.4, label='\u03b2=1 (linear)')
    ax.axhline(0, color='gray', ls='--', alpha=0.4)
    ax.axvspan(16, 31, alpha=0.05, color='blue')
    ax.set_xlabel('Layer'); ax.set_ylabel('Stevens Exponent \u03b2')
    ax.set_title('A. Stevens \u03b2 (Llama)', fontweight='bold'); ax.legend(fontsize=8)

    # Panels B & C: Precision gradient (numerical raw, both models)
    for pi, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[1 + pi])
        supp = load_json(Path(R)/'paradigm_a'/model/'numerical'/'paradigm_c_supplement.json')
        if supp is not None:
            mid = supp.get('summary',{}).get('midpoints')
            pr = None
            if isinstance(supp.get('layers'), list):
                for e in supp['layers']:
                    if e.get('layer') == 20: pr = e.get('raw_precision'); break
            if mid and pr:
                n = min(len(mid), len(pr))
                ax.plot(mid[:n], pr[:n], 'o-', color=DC['numerical'], ms=4, lw=1.5)
                ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Magnitude'); ax.set_ylabel('Precision (1/step)')
        short = 'Llama' if 'llama' in model else 'Mistral'
        ax.set_title(f'{"B" if pi==0 else "C"}. Precision ({short}, Numerical)', fontweight='bold')

    fig.subplots_adjust(top=0.90)
    save_fig(fig, O, 'Figure3_geometry')


# ======================================================================
# FIGURE 4: Causal — Dose-Response + E5 Dissociation
# ======================================================================
def figure_4(R, O):
    print("\n=== Figure 4: Causal Intervention ===")
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)
    fig.text(0.5, 0.97, 'Figure 4. Magnitude subspace is causal at early layers for approximate tasks, but decorative where geometry is strongest.',
             ha='center', fontsize=11, style='italic')

    # Panel A: Llama E5 dose-response
    ax = fig.add_subplot(gs[0])
    e5 = load_json(Path(R)/'paradigm_d'/'llama_e5_b1_patching_results.json')
    if e5 is not None:
        colors_e5 = ['#D62728', '#FF7F0E']
        for ci, (lk, ld) in enumerate(e5.get('layers', {}).items()):
            h7s = ld.get('h7_style_eval', {}); dr = h7s.get('dose_response', {})
            if not dr: continue
            ds = sorted([float(k) for k in dr]); dp = [dr.get(str(d), dr.get(f'{d:.1f}', {})).get('mean_abs_delta_p', 0) for d in ds]
            ln = ld.get('layer', lk); sp = h7s.get('mag_to_random_ratio', 0)
            ax.plot(ds, dp, 'o-', color=colors_e5[ci], lw=2, ms=8, label=f'L{ln} ({sp:.1f}\u00d7)')
    ax.axhline(0, color='black', lw=0.5); ax.set_xlabel('Patching Dose'); ax.set_ylabel('Mean |\u0394p|')
    ax.set_title('A. Dose-Response (Llama, B1)', fontweight='bold'); ax.legend(fontsize=9)

    # Panel B: Mistral E5 dose-response
    ax = fig.add_subplot(gs[1])
    e5 = load_json(Path(R)/'paradigm_d'/'mistral_e5_b1_patching_results.json')
    if e5 is not None:
        colors_e5 = ['#D62728', '#FF7F0E']
        for ci, (lk, ld) in enumerate(e5.get('layers', {}).items()):
            h7s = ld.get('h7_style_eval', {}); dr = h7s.get('dose_response', {})
            if not dr: continue
            ds = sorted([float(k) for k in dr]); dp = [dr.get(str(d), dr.get(f'{d:.1f}', {})).get('mean_abs_delta_p', 0) for d in ds]
            ln = ld.get('layer', lk); sp = h7s.get('mag_to_random_ratio', 0)
            ax.plot(ds, dp, 'o-', color=colors_e5[ci], lw=2, ms=8, label=f'L{ln} ({sp:.1f}\u00d7)')
    ax.axhline(0, color='black', lw=0.5); ax.set_xlabel('Patching Dose'); ax.set_ylabel('Mean |\u0394p|')
    ax.set_title('B. Dose-Response (Mistral, B1)', fontweight='bold'); ax.legend(fontsize=9)

    # Panel C: E5 specificity bar chart (both models)
    ax = fig.add_subplot(gs[2])
    bar_data = []
    for model in MODELS:
        ms = model.replace('_instruct', '')
        e5 = load_json(Path(R)/'paradigm_d'/f'{ms}_e5_b1_patching_results.json')
        if e5 is None: continue
        for lk, ld in e5.get('layers', {}).items():
            h7s = ld.get('h7_style_eval', {})
            ln = ld.get('layer', lk); short = 'Llama' if 'llama' in model else 'Mistral'
            bar_data.append({
                'label': f'{short} L{ln}\n({lk})',
                'mag': h7s.get('mag_mean_abs_delta', 0),
                'rand': h7s.get('random_mean_abs_delta', 0),
                'spec': h7s.get('mag_to_random_ratio', 0)
            })
    if bar_data:
        x = np.arange(len(bar_data)); w = 0.35
        ax.bar(x-w/2, [b['mag'] for b in bar_data], w, label='Magnitude dir', color='#D62728')
        ax.bar(x+w/2, [b['rand'] for b in bar_data], w, label='Random dir', color='gray')
        ax.set_xticks(x); ax.set_xticklabels([b['label'] for b in bar_data], fontsize=8)
        ax.set_ylabel('Mean |\u0394p|'); ax.legend(fontsize=9)
        for i, b in enumerate(bar_data):
            if b['spec'] > 0:
                ax.annotate(f'{b["spec"]:.1f}\u00d7', (i, max(b['mag'], b['rand'])),
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('C. Specificity Ratio', fontweight='bold')

    fig.subplots_adjust(top=0.90)
    save_fig(fig, O, 'Figure4_causal')


# ======================================================================
# FIGURE 5: Controls — Digit Boundary + Variance Partitioning
# ======================================================================
def figure_5(R, O):
    print("\n=== Figure 5: Controls ===")
    from sklearn.linear_model import LinearRegression
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.text(0.5, 0.97, 'Figure 5. Digit-boundary effects are real but subordinate: log-magnitude dominates variance at all layers.',
             ha='center', fontsize=11, style='italic')

    # Panel A: Cohen's d (Llama only for clarity)
    ax = axes[0]
    rob = load_json(Path(R)/'paradigm_a'/'llama_instruct'/'numerical'/'paradigm_c_robustness.json')
    if rob is not None:
        pl = rob.get('robustness',{}).get('digit_boundary',{}).get('per_layer',{})
        ls_, ds_ = [], []
        for lk in sorted(pl.keys(), key=lambda k: int(k)):
            dv = pl[lk].get('cohens_d')
            if dv is not None: ls_.append(int(lk)); ds_.append(dv)
        if ds_:
            ax.bar(ls_, ds_, color=MC['llama_instruct'], alpha=0.7, width=0.8)
            ax.axhline(0.8, color='gray', ls='-', alpha=0.3, label="Large (d=0.8)")
            ax.axvspan(16, 31, alpha=0.05, color='blue', label='Primary layers')
            ax.legend(fontsize=8)
    ax.set_xlabel('Layer'); ax.set_ylabel("Cohen's d (cross- vs within-digit)")
    ax.set_title('A. Digit-Boundary Effect (Llama)', fontweight='bold')

    # Panel B: Variance partitioning (Llama)
    ax = axes[1]
    d = load_json(Path(R)/'paradigm_a'/'llama_instruct'/'numerical'/'paradigm_a_analysis.json')
    if d is not None:
        lm = np.log(np.array(MAGS, dtype=float)); dc = np.array([len(str(m)) for m in MAGS], dtype=float)
        n = len(MAGS); ld_, dd_ = [], []
        for i in range(n):
            for j in range(i+1, n): ld_.append(abs(lm[i]-lm[j])); dd_.append(abs(dc[i]-dc[j]))
        Xl = np.array(ld_).reshape(-1,1); Xd = np.array(dd_).reshape(-1,1); Xf = np.column_stack([ld_, dd_])
        ls_, rl, rd = [], [], []
        for li in range(33):
            pw = get_pairwise(d, li, 'cosine')
            if pw is None or len(pw) != 325: continue
            y = np.array(pw)
            rf = LinearRegression().fit(Xf, y).score(Xf, y)
            rlo = LinearRegression().fit(Xl, y).score(Xl, y)
            rdo = LinearRegression().fit(Xd, y).score(Xd, y)
            ls_.append(li); rl.append(rf - rdo); rd.append(rf - rlo)
        if ls_:
            la = np.array(ls_)
            ax.bar(la-0.2, rl, 0.4, color='#2166AC', alpha=0.8, label='Log-magnitude')
            ax.bar(la+0.2, rd, 0.4, color='#D62728', alpha=0.8, label='Digit-count')
            ax.axvspan(16, 31, alpha=0.05, color='blue')
            ax.legend(fontsize=9)
    ax.set_xlabel('Layer'); ax.set_ylabel('Partial R\u00b2')
    ax.set_title('B. Variance Partitioning (Llama)', fontweight='bold'); ax.set_ylim(0, 0.35)

    fig.subplots_adjust(top=0.90)
    save_fig(fig, O, 'Figure5_controls')


# ======================================================================
# FIGURE 6: Corpus Distribution
# ======================================================================
def figure_6(R, O):
    print("\n=== Figure 6: Corpus Distribution ===")
    d = load_json(Path(R)/'appendix_e'/'corpus_magnitude_distribution.json')
    if d is None: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.text(0.5, 0.97, 'Figure 6. Natural language integer frequencies approximate a 1/s prior, grounding the efficient coding prediction.',
             ha='center', fontsize=11, style='italic')

    # Panel A: Log-log
    ax = axes[0]
    fd = d.get('frequencies', {})
    if isinstance(fd, dict) and fd:
        mg = np.array(sorted([int(k) for k in fd]), dtype=float)
        fr = np.array([fd[str(int(m))] for m in mg], dtype=float)
        mk = (fr>0)&(mg>0); mg, fr = mg[mk], fr[mk]
        ax.scatter(mg, fr, s=3, alpha=0.3, color='#333')
        a = d.get('power_law_fit',{}).get('alpha', 0.773); xf = np.logspace(0, 3, 200)
        ax.plot(xf, xf**(-a)*(fr[0]/xf[0]**(-a)), '-', color='#D62728', lw=2, label=f'Power law (\u03b1={a:.3f})')
        ax.plot(xf, xf**(-1)*(fr[0]/xf[0]**(-1)), '--', color='#2166AC', lw=1.5, alpha=0.7, label='1/s prior (\u03b1=1.0)')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Integer Magnitude'); ax.set_ylabel('Frequency')
        ax.set_title('A. Log-Log Distribution', fontweight='bold'); ax.legend()

    # Panel B: Benford
    ax = axes[1]
    ldd = d.get('leading_digit_distribution', {})
    if ldd:
        dg = list(range(1, 10)); ct = [ldd.get(str(i), 0) for i in dg]; tot = sum(ct)
        obs = [c/tot for c in ct] if tot > 0 else ct; exp = [np.log10(1+1/i) for i in dg]
        x = np.arange(9); w = 0.35
        ax.bar(x-w/2, obs, w, label='Observed', color='#2166AC', alpha=0.8)
        ax.bar(x+w/2, exp, w, label="Benford's Law", color='#D62728', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(dg)
        ax.set_xlabel('Leading Digit'); ax.set_ylabel('Proportion')
        ax.set_title("B. Benford's Law Compliance", fontweight='bold'); ax.legend()

    fig.subplots_adjust(top=0.90)
    save_fig(fig, O, 'Figure6_corpus')


# ======================================================================
# MAIN
# ======================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default=r'C:\weber\results')
    p.add_argument('--output-dir', default=r'C:\weber\results\paper_figures')
    a = p.parse_args(); R = Path(a.results_dir); O = Path(a.output_dir); safe_mkdir(O)
    print("="*70); print("Weber 4.2 \u2014 Paper Figures (6 composites)"); print(f"Output: {O}"); print("="*70)
    if not R.exists(): print(f"ERROR: {R}"); sys.exit(1)
    figure_1(R, O); figure_2(R, O); figure_3(R, O); figure_4(R, O); figure_5(R, O); figure_6(R, O)
    print("\n"+"="*70); print("DONE. 6 paper figures saved to:", O); print("="*70)

if __name__ == '__main__': main()
