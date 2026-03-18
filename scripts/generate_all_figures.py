#!/usr/bin/env python3
"""
Weber's Law Project 4.2 — Complete Figure Generation (F1–F10 + Supplementary)
===============================================================================
Individual pre-registered figures. For combined paper figures, use generate_paper_figures.py.

Usage: python scripts/generate_all_figures.py [--results-dir C:\\weber\\results]
Author: JP Cacioli | March 2026
"""

import json, sys, argparse, warnings, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)

# === HELPERS ===
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

# === CONFIG ===
MODELS = ['llama_instruct', 'mistral_instruct']
ML = {'llama_instruct': 'Llama-3-8B-Instruct', 'mistral_instruct': 'Mistral-7B-Instruct-v0.3', 'llama_base': 'Llama-3-8B (base)'}
DOMAINS = ['numerical', 'temporal', 'spatial']
DL = {'numerical': 'Numerical', 'temporal': 'Temporal', 'spatial': 'Spatial'}
PRIMARY = list(range(16, 32))
MAGS = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,150,200,300,500,700,1000]
BL = [10, 30, 100, 300, 1000]
RATIOS = [1.05, 1.10, 1.20, 1.50, 2.00, 3.00]
DC = {'numerical': '#2166AC', 'temporal': '#B2182B', 'spatial': '#4DAF4A'}
MC = {'llama_instruct': '#2166AC', 'mistral_instruct': '#D6604D'}

plt.rcParams.update({'font.size':10,'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':9,
    'ytick.labelsize':9,'legend.fontsize':9,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'font.family':'sans-serif','axes.spines.top':False,'axes.spines.right':False})

def get_pairwise(data, layer_idx, metric='cosine'):
    """Get pairwise distances. pairwise_cosine/euclidean is list of 33 lists (325 each)."""
    key = f'pairwise_{metric}'
    pw = data.get(key)
    if pw is not None and isinstance(pw, list) and layer_idx < len(pw):
        return pw[layer_idx]
    return None

# === F1 ===
def generate_f1(R, O):
    print("\n=== F1: RSA Heatmap ===")
    for metric in ['cosine', 'euclidean']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
        fig.suptitle(f'F1: RSA Spearman \u03c1 (Weber) \u2014 {metric.title()} Distance', fontsize=13, fontweight='bold')
        for ai, model in enumerate(MODELS):
            ax = axes[ai]; hm = np.full((len(PRIMARY), len(DOMAINS)), np.nan)
            for di, dom in enumerate(DOMAINS):
                d = load_json(Path(R)/'paradigm_a'/model/dom/'paradigm_a_analysis.json')
                if d is None: continue
                for li, layer in enumerate(PRIMARY):
                    lk = find_layer_key(d.get('layers',{}), layer)
                    if lk is None: continue
                    w = d['layers'][lk].get(metric,{}).get('rsa',{}).get('weber',{})
                    rho = w.get('rho') if isinstance(w, dict) else w
                    if rho is not None: hm[li, di] = rho
            im = ax.imshow(hm, aspect='auto', cmap='RdYlBu_r', vmin=0.3, vmax=1.0, interpolation='nearest')
            ax.set_title(ML[model]); ax.set_xlabel('Domain')
            ax.set_xticks(range(3)); ax.set_xticklabels([DL[d] for d in DOMAINS])
            if ai==0: ax.set_ylabel('Layer'); ax.set_yticks(range(len(PRIMARY))); ax.set_yticklabels(PRIMARY)
            else: ax.set_yticks(range(len(PRIMARY)))
            for i in range(hm.shape[0]):
                for j in range(hm.shape[1]):
                    v=hm[i,j]
                    if not np.isnan(v): ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,color='white' if v>0.75 else 'black')
        fig.subplots_adjust(right=0.88); cax=fig.add_axes([0.90,0.15,0.02,0.7])
        fig.colorbar(im, cax=cax, label='RSA Spearman \u03c1'); save_fig(fig, O, f'F1_rsa_heatmap_{metric}')

# === F2 ===
def generate_f2(R, O):
    print("\n=== F2: Psychometric Curves ===")
    cm = plt.cm.viridis
    for model in MODELS:
        raw = load_json(Path(R)/'paradigm_b'/model/'numerical'/'paradigm_b_raw.json')
        if raw is None: continue
        items = raw if isinstance(raw, list) else raw.get('items', raw.get('results', []))
        acc = {}
        for it in items:
            bl=it.get('baseline',it.get('nominal_baseline')); rt=it.get('ratio',it.get('nominal_ratio'))
            c=it.get('correct',it.get('is_correct'))
            if bl is None or rt is None or c is None: continue
            acc.setdefault((min(BL,key=lambda b:abs(b-bl)), min(RATIOS,key=lambda r:abs(r-rt))), []).append(1 if c else 0)
        fig, ax = plt.subplots(figsize=(8,6))
        fig.suptitle(f'F2: Psychometric Curves (B1) \u2014 {ML[model]}', fontsize=13, fontweight='bold')
        cols = cm(np.linspace(0.1, 0.9, len(BL)))
        for bi, bl in enumerate(BL):
            xs,ys=[],[]
            for rt in RATIOS:
                if (bl,rt) in acc: xs.append(rt); ys.append(np.mean(acc[(bl,rt)]))
            if xs: ax.plot(xs, ys, 'o-', color=cols[bi], label=f'Baseline = {bl}', linewidth=2, markersize=6)
        ax.set_xlabel('Magnitude Ratio'); ax.set_ylabel('Proportion Correct')
        ax.set_xscale('log'); ax.set_xticks(RATIOS); ax.set_xticklabels([f'{r:.2f}' for r in RATIOS])
        ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='Chance'); ax.set_ylim(0.4,1.05)
        ax.legend(loc='lower right'); save_fig(fig, O, f'F2_psychometric_{model}')

# === F3 ===
def generate_f3(R, O):
    print("\n=== F3: Weber Fraction Constancy ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('F3: Weber Fraction Constancy Across Baselines', fontsize=13, fontweight='bold')
    for ai, model in enumerate(MODELS):
        ax = axes[ai]
        ps = load_json(Path(R)/'paradigm_b'/model/'numerical'/'psychometric_corrected.json')
        if ps is None: ax.set_title(f'{ML[model]}\n[No data]'); continue
        ax.axhspan(0.10, 0.25, alpha=0.15, color='green', label='Human range (0.10\u20130.25)')
        bwf = ps.get('bootstrap_weber_fractions', {})
        xp, wv, cl, ch = [], [], [], []
        for bk in ['10.0','30.0','100.0','300.0','1000.0']:
            bd = bwf.get(bk, {}); wf = bd.get('median', bd.get('mean'))
            if wf is None or not np.isfinite(wf): continue
            xp.append(float(bk)); wv.append(wf); cl.append(bd.get('ci_low',wf)); ch.append(bd.get('ci_high',wf))
        if wv:
            ye = [[max(0,w-c) for w,c in zip(wv,cl)], [max(0,c-w) for w,c in zip(wv,ch)]]
            ax.errorbar(range(len(xp)), wv, yerr=ye, fmt='o-', color=MC[model], capsize=5, lw=2, ms=8, label='Per-baseline WF')
            ax.set_xticks(range(len(xp))); ax.set_xticklabels([f'{int(x)}' for x in xp])
        agg = bwf.get('aggregate', {}); aw = agg.get('median', agg.get('mean'))
        if aw is not None and np.isfinite(aw):
            ax.axhline(aw, color=MC[model], ls='--', alpha=0.6, label=f'Aggregate WF = {aw:.3f}')
            if agg.get('ci_low') is not None: ax.axhspan(agg['ci_low'], agg['ci_high'], alpha=0.08, color=MC[model])
        ax.set_xlabel('Baseline Magnitude')
        if ai==0: ax.set_ylabel('Weber Fraction')
        ax.set_title(ML[model]); ax.set_ylim(-0.1, max(2.0, max(wv,default=0)*1.3)); ax.legend(loc='upper right', fontsize=8)
    save_fig(fig, O, 'F3_weber_fraction_constancy')

# === F4 ===
def generate_f4(R, O):
    print("\n=== F4: Precision Gradient ===")
    for model in MODELS:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'F4: Precision Gradient \u2014 {ML[model]}', fontsize=13, fontweight='bold')
        for di, dom in enumerate(DOMAINS):
            supp = load_json(Path(R)/'paradigm_a'/model/dom/'paradigm_c_supplement.json')
            rob = load_json(Path(R)/'paradigm_a'/model/dom/'paradigm_c_robustness.json')
            mid, pr, pn = None, None, None
            if supp is not None:
                mid = supp.get('summary',{}).get('midpoints')
                if isinstance(supp.get('layers'), list):
                    for e in supp['layers']:
                        if e.get('layer') == 20:
                            pr = e.get('raw_precision')
                            pn = e.get('normalised_precision_log', e.get('normalised_precision'))
                            break
            if pr is None and rob is not None:
                pc = rob.get('paradigm_c',{}).get('precision',{})
                for rep in ['20','24','16']:
                    if rep in pc.get('raw',{}): pr=pc['raw'][rep]; pn=pc.get('normalised',{}).get(rep); break
            if mid is None: mid = [(MAGS[i]+MAGS[i+1])/2 for i in range(len(MAGS)-1)]
            ax0 = axes[0,di]
            if pr is not None:
                n=min(len(mid),len(pr)); ax0.plot(mid[:n],pr[:n],'o-',color=DC[dom],ms=4,lw=1.5)
                ax0.set_xscale('log'); ax0.set_yscale('log')
            ax0.set_title(f'{DL[dom]} \u2014 Raw'); ax0.set_xlabel('Magnitude')
            if di==0: ax0.set_ylabel('Precision (1/step dist)')
            ax1 = axes[1,di]
            if pn is not None:
                n=min(len(mid),len(pn)); ax1.plot(mid[:n],pn[:n],'o-',color=DC[dom],ms=4,lw=1.5)
                ax1.set_xscale('log'); m=np.nanmean(pn[:n])
                ax1.axhline(m, color='gray', ls='--', alpha=0.5, label=f'Mean = {m:.3f}'); ax1.legend(fontsize=8)
            ax1.set_title(f'{DL[dom]} \u2014 Log-Normalised'); ax1.set_xlabel('Magnitude')
            if di==0: ax1.set_ylabel('Precision / log-step')
        fig.tight_layout(rect=[0,0,1,0.95]); save_fig(fig, O, f'F4_precision_gradient_{model}')

# === F5 ===
def generate_f5(R, O):
    print("\n=== F5: Paradigm D Dose-Response ===")
    for model in MODELS:
        ms = model.replace('_instruct','')
        dr = load_json(Path(R)/'paradigm_d'/f'{ms}_numerical_paradigm_d_results.json')
        if dr is None: dr = load_json(Path(R)/'paradigm_d'/f'{model}_numerical_paradigm_d_results.json')
        e5 = load_json(Path(R)/'paradigm_d'/f'{ms}_e5_b1_patching_results.json')
        if e5 is None: e5 = load_json(Path(R)/'paradigm_d'/f'{model}_e5_b1_patching_results.json')
        np_ = (1 if dr else 0) + (1 if e5 else 0)
        if np_==0: continue
        fig, aa = plt.subplots(1, max(np_,1), figsize=(7*max(np_,1), 6))
        if np_==1: aa=[aa]
        fig.suptitle(f'F5: Dose-Response \u2014 {ML[model]}', fontsize=13, fontweight='bold')
        pi=0
        if dr is not None:
            ax=aa[pi]; pi+=1; h7=dr.get('h7',{}); dose=h7.get('dose_response',{})
            ds=sorted([float(k) for k in dose]); dp=[dose.get(str(d),dose.get(f'{d:.1f}',{})).get('mean_abs_delta_p',0) for d in ds]
            rm=h7.get('aggregate',{}).get('random_mean_abs_delta',0)
            ax.plot(ds,dp,'o-',color='#D62728',lw=2,ms=8,label='Magnitude direction')
            ax.axhline(rm,color='gray',ls='--',lw=1.5,label=f'Random mean = {rm:.4f}')
            ax.axhline(0,color='black',lw=0.5); ax.set_xlabel('Patching Dose'); ax.set_ylabel('Mean |\u0394p|')
            ax.set_title('Primary H7 (Symbolic)'); ax.legend(loc='upper left',fontsize=8)
        if e5 is not None:
            ax=aa[pi] if pi<len(aa) else aa[-1]; cols=['#D62728','#FF7F0E','#2CA02C','#9467BD']; ci=0
            for lk,ld in e5.get('layers',{}).items():
                h7s=ld.get('h7_style_eval',{}); dose=h7s.get('dose_response',{})
                if not dose: continue
                ds=sorted([float(k) for k in dose]); dp=[dose.get(str(d),dose.get(f'{d:.1f}',{})).get('mean_abs_delta_p',0) for d in ds]
                ln=ld.get('layer',lk); sp=h7s.get('mag_to_random_ratio',0)
                ax.plot(ds,dp,'o-',color=cols[ci%4],lw=2,ms=8,label=f'Mag L{ln} ({sp:.1f}\u00d7)'); ci+=1
            ax.axhline(0,color='black',lw=0.5); ax.set_xlabel('Patching Dose'); ax.set_ylabel('Mean |\u0394p|')
            ax.set_title('E5 Exploratory (B1)'); ax.legend(loc='upper left',fontsize=8)
        fig.tight_layout(rect=[0,0,1,0.93]); save_fig(fig, O, f'F5_dose_response_{model}')

# === F6 ===
def generate_f6(R, O):
    print("\n=== F6: Digit-Boundary Cohen's d ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("F6: Digit-Boundary Effect (Cohen's d)", fontsize=13, fontweight='bold')
    for ai, model in enumerate(MODELS):
        ax=axes[ai]; rob=load_json(Path(R)/'paradigm_a'/model/'numerical'/'paradigm_c_robustness.json')
        if rob is None: ax.set_title(f'{ML[model]}\n[No data]'); continue
        pl=rob.get('robustness',{}).get('digit_boundary',{}).get('per_layer',{})
        ls_,ds_=[],[]
        for lk in sorted(pl.keys(), key=lambda k: int(k)):
            dv=pl[lk].get('cohens_d')
            if dv is not None: ls_.append(int(lk)); ds_.append(dv)
        if ds_:
            ax.bar(ls_,ds_,color=MC[model],alpha=0.7,width=0.8)
            ax.axhline(0.2,color='gray',ls=':',alpha=0.5,label="Small (0.2)")
            ax.axhline(0.5,color='gray',ls='--',alpha=0.5,label="Medium (0.5)")
            ax.axhline(0.8,color='gray',ls='-',alpha=0.3,label="Large (0.8)")
            ax.axvspan(16,31,alpha=0.05,color='blue',label='Primary'); ax.legend(fontsize=7)
        ax.set_xlabel('Layer'); ax.set_title(ML[model])
        if ai==0: ax.set_ylabel("Cohen's d")
    save_fig(fig, O, 'F6_digit_boundary_cohens_d')

# === F7 ===
def generate_f7(R, O):
    print("\n=== F7: Frequency-Matched Noun Control ===")
    model='llama_instruct'
    d=load_json(Path(R)/'paradigm_a'/model/'numerical'/'paradigm_a_analysis.json')
    if d is None: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'F7: Frequency-Matched Noun Control \u2014 {ML[model]} (Layer 20)', fontsize=13, fontweight='bold')
    # Number RDM from pairwise_cosine (list of 33 lists)
    pw = get_pairwise(d, 20, 'cosine')
    ax=axes[0]
    if pw is not None and len(pw)==325:
        n=len(MAGS); rdm=np.zeros((n,n)); idx=0
        for i in range(n):
            for j in range(i+1,n): rdm[i,j]=rdm[j,i]=pw[idx]; idx+=1
        im=ax.imshow(rdm,cmap='viridis',aspect='equal')
        ts=max(1,n//6); ax.set_xticks(range(0,n,ts)); ax.set_xticklabels([str(MAGS[i]) for i in range(0,n,ts)],rotation=45,fontsize=7)
        ax.set_yticks(range(0,n,ts)); ax.set_yticklabels([str(MAGS[i]) for i in range(0,n,ts)],fontsize=7)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04); ax.set_title('Number RDM (structured)')
    else:
        ax.text(0.5,0.5,'Number RDM not available',ha='center',va='center',transform=ax.transAxes); ax.set_title('Number RDM')
    axes[1].text(0.5,0.5,'Noun RDM: requires separate\nhidden state extraction.\nSee paradigm_a_figures.py',
                ha='center',va='center',transform=axes[1].transAxes,fontsize=10,style='italic')
    axes[1].set_title('Frequency-Matched Noun RDM')
    fig.tight_layout(rect=[0,0,1,0.93]); save_fig(fig, O, 'F7_frequency_matched_nouns')

# === F8 ===
def generate_f8(R, O):
    print("\n=== F8: Stevens \u03b2 ===")
    for model in MODELS:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'F8: Stevens \u03b2 Across Layers \u2014 {ML[model]}', fontsize=13, fontweight='bold')
        hd=False
        for dom in DOMAINS:
            d=load_json(Path(R)/'paradigm_a'/model/dom/'paradigm_a_analysis.json')
            if d is None: continue
            ld=d.get('layers',{}); ls_,bs_=[],[]
            for lk in sorted_layer_keys(ld):
                b=None
                for met in ['cosine','euclidean']:
                    b=ld[lk].get(met,{}).get('model_fits',{}).get('stevens',{}).get('params',{}).get('beta')
                    if b is not None: break
                if b is not None: ls_.append(parse_layer_int(lk)); bs_.append(b)
            if ls_: hd=True; ax.plot(ls_,bs_,'o-',color=DC[dom],label=DL[dom],lw=2,ms=5)
        if hd:
            ax.axhline(1,color='black',ls=':',alpha=0.4,label='\u03b2=1 (linear)')
            ax.axhline(0,color='gray',ls='--',alpha=0.4,label='\u03b2\u21920 (max compression)')
            ax.axvspan(16,31,alpha=0.05,color='blue'); ax.set_xlabel('Layer'); ax.set_ylabel('Stevens \u03b2'); ax.legend()
        else: ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)
        save_fig(fig, O, f'F8_stevens_beta_{model}')

# === F9 ===
def generate_f9(R, O):
    print("\n=== F9: Corpus Distribution ===")
    d=load_json(Path(R)/'appendix_e'/'corpus_magnitude_distribution.json')
    if d is None: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("F9: Corpus Integer Distribution (OpenWebText, 500k docs)", fontsize=13, fontweight='bold')
    fd=d.get('frequencies',{})
    if isinstance(fd,dict) and fd:
        mg=np.array(sorted([int(k) for k in fd]),dtype=float)
        fr=np.array([fd[str(int(m))] for m in mg],dtype=float)
        mk=(fr>0)&(mg>0); mg,fr=mg[mk],fr[mk]
        ax=axes[0]; ax.scatter(mg,fr,s=3,alpha=0.3,color='#333')
        a=d.get('power_law_fit',{}).get('alpha',0.773); xf=np.logspace(0,3,200)
        ax.plot(xf, xf**(-a)*(fr[0]/xf[0]**(-a)), '-', color='#D62728', lw=2, label=f'Power law (\u03b1={a:.3f})')
        ax.plot(xf, xf**(-1)*(fr[0]/xf[0]**(-1)), '--', color='#2166AC', lw=1.5, alpha=0.7, label='1/s prior (\u03b1=1.0)')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Integer Magnitude'); ax.set_ylabel('Frequency')
        ax.set_title('Log-Log Distribution'); ax.legend()
    ldd=d.get('leading_digit_distribution',{})
    if ldd:
        dg=list(range(1,10)); ct=[ldd.get(str(i),0) for i in dg]; tot=sum(ct)
        obs=[c/tot for c in ct] if tot>0 else ct; exp=[np.log10(1+1/i) for i in dg]
        x=np.arange(9); w=0.35; ax=axes[1]
        ax.bar(x-w/2,obs,w,label='Observed',color='#2166AC',alpha=0.8)
        ax.bar(x+w/2,exp,w,label="Benford's Law",color='#D62728',alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(dg); ax.set_xlabel('Leading Digit'); ax.set_ylabel('Proportion')
        ax.set_title("Benford's Law Compliance"); ax.legend()
    fig.tight_layout(rect=[0,0,1,0.93]); save_fig(fig, O, 'F9_corpus_distribution')

# === F10 ===
def generate_f10(R, O):
    print("\n=== F10: Variance Partitioning ===")
    from sklearn.linear_model import LinearRegression
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('F10: Variance Partitioning \u2014 Log-Mag vs Digit-Count', fontsize=13, fontweight='bold')
    lm=np.log(np.array(MAGS,dtype=float)); dc=np.array([len(str(m)) for m in MAGS],dtype=float)
    n=len(MAGS); ld_,dd_=[],[]
    for i in range(n):
        for j in range(i+1,n): ld_.append(abs(lm[i]-lm[j])); dd_.append(abs(dc[i]-dc[j]))
    Xl=np.array(ld_).reshape(-1,1); Xd=np.array(dd_).reshape(-1,1); Xf=np.column_stack([ld_,dd_])
    for ai,model in enumerate(MODELS):
        ax=axes[ai]; d=load_json(Path(R)/'paradigm_a'/model/'numerical'/'paradigm_a_analysis.json')
        if d is None: ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes); ax.set_title(ML[model]); continue
        ls_,rl,rd=[],[],[]
        for li in range(33):
            pw=get_pairwise(d, li, 'cosine')
            if pw is None or len(pw)!=325: continue
            y=np.array(pw)
            rf=LinearRegression().fit(Xf,y).score(Xf,y)
            rlo=LinearRegression().fit(Xl,y).score(Xl,y)
            rdo=LinearRegression().fit(Xd,y).score(Xd,y)
            ls_.append(li); rl.append(rf-rdo); rd.append(rf-rlo)
        if ls_:
            la=np.array(ls_)
            ax.bar(la-0.2,rl,0.4,color='#2166AC',alpha=0.8,label='Log-mag (partial R\u00b2)')
            ax.bar(la+0.2,rd,0.4,color='#D62728',alpha=0.8,label='Digit-count (partial R\u00b2)')
            ax.axvspan(16,31,alpha=0.05,color='blue'); ax.legend(fontsize=8)
        ax.set_xlabel('Layer'); ax.set_title(ML[model]); ax.set_ylim(0,1.0)
        if ai==0: ax.set_ylabel('Partial R\u00b2')
    save_fig(fig, O, 'F10_variance_partitioning')

# === SUPPLEMENTARY ===
def generate_e1(R, O):
    print("\n=== E1: Base vs Instruct ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('E1: Base vs Instruct \u2014 Numerical', fontsize=13, fontweight='bold')
    for ai,(met,ml) in enumerate([('cosine','Cosine'),('euclidean','Euclidean')]):
        ax=axes[ai]
        for model,lab,col,sty in [('llama_instruct','Instruct','#2166AC','-'),('llama_base','Base','#2166AC','--')]:
            d=load_json(Path(R)/'paradigm_a'/model/'numerical'/'paradigm_a_analysis.json')
            if d is None: continue
            ld=d.get('layers',{}); ls_,rs_=[],[]
            for lk in sorted_layer_keys(ld):
                r=ld[lk].get(met,{}).get('model_fits',{}).get('weber',{}).get('r2')
                if r is not None: ls_.append(parse_layer_int(lk)); rs_.append(r)
            if ls_: ax.plot(ls_,rs_,f'o{sty}',color=col,lw=2,ms=4,label=lab,alpha=0.8 if sty=='-' else 0.5)
        ax.set_xlabel('Layer'); ax.set_ylabel('Weber R\u00b2'); ax.set_title(ml)
        ax.axvspan(16,31,alpha=0.05,color='blue'); ax.legend(); ax.set_ylim(0,1)
    save_fig(fig, O, 'E1_base_vs_instruct')

def generate_e5(R, O):
    print("\n=== E5: Causal Dissociation ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('E5: Early vs Late Layer Causal Effect (B1)', fontsize=13, fontweight='bold')
    for ai, model in enumerate(MODELS):
        ax=axes[ai]; ms=model.replace('_instruct','')
        e5=load_json(Path(R)/'paradigm_d'/f'{ms}_e5_b1_patching_results.json')
        if e5 is None: e5=load_json(Path(R)/'paradigm_d'/f'{model}_e5_b1_patching_results.json')
        if e5 is None: ax.set_title(f'{ML[model]}\n[No data]'); continue
        labs,mgs,rns,sps=[],[],[],[]
        for lk,ld in e5.get('layers',{}).items():
            h=ld.get('h7_style_eval',{}); ln=ld.get('layer',lk)
            labs.append(f'L{ln} ({lk})'); mgs.append(h.get('mag_mean_abs_delta',0))
            rns.append(h.get('random_mean_abs_delta',0)); sps.append(h.get('mag_to_random_ratio',0))
        if labs:
            x=np.arange(len(labs)); w=0.35
            ax.bar(x-w/2,mgs,w,label='Magnitude dir',color='#D62728')
            ax.bar(x+w/2,rns,w,label='Random dir',color='gray')
            ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8); ax.set_ylabel('Mean |\u0394p|'); ax.legend()
            for i,s in enumerate(sps):
                if s>0: ax.annotate(f'{s:.1f}\u00d7',(i,max(mgs[i],rns[i])),ha='center',va='bottom',fontsize=9,fontweight='bold')
        ax.set_title(ML[model])
    save_fig(fig, O, 'E5_causal_dissociation')

def generate_b_task(R, O):
    print("\n=== B Task Comparison ===")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Paradigm B: Task Accuracy', fontsize=13, fontweight='bold')
    tasks=['B1','B2','B3','Symbolic']; x=np.arange(4); w=0.35
    for mi, model in enumerate(MODELS):
        b23=load_json(Path(R)/'paradigm_b'/model/'numerical'/'paradigm_b_b2b3_summary.json')
        td=b23.get('tasks',{}) if b23 else {}
        ac=[]
        for tk in ['b1','b2','b3','symbolic']:
            d=td.get(tk,{}); ac.append(d.get('accuracy',np.nan) if d else np.nan)
        off=-w/2+mi*w; ax.bar(x+off,ac,w,label=ML[model],color=MC[model],alpha=0.8)
        for i,a in enumerate(ac):
            if not np.isnan(a): ax.text(x[i]+off,a+0.02,f'{a:.1%}' if a<=1 else f'{a:.1f}%',ha='center',va='bottom',fontsize=8)
    ax.axhline(0.5,color='gray',ls='--',alpha=0.5,label='Chance')
    ax.set_xticks(x); ax.set_xticklabels(tasks); ax.set_ylabel('Accuracy'); ax.set_ylim(0,1.15); ax.legend()
    save_fig(fig, O, 'B_task_comparison')

def generate_controls(R, O):
    print("\n=== Controls Summary ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Robustness Controls', fontsize=13, fontweight='bold')
    # Shuffled
    ax=axes[0]; ax.set_title('Shuffled-Magnitude Check')
    bd=[]
    for model in MODELS:
        sh=load_json(Path(R)/'robustness'/f'{model}_shuffled_magnitude.json')
        if sh is None: continue
        s=sh.get('summary',{})
        bd.append((ML[model].split('-')[0], s.get('primary_mean_rho_shuffled',0), s.get('primary_mean_rho_original',0)))
    if bd:
        x=np.arange(len(bd)); w=0.35
        ax.bar(x-w/2,[b[1] for b in bd],w,label='Shuffled (token)',color='#2166AC')
        ax.bar(x+w/2,[b[2] for b in bd],w,label='Original carrier',color='lightgray')
        ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bd])
        for i,b in enumerate(bd):
            ax.text(i-w/2,b[1]+0.02,f'\u03c1={b[1]:.3f}',ha='center',fontsize=8)
            ax.text(i+w/2,max(b[2],0)+0.02,f'\u03c1={b[2]:.3f}',ha='center',fontsize=8)
        ax.set_ylabel('RSA Spearman \u03c1'); ax.legend(fontsize=8); ax.set_ylim(-0.2,1.1)
    # Unit boundary
    ax2=axes[1]; ax2.set_title('Unit-Boundary Check')
    ud=[]
    for model in MODELS:
        ub=load_json(Path(R)/'robustness'/f'{model}_unit_boundary.json')
        if ub is None: continue
        doms=ub.get('domains',{})
        eq_sims, diff_sims = [], []
        for dk in ['temporal','spatial']:
            dd=doms.get(dk,{})
            eq=dd.get('mean_equivalent_similarity'); diff=dd.get('mean_different_magnitude_similarity')
            if eq is not None: eq_sims.append(eq)
            if diff is not None: diff_sims.append(diff)
        if eq_sims:
            ud.append((ML[model].split('-')[0], np.mean(eq_sims), np.mean(diff_sims)))
    if ud:
        x=np.arange(len(ud)); w=0.35
        ax2.bar(x-w/2,[b[1] for b in ud],w,label='Equivalent mag\n(cross-unit)',color='#2166AC')
        ax2.bar(x+w/2,[b[2] for b in ud],w,label='Different mag\n(same unit)',color='lightcoral')
        ax2.set_xticks(x); ax2.set_xticklabels([b[0] for b in ud])
        for i,b in enumerate(ud):
            ax2.text(i-w/2,b[1]+0.01,f'{b[1]:.3f}',ha='center',fontsize=8)
            ax2.text(i+w/2,b[2]+0.01,f'{b[2]:.3f}',ha='center',fontsize=8)
        ax2.set_ylabel('Mean Cosine Similarity'); ax2.legend(fontsize=7)
    fig.tight_layout(rect=[0,0,1,0.93]); save_fig(fig, O, 'Controls_summary')

# === MAIN ===
def main():
    p=argparse.ArgumentParser(); p.add_argument('--results-dir',default=r'C:\weber\results')
    p.add_argument('--output-dir',default=r'C:\weber\results\figures'); p.add_argument('--skip-supplementary',action='store_true')
    a=p.parse_args(); R=Path(a.results_dir); O=Path(a.output_dir); safe_mkdir(O)
    print("="*70); print("Weber 4.2 \u2014 All Figures"); print(f"Results: {R}"); print(f"Output: {O}"); print("="*70)
    if not R.exists(): print(f"ERROR: {R}"); sys.exit(1)
    generate_f1(R,O); generate_f2(R,O); generate_f3(R,O); generate_f4(R,O); generate_f5(R,O)
    generate_f6(R,O); generate_f7(R,O); generate_f8(R,O); generate_f9(R,O); generate_f10(R,O)
    if not a.skip_supplementary:
        print("\n"+"="*70); print("Supplementary"); print("="*70)
        generate_e1(R,O); generate_e5(R,O); generate_b_task(R,O); generate_controls(R,O)
    print("\n"+"="*70); print("DONE:", O); print("="*70)
    for fn in ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']:
        ms=list(O.glob(f'{fn}_*.png')); print(f"  {fn}: {len(ms)} file(s)")

if __name__=='__main__': main()
