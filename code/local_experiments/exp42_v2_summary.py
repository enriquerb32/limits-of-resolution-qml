"""exp42_v2_summary.py - v2 headline quantities from the archives (no training).
Per model: test AUC mean/SD over seeds. Per pair: seed-mean gap, paired DeLong on seed-averaged components (iid SE),
year-cluster CR1 SE, calendar-year block bootstrap and years x seeds bootstrap (B=4000), MDE = 2.80 SE, seed share."""
import json, sys, os
import numpy as np, pandas as pd
from scipy.stats import norm
R = sys.argv[1] if len(sys.argv) > 1 else 'results'
J = R + '/json/'
core = np.load(J + 'results_core_hybrids_controls_fixet0_p60_28.1_probs.npz')
cl = np.load(J + 'results_rerun_train_only_canonical_probs.npz')
e28 = np.load(J + 'results_exp28_orig_probs.npz')
y = core['y_te'].astype(int)
assert (cl['y_te'] == y).all() and (e28['y_te'] == y).all()
meta = pd.read_csv(R + '/test_meta.csv'); yr = pd.to_datetime(meta.fecha).dt.year.values
P = {k[:-6]: core[k] for k in core if k.endswith('_probs')}
P['RF'] = cl['train_only_RF_probs']; P['XGB'] = cl['train_only_XGB_probs']
P['RF_untuned'] = cl['train_only_RF_untuned_probs']; P['XGB_untuned'] = cl['train_only_XGB_untuned_probs']
P['CLIM'] = e28['clim_te'][None]
P['LSTM_QTL_first'] = P.pop('LSTM_QTL'); P['LSTM_QTL'] = core['QTL_FIXED_probs']   # corrected transfer model (exp27)
pos, neg = y == 1, y == 0
def comps(p):
    a, b = p[pos], p[neg]
    psi = (a[:, None] > b[None]) + 0.5 * (a[:, None] == b[None])
    A = psi.mean(); d = np.empty(len(p)); d[pos] = (psi.mean(1) - A) / pos.sum(); d[neg] = (psi.mean(0) - A) / neg.sum()
    return A, d
C = {}
for k, v in P.items():
    r = [comps(s) for s in v]; C[k] = (np.array([t[0] for t in r]), np.mean([t[1] for t in r], 0))
def viid(d):
    n1, n0 = pos.sum(), neg.sum(); return (d[pos]**2).sum()*n1/(n1-1) + (d[neg]**2).sum()*n0/(n0-1)
G = np.unique(yr)
def vcl(d):
    s = np.array([d[yr == g].sum() for g in G]); return len(G)/(len(G)-1)*(s**2).sum()
def auc_seedmean(p, idx):
    yy = y[idx]; out = []
    for s in p:
        q = s[idx]; a, b = q[yy == 1], q[yy == 0]
        out.append(((a[:, None] > b[None]) + 0.5*(a[:, None] == b[None])).mean())
    return np.array(out)
rng = np.random.default_rng(0); B = 4000
rows_by_year = [np.where(yr == g)[0] for g in G]
boot_years = [np.concatenate([rows_by_year[i] for i in rng.integers(0, len(G), len(G))]) for _ in range(B)]
models = {k: dict(mean=float(v[0].mean()), sd=float(v[0].std(ddof=1)) if len(v[0]) > 1 else 0.0, n_seeds=len(v[0])) for k, v in C.items()}
pairs = [('RF','XGB'),('RF','PureLSTM'),('XGB','PureLSTM'),('RF','LSTM_QNN'),('RF','RF_2273'),('RF','RF_untuned'),('XGB','XGB_untuned'),
         ('PureLSTM','LSTM_QNN'),('LSTM_QNN','CTRL_LINEAR4D'),('LSTM_QNN','LSTM_PIQNN'),('LSTM_QTL','QTL_CLASSICAL'),('LSTM_QTL','QTL_TENSOR1'),
         ('LSTM_QTL','LSTM_QNN'),('RF','LSTM_QTL'),('LSTM_QTL','PureLSTM'),('FOURIER_FULL2','FOURIER2'),('LSTM_PIQNN','LSTM_QNN')] + [(f, 'LSTM_QNN') for f in ['FOURIER1','FOURIER2','FOURIER3','FOURIER4','FOURIER_FULL1','FOURIER_FULL2','FROZEN_RANDOM','FROZEN_IDENTITY']] \
        + [(m, 'CLIM') for m in ['RF','XGB','PureLSTM','LSTM_QNN','CTRL_LINEAR4D','FOURIER_FULL1']]
out = {}
for a, b in pairs:
    d = C[a][1] - C[b][1]; gap = C[a][0].mean() - C[b][0].mean()
    si, sc = viid(d)**.5, vcl(d)**.5
    # block bootstrap over years (seed means), and years x seeds
    pa, pb = P[a], P[b]
    bb, tw = [], []
    for k, idx in enumerate(boot_years[:2000]):
        ga, gb = auc_seedmean(pa, idx), auc_seedmean(pb, idx)
        bb.append(ga.mean() - gb.mean())
        sa = rng.integers(0, len(ga), len(ga)); sb = sa if len(gb) == len(ga) else rng.integers(0, len(gb), len(gb))
        tw.append(ga[sa].mean() - gb[sb].mean())
    sd_seed = np.std(C[a][0] - C[b][0], ddof=1) if len(C[a][0]) == len(C[b][0]) and len(C[a][0]) > 1 else float('nan')
    out[f'{a}-{b}'] = dict(gap=float(gap), se_iid=float(si), p_delong=float(2*norm.sf(abs(gap)/si)),
                           ci_delong=[float(gap-1.96*si), float(gap+1.96*si)], se_year=float(sc), ratio_year_iid=float(sc/si),
                           ci_block=np.quantile(bb, [.025, .975]).tolist(), ci_two_way=np.quantile(tw, [.025, .975]).tolist(),
                           mde_iid=float(2.80*si), mde_year=float(2.80*sc), seed_sd_gap=float(sd_seed),
                           seed_share=float(sd_seed**2/15/(si**2)) if sd_seed == sd_seed else None)
summary = dict(n_test=int(len(y)), prevalence=float(y.mean()), n_years=int(len(G)),
               rows_by_station=meta.ciudad.value_counts().to_dict(), models=models, pairs=out)
json.dump(summary, open(R + '/local_exp42_v2_summary.json', 'w'), indent=1)
print(f"n={len(y)} prev={y.mean():.3f} years={len(G)}")
for k, v in sorted(models.items(), key=lambda t: -t[1]['mean']): print(f"{k:18s} {v['mean']:.4f} ± {v['sd']:.4f}")
for k, v in out.items():
    res = 'RESOLVED' if all(c[0] > 0 or c[1] < 0 for c in [v['ci_delong'], v['ci_block']]) and abs(v['gap']) > 1.96*v['se_year']*1.0 else ''
    print(f"{k:28s} gap {v['gap']:+.4f} p={v['p_delong']:.3f} DeLong[{v['ci_delong'][0]:+.3f},{v['ci_delong'][1]:+.3f}] block[{v['ci_block'][0]:+.3f},{v['ci_block'][1]:+.3f}] ratio {v['ratio_year_iid']:.2f} MDE {v['mde_iid']:.3f} {res}")
