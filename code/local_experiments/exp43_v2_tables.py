"""exp43_v2_tables.py - v2 tables computed from the archives (no training): noise budget, Brier and calibration
(raw and Platt, climatology raw and Platt), within-station AUC, station x month label, resolution curve,
jackknife, single-seed verdicts. Output: results/local_exp43_v2_tables.json"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
R = sys.argv[1]; J = R + '/json/'
core = np.load(J + 'results_core_hybrids_controls_fixet0_p60_28.1_probs.npz')
cl = np.load(J + 'results_rerun_train_only_canonical_probs.npz')
e28 = np.load(J + 'results_exp28_orig_probs.npz'); sm = np.load(J + 'results_exp28_stationmonth_probs.npz')
y = core['y_te'].astype(int); meta = pd.read_csv(R + '/test_meta.csv')
yr = pd.to_datetime(meta.fecha).dt.year.values; st = meta.ciudad.values; G = np.unique(yr)
P = {'RF': cl['train_only_RF_probs'], 'XGB': cl['train_only_XGB_probs'], 'PureLSTM': core['PureLSTM_probs'],
     'CTRL_LINEAR4D': core['CTRL_LINEAR4D_probs'], 'LSTM_QNN': core['LSTM_QNN_probs'], 'LSTM_QTL': core['QTL_FIXED_probs'],
     'FOURIER_FULL1': core['FOURIER_FULL1_probs']}
clim = e28['clim_te']
rng = np.random.default_rng(1); B = 2000
rows = [np.where(yr == g)[0] for g in G]
BOOT = [np.concatenate([rows[i] for i in rng.integers(0, len(G), len(G))]) for _ in range(B)]
def auc(yy, p):
    a, b = p[yy == 1], p[yy == 0]; return ((a[:, None] > b[None]) + .5 * (a[:, None] == b[None])).mean()
def comps(yy, p):
    pos, neg = yy == 1, yy == 0; a, b = p[pos], p[neg]
    psi = (a[:, None] > b[None]) + .5 * (a[:, None] == b[None]); A = psi.mean()
    d = np.empty(len(p)); d[pos] = (psi.mean(1) - A) / pos.sum(); d[neg] = (psi.mean(0) - A) / neg.sum(); return A, d
def viid(yy, d):
    n1, n0 = (yy == 1).sum(), (yy == 0).sum(); return (d[yy == 1]**2).sum()*n1/(n1-1) + (d[yy == 0]**2).sum()*n0/(n0-1)
out = {}
# (a) noise budget for one AUC
nb = {}
for k in ['RF', 'XGB', 'PureLSTM', 'LSTM_QNN', 'LSTM_QTL']:
    aucs = np.array([auc(y, s) for s in P[k]]); ses = [viid(y, comps(y, s)[1])**.5 for s in P[k]]
    blk = [np.std([auc(y[ix], s[ix]) for ix in BOOT[:500]], ddof=1) for s in P[k][:5]]
    sd = aucs.std(ddof=1); se = float(np.mean(ses)); seb = float(np.mean(blk))
    nb[k] = dict(mean=float(aucs.mean()), sd=float(sd), se_iid=se, se_block=seb, share_iid=float(sd**2/(sd**2+se**2)),
                 share_block=float(sd**2/(sd**2+seb**2)))
out['noise_budget'] = nb
# (b) Brier raw, ECE, delta BS with year block CI; Platt on validation
def ece(yy, p, bins=10):
    e = np.clip((p * bins).astype(int), 0, bins - 1); return float(sum(abs(yy[e == b].mean() - p[e == b].mean()) * (e == b).mean() for b in range(bins) if (e == b).any()))
def bs(yy, p): return float(((p - yy)**2).mean())
bsc = bs(y, clim)
br = {'CLIM': dict(auc=float(auc(y, clim)), brier=bsc, ece=ece(y, clim))}
for k in ['RF', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN', 'LSTM_QTL']:
    b = np.array([bs(y, s) for s in P[k]]); bss = 1 - b / bsc
    dm = ((P[k] - y)**2).mean(0) - (clim - y)**2
    ci = np.quantile([dm[ix].mean() for ix in BOOT], [.025, .975]).tolist()
    br[k] = dict(auc=float(np.mean([auc(y, s) for s in P[k]])), brier=float(b.mean()), brier_sd=float(b.std(ddof=1)),
                 bss=float(bss.mean()), bss_min=float(bss.min()), bss_max=float(bss.max()), dbs=float(dm.mean()), dbs_ci=ci,
                 ece=float(np.mean([ece(y, s) for s in P[k]])))
out['brier_raw'] = br
def platt(pv, yv, pt):
    lg = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
    m = LogisticRegression(C=1e6).fit(lg(pv)[:, None], yv); return m.predict_proba(lg(pt)[:, None])[:, 1]
def recal_table(z, label):
    yv, yt = z['y_va'].astype(int), z['y_te'].astype(int)
    cr = platt(z['clim_val'], yv, z['clim_te']); bsc_raw, bsc_rec = bs(yt, z['clim_te']), bs(yt, cr)
    t = {'CLIM': dict(auc=float(auc(yt, z['clim_te'])), brier_raw=bsc_raw, brier_platt=bsc_rec, ece_raw=ece(yt, z['clim_te']), ece_platt=ece(yt, cr))}
    for k in ['RF_2273', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN']:
        pv, pt = z[k + '_val'], z[k + '_probs']
        rc = np.array([platt(pv[s], yv, pt[s]) for s in range(len(pt))])
        braw = np.array([bs(yt, p) for p in pt]); bpl = np.array([bs(yt, p) for p in rc])
        s_raw, s_rec = 1 - bpl / bsc_raw, 1 - bpl / bsc_rec
        dm = ((rc - yt)**2).mean(0) - (cr - yt)**2
        t[k] = dict(auc=float(np.mean([auc(yt, p) for p in pt])), auc_sd=float(np.std([auc(yt, p) for p in pt], ddof=1)),
                    brier_raw=float(braw.mean()), brier_platt=float(bpl.mean()),
                    bss_platt_vs_raw=float(s_raw.mean()), n_pos_raw=int((s_raw > 0).sum()), range_raw=[float(s_raw.min()), float(s_raw.max())],
                    bss_platt_vs_recal=float(s_rec.mean()), n_pos_recal=int((s_rec > 0).sum()), range_recal=[float(s_rec.min()), float(s_rec.max())],
                    dbs_vs_recal_ci=np.quantile([dm[ix].mean() for ix in BOOT], [.025, .975]).tolist(),
                    ece_raw=float(np.mean([ece(yt, p) for p in pt])), ece_platt=float(np.mean([ece(yt, p) for p in rc])))
    return t
out['recal_orig'] = recal_table(e28, 'orig'); out['recal_stationmonth'] = recal_table(sm, 'stationmonth')
# station-month paired
ysm = sm['y_te'].astype(int)
def paired(yy, pa, pb):
    ra = [comps(yy, s) for s in pa]; rb = [comps(yy, s) for s in pb]
    d = np.mean([t[1] for t in ra], 0) - np.mean([t[1] for t in rb], 0); g = np.mean([t[0] for t in ra]) - np.mean([t[0] for t in rb])
    se = viid(yy, d)**.5; return dict(gap=float(g), p=float(2*norm.sf(abs(g)/se)), se=float(se))
out['stationmonth_pairs'] = {f'{a}-{b}': paired(ysm, sm[a + '_probs'], sm[b + '_probs']) for a, b in
                             [('RF_2273', 'LSTM_QNN'), ('CTRL_LINEAR4D', 'LSTM_QNN'), ('RF_2273', 'XGB'), ('PureLSTM', 'LSTM_QNN')]}
out['stationmonth_test_rate'] = float(ysm.mean())
# (c) within-station AUC
def within(p):
    num = den = 0.0; per = {}
    for s in np.unique(st):
        m = st == s; a = auc(y[m], p[m]); w = (y[m] == 1).sum() * (y[m] == 0).sum(); per[s] = float(a); num += a * w; den += w
    return num / den, per
ws = {'CLIM': dict(zip(['within', 'per'], within(clim)))}
ws['CLIM']['pooled'] = float(auc(y, clim))
for k in ['RF', 'XGB', 'PureLSTM', 'CTRL_LINEAR4D', 'LSTM_QNN', 'LSTM_QTL']:
    r = [within(s) for s in P[k]]
    ws[k] = dict(pooled=float(np.mean([auc(y, s) for s in P[k]])), within=float(np.mean([t[0] for t in r])), within_sd=float(np.std([t[0] for t in r], ddof=1)),
                 per={s: float(np.mean([t[1][s] for t in r])) for s in r[0][1]})
def within_gap_ci(pa, pb):
    gs = []
    for ix in BOOT[:1000]:
        def w(p):
            num = den = 0.0
            for s in np.unique(st[ix]):
                m = st[ix] == s; yy = y[ix][m]
                if yy.min() == yy.max(): continue
                ww = (yy == 1).sum() * (yy == 0).sum(); num += auc(yy, p[ix][m]) * ww; den += ww
            return num / den
        ga = np.mean([w(s) for s in (pa if pa.ndim == 2 else pa[None])[:5]]); gb = np.mean([w(s) for s in (pb if pb.ndim == 2 else pb[None])[:5]])
        gs.append(ga - gb)
    return np.quantile(gs, [.025, .975]).tolist()
ws['RF_minus_CLIM'] = dict(gap=ws['RF']['within'] - ws['CLIM']['within'], ci=within_gap_ci(P['RF'], clim))
ws['QNN_minus_CLIM'] = dict(gap=ws['LSTM_QNN']['within'] - ws['CLIM']['within'], ci=within_gap_ci(P['LSTM_QNN'], clim))
ws['RF_minus_QNN'] = dict(gap=ws['RF']['within'] - ws['LSTM_QNN']['within'], ci=within_gap_ci(P['RF'], P['LSTM_QNN']))
out['within_station'] = ws
out['station_counts'] = {s: dict(n=int((st == s).sum()), pos=int(y[st == s].sum())) for s in np.unique(st)}
# (e) resolution curve (DeLong MDE, single seed and seed means) and jackknife
def mde_pair(idx, pa, pb, single):
    yy = y[idx]
    if single:
        return np.mean([2.8*viid(yy, comps(yy, pa[s][idx])[1] - comps(yy, pb[s][idx])[1])**.5 for s in range(0, 15, 3)])
    da = np.mean([comps(yy, s[idx])[1] for s in pa], 0); db = np.mean([comps(yy, s[idx])[1] for s in pb], 0)
    return 2.8 * viid(yy, da - db)**.5
def hm(n, prev=0.406, A=0.68, r=0.75):
    n1, n0 = n*prev, n*(1-prev); q1, q2 = A/(2-A), 2*A*A/(1+A)
    se = ((A*(1-A) + (n1-1)*(q1-A*A) + (n0-1)*(q2-A*A)) / (n1*n0))**.5; return 2.8 * se * (2*(1-r))**.5
curve = {}
for k in [5, 8, 10, 13, 16, 19]:
    vals = {'single_RF_LSTM': [], 'single_RF_QNN': [], 'mean_RF_QNN': []}; Ns = []
    for _ in range(60 if k < 19 else 1):
        ys = rng.choice(G, k, replace=False) if k < 19 else G
        idx = np.where(np.isin(yr, ys))[0]; Ns.append(len(idx))
        vals['single_RF_LSTM'].append(mde_pair(idx, P['RF'], P['PureLSTM'], True))
        vals['single_RF_QNN'].append(mde_pair(idx, P['RF'], P['LSTM_QNN'], True))
        vals['mean_RF_QNN'].append(mde_pair(idx, P['RF'], P['LSTM_QNN'], False))
    curve[k] = dict(N=int(np.median(Ns)), HM=float(hm(np.median(Ns))), **{a: float(np.median(b)) for a, b in vals.items()})
out['resolution_curve'] = curve
def jack(pa, pb):
    g = []
    for u in G:
        m = yr != u; g.append(np.mean([auc(y[m], s[m]) for s in pa]) - np.mean([auc(y[m], s[m]) for s in pb]))
    g = np.array(g); K = len(G); return float(((K-1)/K*((g-g.mean())**2).sum())**.5)
out['jackknife'] = {'RF-LSTM_QNN': jack(P['RF'], P['LSTM_QNN']), 'RF-PureLSTM': jack(P['RF'], P['PureLSTM'])}
# (f) single-seed verdicts: share of 15x15 pairings with DeLong p<0.05 (RF>B)
ss = {}
for b in ['XGB', 'PureLSTM', 'LSTM_QNN']:
    c = 0; ses = []
    for i in range(15):
        Ai, di = comps(y, P['RF'][i])
        for j in range(15):
            Bj, dj = comps(y, P[b][j]); se = viid(y, di - dj)**.5; ses.append(se)
            c += (Ai - Bj) / se > 1.96
    ss[b] = dict(share_sig=c/225, se_mean=float(np.mean(ses)), mde=float(2.8*np.mean(ses)))
out['single_seed'] = ss
json.dump(out, open(R + '/local_exp43_v2_tables.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:200]); print('done')
