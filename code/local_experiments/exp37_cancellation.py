"""exp37: why is the year-cluster/iid SE ratio of RF-QNN ~0.55?
Decomposes the year-cluster variance of the paired DeLong influence into within-station
and cross-station terms, and tests station-demeaned influences. No retraining."""
import numpy as np, pandas as pd, json, sys, itertools
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--probs',default='results/json/results_hybrids_trainonly_fixet0_p60_28.1_probs.npz')
ap.add_argument('--meta',default='results/test_meta.csv'); ap.add_argument('--out',default='results/local_exp37_cancellation.json')
ap.add_argument('--pairs',default=None,help='name=keyA:keyB,... (default: panel-1 keys)'); args=ap.parse_args()
R='results/'
z=np.load(args.probs)
m=pd.read_csv(args.meta); y=z['y_te'].astype(int)
assert (m.y_te.values==y).all()
yr=pd.to_datetime(m.fecha).dt.year.values; mo=pd.to_datetime(m.fecha).dt.month.values; st=m.ciudad.values
pairs={'RF-QNN':('RF_2273_probs','LSTM_QNN_probs'),'RF-LSTM':('RF_2273_probs','PureLSTM_probs'),
       'LSTM-QNN':('PureLSTM_probs','LSTM_QNN_probs'),'RF-Lin4D':('RF_2273_probs','CTRL_LINEAR4D_probs')}
def infl(p):
    pos=p[y==1]; neg=p[y==0]
    psi=(pos[:,None]>neg[None,:])+0.5*(pos[:,None]==neg[None,:])
    a=psi.mean(); out=np.empty(len(p))
    out[y==1]=(psi.mean(1)-a)/len(pos); out[y==0]=(psi.mean(0)-a)/len(neg)
    return a,out
def var_iid(d):  # DeLong variance from influences: sum over rows of (n_class/(n_class-1)) d^2
    n1=(y==1).sum(); n0=(y==0).sum()
    return (d[y==1]**2).sum()*n1/(n1-1)+(d[y==0]**2).sum()*n0/(n0-1)
def var_cl(d,g):
    G=np.unique(g); s=np.array([d[g==k].sum() for k in G]); return len(G)/(len(G)-1)*(s**2).sum()
def demean(d,keys):
    df=pd.DataFrame({'d':d,'y':y}); 
    for k,v in keys.items(): df[k]=v
    return d-df.groupby(list(keys)+['y']).d.transform('mean').values
if args.pairs: pairs={kv.split('=')[0]:tuple(kv.split('=')[1].split(':')) for kv in args.pairs.split(',')}
res={}
stations=sorted(set(st))
for name,(a,b) in pairs.items():
    IA=[infl(q) for q in z[a]]; IB=[infl(q) for q in z[b]]
    A=np.mean([u[0] for u in IA]); B=np.mean([u[0] for u in IB])
    d=np.mean([u[1] for u in IA],0)-np.mean([u[1] for u in IB],0)  # seed-averaged DeLong components
    vi=var_iid(d); vc=var_cl(d,yr); r={'gap':A-B,'se_iid':vi**.5,'se_year':vc**.5,'ratio_year':(vc/vi)**.5,
       'ratio_station_year':(var_cl(d,np.char.add(st.astype(str),yr.astype(str)))/vi)**.5}
    # decomposition of sum_years (sum_s S_sy)^2 = sum_s S_sy^2 + sum_{s!=t} S_sy S_ty
    Y=np.unique(yr); S=np.array([[d[(yr==k)&(st==s)].sum() for s in stations] for k in Y])
    within=(S**2).sum(); cross=((S.sum(1))**2).sum()-within
    r['station_totals']=dict(zip(stations,S.sum(0).round(5).tolist()))
    r['within_station_term']=within; r['cross_station_term']=cross; r['cross_share']=cross/within
    C=np.corrcoef(S.T); r['mean_cross_corr_yearly_sums']=float(C[np.triu_indices(len(stations),1)].mean())
    for lab,keys in {'station':{'s':st},'station_month':{'s':st,'m':mo}}.items():
        dd=demean(d,keys); r[f'ratio_year_demeaned_{lab}']=(var_cl(dd,yr)/var_iid(dd))**.5
    # break regional co-occurrence: permute year labels independently within station
    rng=np.random.default_rng(0); rr=[]; rd=[]
    for _ in range(2000):
        g=yr.copy()
        for s in stations:
            ix=np.where(st==s)[0]; u=np.unique(yr[ix]); mp=dict(zip(u,rng.permutation(u))); g[ix]=[mp[v] for v in yr[ix]]
        rr.append((var_cl(d,g)/vi)**.5)
        if _<300:  # null for the station-month demeaned ratio (demeaning itself induces dependence)
            dd=demean(d,{'s':st,'m':mo}); rd.append((var_cl(dd,g)/var_iid(dd))**.5)
    r['perm_p_lower']=float((np.array(rr)<=r['ratio_year']).mean()); r['ratio_year_perm_within_station_median']=float(np.median(rr)); r['ratio_year_perm_q05_q95']=np.quantile(rr,[.05,.95]).tolist()
    r['ratio_demeaned_station_month_perm_median']=float(np.median(rd))
    res[name]=r
json.dump(res,open(args.out,'w'),indent=1,default=float)
for k,v in res.items():
    print(k,{a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items()})
