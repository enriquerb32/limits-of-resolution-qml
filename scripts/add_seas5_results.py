"""Writes the exp33 (SEAS5) results into main.tex and supp.tex. Run once, from the paper folder."""
import re, sys
m = open('main.tex', encoding='utf-8').read()
if 'SEAS5 $-$ climatology & $+0.0495$' in m:
    sys.exit('already applied')

def rep(s, old, new):
    assert s.count(old) == 1, f'not unique/not found: {old[:80]}'
    return s.replace(old, new)

# ---- table rows (from scripts/fill_seas5_table.py, label orig, lead 1)
ROWS = r"""SEAS5 $-$ climatology & $+0.0495$ & 0.015 & $[+0.011,+0.081]$ & $[+0.013,+0.081]$ & 0.057 / 0.050 \\
SEAS5 (lead 2) $-$ climatology & $-0.0145$ & 0.266 & $[-0.044,+0.014]$ & $[-0.043,+0.012]$ & 0.036 / 0.041 \\
Random Forest $-$ SEAS5 & $-0.0472$ & 0.032 & $[-0.073,-0.016]$ & $[-0.074,-0.018]$ & 0.062 / 0.040 \\
LSTM+QNN $-$ SEAS5 & $-0.0664$ & 0.002 & $[-0.094,-0.031]$ & $[-0.094,-0.032]$ & 0.059 / 0.045 \\
\midrule
Blend RF $-$ blend climatology & $+0.0045$ & 0.297 & $[-0.002,+0.011]$ & $[-0.002,+0.011]$ & 0.012 / 0.009 \\
Blend XGBoost $-$ blend climatology & $+0.0016$ & 0.758 & $[-0.010,+0.012]$ & $[-0.009,+0.012]$ & 0.014 / 0.016 \\
Blend LSTM $-$ blend climatology & $+0.0035$ & 0.441 & $[-0.005,+0.012]$ & $[-0.005,+0.012]$ & 0.013 / 0.012 \\
Blend Linear-4D $-$ blend climatology & $+0.0009$ & 0.877 & $[-0.011,+0.013]$ & $[-0.011,+0.013]$ & 0.017 / 0.017 \\
Blend LSTM+QNN $-$ blend climatology & $+0.0012$ & 0.788 & $[-0.007,+0.010]$ & $[-0.008,+0.011]$ & 0.012 / 0.012 \\
Blend LSTM+QNN $-$ blend Linear-4D & $+0.0003$ & 0.935 & $[-0.006,+0.007]$ & $[-0.008,+0.008]$ & 0.010 / 0.010 \\
\midrule
\multicolumn{6}{@{}l}{Brier skill (lead 1): SEAS5 $+0.062$; blend climatology $+0.066$; blend RF $+0.073$; blend LSTM+QNN $+0.065$ (year-block intervals in \Cref{S-tab:seas5_full})} \\"""
a = m.index(r"SEAS5 $-$ climatology & \tbd{}")
b = m.index(r"(year-block intervals in \Cref{S-app:seas5})} \\", a) + len(r"(year-block intervals in \Cref{S-app:seas5})} \\")
m = m[:a] + ROWS + m[b:]

# ---- interpretation paragraph
a = m.index(r"SEAS5 scores AUC \tbd{0.xxx} at lead 1")
b = m.index("\n", a)
PARA = r"""At lead 1 SEAS5 scores AUC 0.730, against 0.681 for the climatology and 0.683 for Random Forest, the best statistical model (\Cref{tab:seas5}). Its gain over the climatology, $+0.050$, is resolved by every procedure (DeLong $p=0.015$, year-block $[+0.011,+0.081]$, cluster-robust interval excluding zero for all 12 start months), and Random Forest and \texttt{LSTM+QNN} lie 0.047 and 0.066 below it; with the station$\times$month label, which removes the base rates, SEAS5 scores 0.624 against 0.50--0.52 for every statistical model ($+0.120$ over the climatology, year-block $[+0.053,+0.193]$; \Cref{S-tab:seas5_full}). The station predictors add nothing resolvable to it: combined with SEAS5, every statistical model lies within $+0.001$ to $+0.005$ of the combination of the climatology with SEAS5, with MDEs of 0.012--0.017, and within $-0.008$ to $-0.001$ (MDEs 0.005--0.015) with the station$\times$month label, whereas SEAS5 adds 0.051 to Random Forest and 0.067 to \texttt{LSTM+QNN}. The circuit question is unchanged by the stronger reference: the combinations with \texttt{LSTM+QNN} and with the Linear-4D control differ by $+0.0003$ ($[-0.006,+0.007]$, MDE 0.010). At lead 2, which is available before the target month begins, the advantage disappears: SEAS5 is not distinguished from the climatology ($-0.014$, $[-0.044,+0.014]$; station$\times$month label $+0.011$, $[-0.051,+0.087]$), its Brier skill is negative ($-0.019$), and no combination improves on the climatology. The discrimination at lead 1 therefore comes from the first part of the target month, which a forecast started on its first day can resolve, and not from monthly predictability beyond it. For the benchmark, the result has two sides. The protocol resolves a real difference of 0.05 AUC between an operational system and the climatology on the same 528 rows, so the null results among the statistical models are not a failure of the procedures; and an operational dynamical forecast, not the climatology, is the reference that the statistical models of this panel, classical or quantum, would have to beat. These comparisons are secondary; the bias correction (quantiles of 1981--1996 hindcasts at the training base rate) was specified before the test data were analysed, and an uncorrected count of members above 28.1\,mm scores slightly higher at lead 1 (0.742; \Cref{S-tab:seas5_full})."""
m = m[:a] + PARA + m[b:]

# ---- abstract sentence (+ two-word trim elsewhere to stay near 250 words)
m = rep(m, r"and against the ECMWF SEAS5 seasonal forecast \tbd{no model} added resolvable discrimination.",
        r"and none added resolvable information to the ECMWF SEAS5 seasonal forecast, which exceeded the climatology by 0.050 AUC at lead 1.")
m = rep(m, r"a within-year structure that permutation tests confirm but no simulated mechanism reproduces.",
        r"a within-year structure that no simulated mechanism reproduces.")

# ---- contribution 4
m = rep(m, r"compared against climatological references on discrimination,",
        r"compared against climatological and operational dynamical (ECMWF SEAS5) references on discrimination,")

# ---- conclusions
m = rep(m, r"With a threshold specific to each station and month, which removes the base rates, no model discriminates (AUC 0.49--0.52).",
        r"With a threshold specific to each station and month, which removes the base rates, no model discriminates (AUC 0.49--0.52). The ECMWF SEAS5 forecast started on the first day of the target month exceeds the climatology by 0.050 AUC under both procedures and discriminates with the station$\times$month label (0.624), and no statistical model adds resolvable information to it; at lead 2 its advantage disappears.")
open('main.tex', 'w', encoding='utf-8').write(m)

# ---- supplement: replace the placeholder with a full table
s = open('supp.tex', encoding='utf-8').read()
a = s.index(r"\emph{Further results.} \tbd{")
b = s.index("\n", a)
TABLE = r"""\emph{Further results.} \Cref{tab:seas5_full} gives every SEAS5 score at both leads and under both labels. The bias-corrected probability (QM) was the pre-specified score; the uncorrected member count (RAW) scores higher at lead 1 with the pooled label, because the model's own seasonal cycle of monthly totals already separates the wet months, while the anomaly-only score (EM) carries no base-rate information and is below the climatology with the pooled label but equals QM with the station$\times$month label. At lead 2 the combination of Random Forest with SEAS5 scores 0.012 above that of \texttt{LSTM+QNN} with a cluster/iid ratio of 0.56 (year-block $[+0.002,+0.020]$, DeLong $p=0.12$, 6 of 12 start months), the same within-year structure that \Cref{M-sec:budget} reports for Random Forest against the hybrids.

\begin{table}[htbp]
\centering
\caption{SEAS5 scores and combinations (\protect\path{results/local_exp33_seas5_orig.json}, \protect\path{results/local_exp33_seas5_stationmonth.json}; 528 test rows). AUC: seed mean where 15 seeds. BSS: Brier skill against the climatology recalibrated on the validation partition (Platt maps for the SEAS5 scores, logistic combinations fitted on validation); $\Delta$BS: year-block 95\,\% interval of the Brier difference against that climatology (negative = better).}
\label{tab:seas5_full}
{\small
\adjustbox{max width=\textwidth}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
& \multicolumn{3}{c}{Pooled P60 label} & \multicolumn{3}{c}{Station$\times$month P60 label} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Score & AUC & BSS & $\Delta$BS 95\,\% CI & AUC & BSS & $\Delta$BS 95\,\% CI \\
\midrule
Climatology & 0.681 & 0 & -- & 0.504 & 0 & -- \\
SEAS5 QM, lead 1 & 0.730 & $+0.062$ & $[-0.023,-0.002]$ & 0.624 & $+0.043$ & $[-0.021,+0.001]$ \\
SEAS5 RAW, lead 1 & 0.742 & $+0.089$ & $[-0.028,-0.010]$ & -- & -- & -- \\
SEAS5 EM, lead 1 & 0.622 & $-0.059$ & $[-0.002,+0.028]$ & 0.622 & $+0.031$ & $[-0.016,+0.001]$ \\
SEAS5 QM, lead 2 & 0.666 & $-0.019$ & $[-0.002,+0.011]$ & 0.514 & $-0.005$ & $[-0.005,+0.008]$ \\
SEAS5 RAW, lead 2 & 0.673 & $-0.011$ & $[-0.007,+0.013]$ & -- & -- & -- \\
SEAS5 EM, lead 2 & 0.564 & $-0.102$ & $[+0.009,+0.035]$ & 0.525 & $+0.003$ & $[-0.003,+0.002]$ \\
\midrule
Blend climatology, lead 1 & 0.730 & $+0.066$ & $[-0.022,-0.007]$ & 0.625 & $+0.043$ & $[-0.022,+0.001]$ \\
Blend RF, lead 1 & 0.734 & $+0.073$ & $[-0.024,-0.006]$ & 0.624 & $+0.043$ & $[-0.022,+0.001]$ \\
Blend XGBoost, lead 1 & 0.731 & $+0.066$ & $[-0.025,-0.003]$ & 0.624 & $+0.043$ & $[-0.021,+0.001]$ \\
Blend LSTM, lead 1 & 0.733 & $+0.069$ & $[-0.023,-0.007]$ & 0.617 & $+0.034$ & $[-0.020,+0.003]$ \\
Blend Linear-4D, lead 1 & 0.731 & $+0.063$ & $[-0.022,-0.004]$ & 0.623 & $+0.041$ & $[-0.021,+0.002]$ \\
Blend LSTM+QNN, lead 1 & 0.731 & $+0.065$ & $[-0.022,-0.006]$ & 0.620 & $+0.039$ & $[-0.020,+0.002]$ \\
Blend climatology, lead 2 & 0.682 & $-0.000$ & $[-0.002,+0.002]$ & 0.515 & $-0.005$ & $[-0.005,+0.007]$ \\
Blend RF, lead 2 & 0.688 & $+0.003$ & $[-0.006,+0.005]$ & 0.515 & $-0.007$ & $[-0.005,+0.008]$ \\
Blend LSTM+QNN, lead 2 & 0.676 & $-0.013$ & $[-0.002,+0.008]$ & 0.511 & $-0.008$ & $[-0.004,+0.007]$ \\
\bottomrule
\end{tabular}}}
\end{table}"""
s = s[:a] + TABLE + s[b:]
s = s.replace(r"start dates January 1981 to \tbd{May 2026}", r"start dates January 1981 to November 2026 (test targets to May 2026)")
open('supp.tex', 'w', encoding='utf-8').write(s)
print('SEAS5 results written into main.tex and supp.tex')
