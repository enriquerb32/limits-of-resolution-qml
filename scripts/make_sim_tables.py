"""
scripts/make_sim_tables.py - Build LaTeX table rows for Table sim and Table sim_all from simulation JSONs.

Usage:
  python scripts/make_sim_tables.py [results_dir] [out_dir]
"""
import json, sys, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "tables")
os.makedirs(OUT, exist_ok=True)

sc = [
    ('B', 'simB_rerun.json', 'shared year effects ($\\tau_\\pi=\\tau_s=0.3$)'),
    ('C', 'simC_diff.json', '+ scorer-specific year effects ($\\tau_m=0.3$)'),
    ('D', 'simD_unequal.json', 'C with unequal year sizes'),
    ('E', 'simE_strong.json', 'all $\\tau=0.6$')
]

main = []
supp = []

for k, f, desc in sc:
    fp = os.path.join(D, f)
    if not os.path.exists(fp):
        continue
    d = json.load(open(fp, encoding="utf-8"))
    R = d['rows']
    r0 = R[0]
    rl = R[-1]
    main.append(
        f"{k}: {desc} & {r0['rejection_delong']:.3f} & {r0['rejection_block']:.3f} & "
        f"{r0['coverage_delong']:.3f} & {r0['coverage_block']:.3f} & {r0['sd_delta_hat']:.4f} & "
        f"{r0['mean_se_delong']:.4f} / {r0['mean_se_block']:.4f} & "
        f"{rl['rejection_delong']:.2f} / {rl['rejection_block']:.2f} ({rl['delta_true']:.3f}) \\\\"
    )
    supp.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{Scenario {k}: {desc}}}}} \\\\")
    for r in R:
        supp.append(
            f"{r['delta_true']:.4f} ({r.get('delta_true_mc_se', 0.0001):.4f}) & "
            f"{r['rejection_delong']:.3f} ({r.get('n_reject_delong', int(round(r['rejection_delong']*r['reps'])))}) & "
            f"{r['rejection_block']:.3f} ({r.get('n_reject_block', int(round(r['rejection_block']*r['reps'])))}) & "
            f"{r['coverage_delong']:.3f} & {r['coverage_block']:.3f} & {r['mean_delta_hat']:.4f} & "
            f"{r['sd_delta_hat']:.4f} & {r['mean_se_delong']:.4f} & {r['mean_se_block']:.4f} \\\\"
        )
    supp.append("\\midrule")

# earlier runs
fp_rho = os.path.join(D, "sim_rho09.json")
if os.path.exists(fp_rho):
    d = json.load(open(fp_rho, encoding="utf-8"))
    supp.append("\\multicolumn{9}{l}{\\emph{Scenario B, first run (seed 0, 400 replications, $B=300$)}} \\\\")
    for r in d['rows']:
        supp.append(
            f"{r['delta_true']:.4f} (--) & {r['rejection_delong']:.3f} ({int(round(r['rejection_delong']*r['reps']))}) & "
            f"{r['rejection_block']:.3f} ({int(round(r['rejection_block']*r['reps']))}) & {r['coverage_delong']:.3f} & "
            f"{r['coverage_block']:.3f} & {r['mean_delta_hat']:.4f} & {r['sd_delta_hat']:.4f} & "
            f"{r['mean_se_delong']:.4f} & {r['mean_se_block']:.4f} \\\\"
        )
    supp.append("\\midrule")

A = [
    (0.0008, .05, .07, .95, .94, .0165, .0172, .0166),
    (0.0092, .09, .12, .93, .90, .0179, .0171, .0167),
    (0.0182, .19, .23, .93, .91, .0179, .0171, .0165),
    (0.0281, .39, .44, .95, .94, .0166, .0170, .0165),
    (0.0389, .65, .67, .95, .94, .0169, .0170, .0164),
    (0.0483, .80, .80, .95, .92, .0171, .0170, .0166)
]
supp.append("\\multicolumn{9}{l}{\\emph{Scenario A: $\\rho=0.75$, shared year effects (500 replications, $B=500$; rates rounded to 2 decimals)}} \\\\")
for t in A:
    supp.append(f"{t[0]:.4f} (--) & {t[1]:.2f} & {t[2]:.2f} & {t[3]:.2f} & {t[4]:.2f} & -- & {t[5]:.4f} & {t[6]:.4f} & {t[7]:.4f} \\\\")

open(os.path.join(OUT, 'sim_main_rows.tex'), 'w', encoding='utf-8').write("\n".join(main) + "\n")
open(os.path.join(OUT, 'sim_supp_rows.tex'), 'w', encoding='utf-8').write("\n".join(supp) + "\n")
print(f"Generated sim table rows in {OUT}")
