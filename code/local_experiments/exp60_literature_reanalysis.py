#!/usr/bin/env python3
"""
exp60_literature_reanalysis.py - resolution of published QML-versus-classical AUC comparisons (no data, no training)
==================================================================================================================
For published binary-classification studies that compare a quantum
model with a classical baseline by ROC AUC on a held-out test set and print the test size, the script gives, for the
reported difference (best quantum minus best classical model in the setting the authors emphasise):
  - an approximate 95 % interval, delta +/- 1.96 SE, with SE the paired Hanley-McNeil standard error
    SE_HM(AUC, n1, n0) * sqrt(2 (1 - rho)), test cases treated as independent;
  - the smallest equivalence margin those data would accept, max(|lo90|, |hi90|) of the 90 % interval (the TOST
    bound of Table S13 of the paper);
  - the 80 %-power MDE of the paired difference (Table S17), as a measure of design sensitivity only.
The search that produced the candidate studies is described in SEARCH and in the supplement.

Assumptions, stated per row: when the class balance of the test set is not printed it is taken as 0.5; the
correlation between the AUCs of the two models is unknown and is set to 0.75 (the conservative value of Section
4.3, which gives the smaller MDE) and 0.5. When the reported values are means over several independent test sets,
the MDE is given for the pooled test size (most favourable to the study) and for one test set. Clustering of test
cases, which none of these studies models, can only be added on top. The values were transcribed from the arXiv
versions named in SOURCES on 21 November 2026; studies screened but not usable for an MDE are listed in SCREENED.

  python code/local_experiments/exp60_literature_reanalysis.py
Output: results/local_exp60_literature_reanalysis.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from qml_resolution import hanley_mcneil_se, paired_mde_hanley_mcneil  # noqa: E402

# n_set: test cases per test set; n_sets: independent test sets averaged; prev: positive fraction (None: not printed)
STUDIES = [
    dict(key='wu2021_tth', study='Wu et al. (2021)', arxiv='2012.11560', task='ttH(gamma gamma) vs background',
         quantum='VQC (10 qubits)', classical='SVM / BDT', auc_q=0.81, auc_c=0.83, n_set=100, n_sets=10, prev=None,
         claim='comparable (within the margin of error)', where='Results, Fig. 3a'),
    dict(key='wu2021_hmm', study='Wu et al. (2021)', arxiv='2012.11560', task='H(mu mu) vs Z/gamma* background',
         quantum='VQC (10 qubits)', classical='SVM', auc_q=0.83, auc_c=0.82, n_set=100, n_sets=10, prev=None,
         claim='performs similarly', where='Results, Fig. 3b'),
    dict(key='belis2021_qsvm16', study='Belis et al. (2021)', arxiv='2104.07692', task='ttH(bb) vs ttbb',
         quantum='QSVM, 16 features', classical='SVM (RBF)', auc_q=0.66, auc_c=0.65, n_set=720, n_sets=5, prev=None,
         claim='similar or better', where='Table 2a'),
    dict(key='belis2021_qsvm64', study='Belis et al. (2021)', arxiv='2104.07692', task='ttH(bb) vs ttbb',
         quantum='QSVM, 64 features', classical='logistic regression', auc_q=0.68, auc_c=0.68, n_set=720, n_sets=5,
         prev=None, claim='similar or better', where='Table 2b'),
    dict(key='belis2021_vqc', study='Belis et al. (2021)', arxiv='2104.07692', task='ttH(bb) vs ttbb',
         quantum='VQC, 8 features', classical='random forest', auc_q=0.66, auc_c=0.66, n_set=720, n_sets=5, prev=None,
         claim='similar or better', where='Table 3'),
    dict(key='grossi2022', study='Grossi et al. (2022)', arxiv='2208.07963', task='card fraud vs genuine',
         quantum='QSVM', classical='XGBoost', auc_q=0.81, auc_c=0.834, n_set=999, n_sets=1, prev=478 / 999,
         claim='similar', where='Tables I, IV, VI'),
    dict(key='heredge2021', study='Heredge et al. (2021)', arxiv='2103.12257', task='B-meson pairs vs continuum',
         quantum='QSVM (separate-particle circuit, Bloch encoding)', classical='SVM (RBF)', auc_q=0.848, auc_c=0.793, n_set=10000, n_sets=10,
         prev=None, claim='best average AUC', where='Table 1'),
    dict(key='kulkarni2023', study='Kulkarni et al. (2023)', arxiv='2202.10452', task='pneumonia vs normal radiograph',
         quantum='CNN with VQC layer', classical='CNN', auc_q=0.836, auc_c=0.727, n_set=624, n_sets=1, prev=390 / 624,
         claim='statistically significant (test over 30 training runs)', where='Section 3.5, Table 1'),
]
# decimals of the printed quantum and classical AUCs: the rounding of delta is at most 0.5 (10^-dq + 10^-dc)
DECIMALS = {'wu2021_tth': (2, 2), 'wu2021_hmm': (2, 2), 'belis2021_qsvm16': (2, 2), 'belis2021_qsvm64': (2, 2),
            'belis2021_vqc': (2, 2), 'grossi2022': (2, 3), 'heredge2021': (3, 3), 'kulkarni2023': (3, 3)}
# per-model spread of the AUC printed by the study, over its test sets or trials, for one test set:
# (range of the printed values, what the paper says it is)
SPREAD = {'wu2021': ((0.03, 0.06), 'SD over the 10 datasets of 100 test events (Fig. 3)'),
          'belis2021': ((0.01, 0.02), 'printed as +/- over the 5 test sets of 720 samples; definition not stated'),
          'heredge2021': ((0.001 * 10 ** 0.5, 0.001 * 10 ** 0.5),
                          'SD implied by the printed 0.001 = SD / sqrt(10) (Table 1); test sets redrawn at random'),
          'grossi2022': ((0.01, 0.01), 'printed as +/- over the trials; definition not stated')}
# version against which the transcribed values were checked (see the supplement, Search)
CHECKED = {'wu2021': 'arXiv', 'belis2021': 'published', 'heredge2021': 'published', 'grossi2022': 'arXiv (published version confirms the wording and a test set of 1,000 only)',
           'kulkarni2023': 'arXiv'}
MDE_ONLY = [  # test size printed, but no classical AUC at that setting
    dict(key='terashi2020_vqc', study='Terashi et al. (2020)', arxiv='2002.09935', task='SUSY chargino pairs vs WW',
         quantum='VQC (QASM simulator)', auc_q=0.555, n_set=40, n_sets=1, prev=0.5,
         where='Section 3.3, Table 2'),
]
SCREENED = {  # read, but no test size or no quantum-classical AUC difference printed
    'Blance and Spannowsky (2021), arXiv 2010.07335': 'accuracy only; classical baseline shown only as a ROC curve',
    'Alvi et al. (2023), arXiv 2206.08391': 'anomaly detection; AUC only in a figure; "no evidence that QML provides any advantage"',
    'Gianelle et al. (2022), arXiv 2202.13943': 'tagging power; test sets of 40,000 and 290,000 jets; values only in figures',
    'Schuhmacher et al. (2023), arXiv 2301.10787': 'validation AUC; test-set size not printed',
    'Wu et al. (2021b), arXiv 2104.05059': 'AUCs only in figures',
    'Terashi et al. (2020), arXiv 2002.09935 (QCL)': 'QCL AUCs only in figures ("comparable")',
    'Bowles et al. (2024), arXiv 2403.07059': 'accuracy on mostly synthetic data; no uncertainty for the differences',
    'Tudisco et al. (2025), arXiv 2505.20804': 'precision and recall only; test size not printed',
    'El Alami et al. (2024), arXiv 2412.19441': 'no classical baseline',
}
REGRESSION = ['da Silva et al. (2026)', 'Pastori et al. (2026)', 'Fellner et al. (2026)', 'Marchisio et al. (2026)']
RHOS = (0.75, 0.5)


SEARCH = dict(
    date='2026-11-21',
    sources=['the QML studies already cited in the manuscript',
             'four web searches (queries below) and the papers they returned',
             'quantum binary-classification benchmarks in particle physics that compare with BDT/SVM/DNN baselines'],
    queries=['quantum machine learning precipitation rainfall classification hybrid quantum neural network accuracy AUC '
             'compared classical',
             'variational quantum classifier high energy physics event classification AUC test events comparison BDT',
             'quantum support vector machine medical diagnosis dataset AUC outperforms classical test set hybrid quantum '
             'classifier 2023 2024',
             'quantum machine learning fraud detection credit card AUC classical comparison test set quantum kernel',
             'hybrid quantum classical neural network pneumonia chest X-ray AUC test set classical CNN comparison'],
    inclusion='binary classification on real or simulated-physics data; a quantum and a classical model compared on '
              'a held-out test set; test size and test AUC of both printed for the same setting',
)


SUPERIORITY = {'heredge2021', 'kulkarni2023'}  # the two comparisons stated as an advantage of the quantum model


def mde(n, auc, prev, rho):
    return paired_mde_hanley_mcneil(n, auc=auc, prevalence=prev, rho=rho)


def paired_se(n, auc, prev, rho):
    return hanley_mcneil_se(auc, prev * n, (1 - prev) * n) * (2 * (1 - rho)) ** 0.5


def main():
    rows = []
    for s in STUDIES:
        prev = s['prev'] if s['prev'] is not None else 0.5
        auc = (s['auc_q'] + s['auc_c']) / 2
        delta = s['auc_q'] - s['auc_c']
        n_pool = s['n_set'] * s['n_sets']
        r = dict(s, delta=round(delta, 4), prev_used=prev, prev_assumed=s['prev'] is None, n_pooled=n_pool)
        for k in RHOS:
            se = paired_se(n_pool, auc, prev, k)
            r[f'se_{k}'] = se
            r[f'ci95_{k}'] = [delta - 1.959964 * se, delta + 1.959964 * se]
            r[f'smallest_margin_{k}'] = abs(delta) + 1.644854 * se
        r['mde_pooled'] = {str(k): mde(n_pool, auc, prev, k) for k in RHOS}
        r['mde_one_set'] = {str(k): mde(s['n_set'], auc, prev, k) for k in RHOS}
        r['ci95_excludes_zero_0.75'] = r['ci95_0.75'][0] > 0 or r['ci95_0.75'][1] < 0
        dq, dc = DECIMALS[s['key']]
        r['rounding'] = 0.5 * (10.0 ** -dq + 10.0 ** -dc)
        r['smallest_margin_worst_rounding_0.75'] = r['smallest_margin_0.75'] + r['rounding']
        r['hm_se_one_auc_one_set'] = hanley_mcneil_se(auc, prev * s['n_set'], (1 - prev) * s['n_set'])
        r['checked_against'] = CHECKED[s['key'].split('_')[0]]
        rows.append(r)
        print(f"{s['study']:24s} {s['quantum'][:22]:22s} D {delta:+.3f}  CI95 [{r['ci95_0.75'][0]:+.3f},"
              f"{r['ci95_0.75'][1]:+.3f}]  margin {r['smallest_margin_0.75']:.3f} (rho .5 {r['smallest_margin_0.5']:.3f})"
              f"  MDE {r['mde_pooled']['0.75']:.3f}")
    only = [dict(s, mde={str(k): mde(s['n_set'], s['auc_q'], s['prev'], k) for k in RHOS}) for s in MDE_ONLY]
    sim = [r for r in rows if r['key'] not in SUPERIORITY]  # the comparisons stated as similarity
    excl05 = [r['key'] for r in sim if r['ci95_0.5'][0] > 0 or r['ci95_0.5'][1] < 0]
    out = dict(rhos=RHOS, search=SEARCH, contrasts=rows, mde_only=only, screened=SCREENED, regression_only=REGRESSION,
               n_contrasts=len(rows), n_studies=len({r['study'] for r in rows}), n_similarity=len(sim),
               similarity_ci_excludes_zero_rho075=[r['key'] for r in sim if r['ci95_excludes_zero_0.75']],
               similarity_ci_excludes_zero_rho05=excl05,
               superiority_ci_excludes_zero_both=all(
                   (r['ci95_0.5'][0] > 0) and (r['ci95_0.75'][0] > 0) for r in rows if r['key'] in SUPERIORITY),
               margin_range_similarity=[min(r['smallest_margin_0.75'] for r in sim),
                                        max(r['smallest_margin_0.75'] for r in sim)],
               margin_range_similarity_rho05=[min(r['smallest_margin_0.5'] for r in sim),
                                              max(r['smallest_margin_0.5'] for r in sim)])
    out['consistency'] = {}
    for key, (rng, what) in SPREAD.items():
        hm = sorted({round(r['hm_se_one_auc_one_set'], 4) for r in rows if r['key'].startswith(key)})
        out['consistency'][key] = dict(printed_spread=rng, printed_as=what, hm_se_one_auc_one_set=hm)
        print(f'  consistency {key:12s} printed {rng[0]:.4f}-{rng[1]:.4f} ({what}); Hanley-McNeil SE {hm}')
    print(f"similarity claims {len(sim)}: smallest margins {out['margin_range_similarity']} "
          f"(rho .5 {out['margin_range_similarity_rho05']}); CI excludes zero at rho .75: "
          f"{out['similarity_ci_excludes_zero_rho075']}, at rho .5: {excl05}; "
          f"superiority CIs exclude zero: {out['superiority_ci_excludes_zero_both']}")
    path = os.path.join(ROOT, 'results', 'local_exp60_literature_reanalysis.json')
    json.dump(out, open(path, 'w'), indent=2)
    print('wrote', path)


if __name__ == '__main__':
    main()