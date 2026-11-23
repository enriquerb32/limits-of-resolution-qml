# Pre-analysis plan for a small benchmark with clustered test cases

Fill this in **before any test prediction is seen**, commit it, and deposit it where the timestamp is not under your
control (Zenodo, OSF). Every item is a decision that the reporting protocol of the paper (Table 10) treats as
substantive. The inferential items can be run with `python -m qml_resolution.report`.

Date and deposit identifier: ______

## 1. Estimand
- Metric (e.g. ROC AUC, Brier score):
- Single run or mean over S seeds (S = ):
- Population of test cases (e.g. new years at the same sites):
- Conditioned on (training data, fitted models):

## 2. Statements and the interval that decides each
| Comparison | About the fitted models or about a procedure/architecture? | Deciding interval |
|---|---|---|
| | | primary cluster interval / cluster × seed interval as well |

## 3. Clusters and primary procedure
- Cluster unit and number of clusters:
- Primary procedure (e.g. CR2 with Bell–McCaffrey degrees of freedom):
- Sensitivity analyses (e.g. clusters of consecutive periods, wild cluster bootstrap):
- Floor: the cluster/iid ratio below which the standard error is floored at the iid one (e.g. the 5th percentile of
  the ratio simulated on the test design with independent cases), or the mechanism expected to produce low ratios:

## 4. Calibration of the primary procedure
- Simulated mechanisms (dependence within and between clusters, serial correlation):
- True differences, replications, coverage band that counts as calibrated:

## 5. Families and multiplicity
- Primary families and correction (e.g. Holm):
- Everything else is secondary and unadjusted.

## 6. Resolution
- MDE of each primary comparison from test size, prevalence, assumed AUC and correlation
  (`qml_resolution.paired_mde_hanley_mcneil`), times the design effect 1 + (m − 1)ρ for clusters of m rows:
- Expected uncertainty of the MDE (few clusters):

## 7. Equivalence
- Smallest difference of interest and equivalence margin (chosen by relevance, not from the MDE):

## 8. References and controls
- Base-rate reference (e.g. climatology), operational reference:
- For quantum models: classical controls at the circuit interface (matched linear map, Fourier surrogates with the
  circuit's spectrum, frozen circuit) and planted circuit signals (strengths, null):

## 9. Robustness rules
- Checks that turn a resolved difference into an exploratory one (omitted clusters, consecutive-period clusters,
  causal refits, ...):

## 10. Predictions
- Expected resolution and verdicts:
