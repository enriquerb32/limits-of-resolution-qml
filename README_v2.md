# v2 rerun (primary results of the revised paper)

v2 changes the data pipeline and the environment, not the models or the statistics. The v1 archives are kept,
unchanged, in `results_v1/` of the `v2` branch (and in `results/` of `main`).

| Issue (v1) | v2 |
|---|---|
| Daily inner join of five files: a missing pressure or wind day removed that day's rain; 23 % of the test months incomplete; Valencia and San Javier almost absent after 2012 | `code/etl_v2.py`: full calendar per station; the label is the monthly sum of the precipitation file alone, defined when at most `V2_MAX_MISSING` (3) days are missing |
| Missing precipitation and sunshine set to 0; covariates forward/backward filled over any gap | missing stays missing; monthly means over observed days (>= 10); gaps of at most `V2_INTERP_MONTHS` (2) months interpolated; windows without features or lags are dropped, not filled |
| Months with < 10 joined days dropped and the rest concatenated (windows could jump over gaps) | every calendar month kept; windows are calendar-contiguous |
| ET0 with a constant wind (u2 = 2 m/s) | ET0 with the observed wind, u2 = 0.748 u10 (FAO-56 eq. 47) |
| JJA ET0 total summed over whatever summer months were present | JJA total requires the three months |
| Tree baselines with manual hyperparameters; XGBoost without search or early stopping | `exp40_v2_tune.py`: Optuna (TPE, 50 trials, declared spaces) for both; XGBoost stops early on the validation AUC |
| Mixed environments (scikit-learn 1.6.1 on Colab, 1.9.1 locally; XGBoost version not recorded) | one environment, `requirements-v2.lock` (hashes), `results_v2_environment.txt`, fixed thread count |

New in v2: `QTL_TENSOR1` in `exp27_controls.py` (the 81-function tensor basis that spans the R_Y-embedding head
exactly), the angle distribution and a random-weight R^2 reference in `exp34 --part A`, the teacher-circuit positive
control `exp38_teacher_control.py`, and the cancellation diagnostic `exp37_cancellation.py`.

Switch: everything is selected by the environment variable `LOR_DATASET=v2`; without it every script behaves as v1.
File names: the analyses read v1 archive names (`..._fixet0_p60_28.1_...`); in v2 these files hold v2 results
(the v2 P60 is stored in the archive json as `threshold`). Run with `run_v2.ps1`.
