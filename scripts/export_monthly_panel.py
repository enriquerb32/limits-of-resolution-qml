"""
scripts/export_monthly_panel.py - monthly station panel of the v2 pipeline, for redistribution.

The daily AEMET records are not redistributed. This script aggregates the daily file written by code/etl_v2.py
(or by exp32_second_panel.py --step etl) with the v2 monthly aggregation of the pipeline
(code/local_experiments/dataset_v2.agregar_mensual_v2) and writes only monthly values.

Usage (from the repository root):
  python scripts/export_monthly_panel.py --daily code/local_experiments/datos_limpios_multiestacion.csv \
      --out results/panel_v2_monthly.csv                                   # first panel
  python scripts/export_monthly_panel.py --daily <panel2 datos.csv> --out results/panel2/med2v2/panel_monthly.csv

Columns: ciudad, fecha (first day of the month), precip_label (monthly sum, defined when at most 3 days are
missing; the label of the paper is precip_label > P60 of the training labels), dias (days with precipitation),
label_ok, precip (lag value: sum rescaled to the month length, >= 10 valid days), precip_ok, the nine monthly
covariates (pmed, t_med, dp_std, t_rango, t_min, veloc_ms, racha_ms_max, sol_sum, et0_sum; ET0 FAO-56 with the
observed wind), their day counts, feat_ok and win_ok (the month is the target of a valid 12-month window).
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "local_experiments"))
sys.path.insert(1, os.path.join(ROOT, "code"))

COLUMNS = ["ciudad", "precip_label", "dias", "label_ok", "precip", "precip_ok",
           "pmed", "t_med", "dp_std", "t_rango", "t_min", "veloc_ms", "racha_ms_max", "sol_sum", "et0_sum",
           "n_pmed", "n_t", "n_v", "n_r", "n_s", "n_e", "precip_max_dia", "feat_ok", "win_ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True, help="daily file written by etl_v2.py (with precip_valid)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from dataset_v2 import agregar_mensual_v2

    daily = pd.read_csv(a.daily, index_col="fecha", parse_dates=["fecha"])
    agg = agregar_mensual_v2(daily)[COLUMNS]
    agg.index.name = "fecha"
    agg = agg.reset_index().sort_values(["ciudad", "fecha"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    agg.to_csv(a.out, index=False, float_format="%.6g", date_format="%Y-%m-%d")
    print(f"wrote {a.out}: {len(agg)} station-months, {agg['ciudad'].nunique()} stations, "
          f"{agg['fecha'].min():%Y-%m} to {agg['fecha'].max():%Y-%m}, {int(agg['win_ok'].sum())} valid windows")


if __name__ == "__main__":
    main()
