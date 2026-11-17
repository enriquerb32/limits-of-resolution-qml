"""Every number in the main-text tables of the v2 manuscript must follow from the archived v2 results.

The printed values are read from ``paper/main.tex``; the reference values come from the JSON files the
analysis scripts wrote (``results/``). A printed value passes when the archived value rounds to it.
Sign conventions that differ between table and archive (``A - B`` printed, ``B - A`` archived) are
handled by :func:`flip`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _latex import Printed, numbers, table_rows

from qml_resolution import paired_mde_hanley_mcneil

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
RES = ROOT / "results"


def _load(name: str) -> Any:
    return json.loads((RES / name).read_text(encoding="utf-8"))


S42 = _load("local_exp42_v2_summary.json")
T43 = _load("local_exp43_v2_tables.json")
SEAS5 = _load("local_exp33_seas5_orig.json")
PANEL2 = _load("local_exp32_panel2_med2v2.json")["pooled"]
MODELS = S42["models"]
PAIRS = S42["pairs"]


def flip(pair: dict[str, Any]) -> dict[str, Any]:
    """The same comparison with the sign reversed (B - A instead of A - B)."""
    out = dict(pair)
    for key in ("gap", "delta", "z"):
        if key in out:
            out[key] = -out[key]
    for key in ("ci_delong", "ci_block", "ci_two_way", "dbs_ci"):
        if key in out:
            lo, hi = out[key]
            out[key] = [-hi, -lo]
    return out


def assert_row(printed: list[Printed], expected: list[float], where: str) -> None:
    assert len(printed) == len(expected), f"{where}: {len(printed)} printed numbers, {len(expected)} expected"
    for i, (p, e) in enumerate(zip(printed, expected, strict=True)):
        assert p.matches(e), f"{where}, number {i + 1}: printed {p.value} but archive gives {e:.6g}"


def flat(cells: list[str]) -> list[Printed]:
    return [n for c in cells for n in numbers(c)]


def paired_row(pair: dict[str, Any], mde_key: str = "mde_iid", se_key: str = "se_iid") -> list[float]:
    """Delta, SE, z, p, year-block CI, years x seeds CI, cluster/iid ratio, MDE (Tables 3, 5, 6)."""
    gap = pair["gap"]
    return [
        gap,
        pair[se_key],
        gap / pair[se_key],
        pair["p_delong"],
        *pair["ci_block"],
        *pair["ci_two_way"],
        pair["ratio_year_iid"],
        pair[mde_key],
    ]


# ------------------------------------------------------------------ Table 1: stations and test prevalence
def test_station_table_matches_test_partition() -> None:
    meta = pd.read_csv(RES / "test_meta.csv")
    meta["year"] = pd.to_datetime(meta["fecha"]).dt.year
    rows = table_rows(TEX, "tab:station_coverage")
    for station, g in meta.groupby("ciudad"):
        cells = rows[str(station).replace("_", " ")]
        n, pos = len(g), int(g["y_te"].sum())
        assert_row(flat(cells[1:]), [n, g["year"].max(), 100 * pos / n, pos, n], str(station))
    n, pos = len(meta), int(meta["y_te"].sum())
    assert_row(flat(rows["Total"]), [n, 100 * pos / n, pos, n], "Total")
    assert n == S42["n_test"] == 529
    assert meta["year"].nunique() == S42["n_years"] == 19


# ------------------------------------------------------------------ Table 2: noise budget
BUDGET_COLS = ["RF", "XGB", "PureLSTM", "LSTM_QNN", "LSTM_QTL"]


def test_noise_budget() -> None:
    rows = table_rows(TEX, "tab:budget")
    nb = T43["noise_budget"]
    for j, m in enumerate(BUDGET_COLS):
        assert numbers(rows["Mean AUC (15 seeds)"][j])[0].matches(nb[m]["mean"]), m
        assert numbers(rows["SD across seeds"][j])[0].matches(nb[m]["sd"]), m
        assert_row(numbers(rows["SE of one AUC, iid / block"][j]), [nb[m]["se_iid"], nb[m]["se_block"]], m)
        share = numbers(rows["Seed share, iid / block"][j])
        assert_row(share, [100 * nb[m]["share_iid"], 100 * nb[m]["share_block"]], m)
        # internal consistency of the archive: the budget uses the same 15-seed AUCs as the summary
        key = "RF_2273" if m == "RF" else m
        assert nb[m]["mean"] == pytest.approx(MODELS[key]["mean"], abs=1e-12)
        assert nb[m]["share_iid"] == pytest.approx(nb[m]["sd"] ** 2 / (nb[m]["sd"] ** 2 + nb[m]["se_iid"] ** 2))


# ------------------------------------------------------------------ Tables 3, 5, 6: paired comparisons
PAIRED = {
    "tab:paired_ci": {
        "RF - XGBoost": ("RF-XGB", False),
        "RF - LSTM": ("RF-PureLSTM", False),
        "XGBoost - LSTM": ("XGB-PureLSTM", False),
    },
    "tab:qnn_paired": {
        "RF - LSTM+QNN": ("RF-LSTM_QNN", False),
        "Linear-4D - LSTM+QNN": ("LSTM_QNN-CTRL_LINEAR4D", True),
        "LSTM - LSTM+QNN": ("PureLSTM-LSTM_QNN", False),
        "RF - LSTM+QTL": ("RF-LSTM_QTL", False),
        "LSTM+QTL - its extractor": ("LSTM_QTL-PureLSTM", False),
        "LSTM+QTL - QTL-classical": ("LSTM_QTL-QTL_CLASSICAL", False),
        "LSTM+QTL - QTL tensor-81": ("LSTM_QTL-QTL_TENSOR1", False),
    },
    "tab:controls": {
        "Fourier-1 - LSTM+QNN": ("FOURIER1-LSTM_QNN", False),
        "Fourier-2 - LSTM+QNN": ("FOURIER2-LSTM_QNN", False),
        "Fourier-3 - LSTM+QNN": ("FOURIER3-LSTM_QNN", False),
        "Fourier-4 - LSTM+QNN": ("FOURIER4-LSTM_QNN", False),
        "Full Fourier-1 - LSTM+QNN": ("FOURIER_FULL1-LSTM_QNN", False),
        "Full Fourier-2 - LSTM+QNN": ("FOURIER_FULL2-LSTM_QNN", False),
        "Frozen random - LSTM+QNN": ("FROZEN_RANDOM-LSTM_QNN", False),
        "Frozen zero-angle - LSTM+QNN": ("FROZEN_IDENTITY-LSTM_QNN", False),
        "Full Fourier-2 - Fourier-2": ("FOURIER_FULL2-FOURIER2", False),
    },
}
PAIRED_CASES = [(tab, row, key, f) for tab, rows in PAIRED.items() for row, (key, f) in rows.items()]


@pytest.mark.parametrize(("table", "row", "key", "flipped"), PAIRED_CASES, ids=[c[1] for c in PAIRED_CASES])
def test_paired_comparisons(table: str, row: str, key: str, flipped: bool) -> None:
    rows = table_rows(TEX, table)
    assert set(rows) >= set(PAIRED[table]), f"{table}: rows {sorted(rows)}"
    pair = flip(PAIRS[key]) if flipped else PAIRS[key]
    assert_row(flat(rows[row]), paired_row(pair), f"{table} / {row}")


def test_paired_tables_have_no_unchecked_rows() -> None:
    for table, expected in PAIRED.items():
        printed = {r for r in table_rows(TEX, table) if r != "Difference"}
        assert printed == set(expected), f"{table}: unchecked rows {printed - set(expected)}"


# ------------------------------------------------------------------ Table 4: 15-seed AUC of every model
CASE = {
    "Month x station climatology": "CLIM",
    "Random Forest (tuned)": "RF_2273",
    "XGBoost (tuned, early stopping)": "XGB",
    "Standalone LSTM": "PureLSTM",
    "LSTM+QNN (24 quantum parameters)": "LSTM_QNN",
    "LSTM+QTL": "LSTM_QTL",
    "LSTM+PI-QNN (penalized variant)": "LSTM_PIQNN",
    "Full Fourier-1 (81 features)": "FOURIER_FULL1",
    "Fourier-1 (36)": "FOURIER1",
    "Fourier-2 (68)": "FOURIER2",
    "Fourier-3 (100)": "FOURIER3",
    "Fourier-4 (132)": "FOURIER4",
    "QTL tensor-81 (linear map on 81 features)": "QTL_TENSOR1",
    "Full Fourier-2 (625 features)": "FOURIER_FULL2",
    "QTL-classical (linear head)": "QTL_CLASSICAL",
    "Linear-4D control (20)": "CTRL_LINEAR4D",
    "Frozen random circuit": "FROZEN_RANDOM",
    "Frozen zero-angle circuit": "FROZEN_IDENTITY",
}


def _norm(label: str) -> str:
    return label.replace("Month\\times station", "Month x station").replace("Month\\timesstation", "Month x station")


def test_model_auc_table() -> None:
    rows = {_norm(k): v for k, v in table_rows(TEX, "tab:case").items() if k != "Model"}
    assert set(rows) == set(CASE), f"rows {sorted(set(rows) ^ set(CASE))}"
    qnn = MODELS["LSTM_QNN"]["mean"]
    for label, key in CASE.items():
        m = MODELS[key]
        expected = [m["mean"], m["mean"] - qnn] if key == "CLIM" else [m["mean"], m["sd"], m["mean"] - qnn]
        assert all(s["n_seeds"] == 15 for k, s in MODELS.items() if k != "CLIM")
        assert_row(flat(rows[label]), expected, label)


# ------------------------------------------------------------------ Table 7: Brier and calibration
BRIER = {
    "Random Forest": ("RF", "RF_2273"),
    "XGBoost": ("XGB", "XGB"),
    "Standalone LSTM": ("PureLSTM", "PureLSTM"),
    "Linear-4D control": ("CTRL_LINEAR4D", "CTRL_LINEAR4D"),
    "LSTM+QNN": ("LSTM_QNN", "LSTM_QNN"),
}


def test_brier_table() -> None:
    rows = table_rows(TEX, "tab:brier")
    clim_raw, clim_rc = T43["brier_raw"]["CLIM"], T43["recal_orig"]["CLIM"]
    assert_row(
        flat(rows["Climatology"]),
        [clim_raw["brier"], 0, clim_raw["ece"], clim_rc["brier_platt"], 0, clim_rc["ece_platt"]],
        "Climatology",
    )
    for label, (k_raw, k_rc) in BRIER.items():
        r, c = T43["brier_raw"][k_raw], T43["recal_orig"][k_rc]
        expected = [
            r["brier"],
            r["bss"],
            *r["dbs_ci"],
            r["ece"],
            c["brier_platt"],
            c["bss_platt_vs_raw"],
            c["n_pos_raw"],
            c["bss_platt_vs_recal"],
            c["n_pos_recal"],
            *c["dbs_vs_recal_ci"],
            c["ece_platt"],
        ]
        assert_row(flat(rows[label]), expected, label)


# ------------------------------------------------------------------ Table 8: within-station AUC
WITHIN = {
    "Random Forest": "RF",
    "XGBoost": "XGB",
    "Standalone LSTM": "PureLSTM",
    "Linear-4D control": "CTRL_LINEAR4D",
    "LSTM+QNN": "LSTM_QNN",
    "LSTM+QTL": "LSTM_QTL",
}
STATIONS = ["Alicante", "Barcelona", "San_Javier", "Valencia"]


def test_within_station_table() -> None:
    rows = {_norm(k): v for k, v in table_rows(TEX, "tab:within").items()}
    ws = T43["within_station"]
    c = ws["CLIM"]
    assert_row(
        flat(rows["Month x station climatology"]), [c["pooled"], c["within"], *(c["per"][s] for s in STATIONS)], "clim"
    )
    for label, key in WITHIN.items():
        w = ws[key]
        expected = [w["pooled"], w["within"], w["within_sd"], *(w["per"][s] for s in STATIONS)]
        assert_row(flat(rows[label]), expected, label)


# ------------------------------------------------------------------ Table 9: station x month label
def test_stationmonth_table() -> None:
    rows = {_norm(k): v for k, v in table_rows(TEX, "tab:stationmonth").items()}
    rc = T43["recal_stationmonth"]
    assert_row(flat(rows["Month x station climatology"]), [rc["CLIM"]["auc"]], "clim")
    for label, (_, key) in BRIER.items():
        c = rc[key]
        expected = [
            c["auc"],
            c["auc_sd"],
            c["bss_platt_vs_raw"],
            c["n_pos_raw"],
            c["bss_platt_vs_recal"],
            c["n_pos_recal"],
            *c["dbs_vs_recal_ci"],
        ]
        assert_row(flat(rows[label]), expected, label)


# ------------------------------------------------------------------ Table 10: SEAS5
SEAS5_ROWS = {
    "SEAS5 - climatology": "SEAS5_QM_L1-CLIM",
    "SEAS5 (lead 2) - climatology": "SEAS5_QM_L2-CLIM",
    "Random Forest - SEAS5": "RF_2273-SEAS5_QM_L1",
    "LSTM+QNN - SEAS5": "LSTM_QNN-SEAS5_QM_L1",
    "Blend RF - blend climatology": "BLEND_RF_2273_L1-BLEND_CLIM_L1",
    "Blend XGBoost - blend climatology": "BLEND_XGB_L1-BLEND_CLIM_L1",
    "Blend LSTM - blend climatology": "BLEND_PureLSTM_L1-BLEND_CLIM_L1",
    "Blend Linear-4D - blend climatology": "BLEND_CTRL_LINEAR4D_L1-BLEND_CLIM_L1",
    "Blend LSTM+QNN - blend climatology": "BLEND_LSTM_QNN_L1-BLEND_CLIM_L1",
    "Blend LSTM+QNN - blend Linear-4D": "BLEND_LSTM_QNN_L1-BLEND_CTRL_LINEAR4D_L1",
}


def test_seas5_table() -> None:
    rows = {k: v for k, v in table_rows(TEX, "tab:seas5").items() if k != "Comparison"}
    assert set(rows) == set(SEAS5_ROWS)
    assert SEAS5["n_test"] == 529
    for label, key in SEAS5_ROWS.items():
        p = SEAS5["pairs"][key]
        expected = [p["delta"], p["p_delong"], *p["ci_delong"], *p["ci_block"], *p["ci_two_way"], p["ratio"]]
        expected.append(p["mde_delong"])
        assert_row(flat(rows[label]), expected, label)


# ------------------------------------------------------------------ Table 11: second panel
PANEL2_ROWS = {
    "RF - XGBoost": ("RF_2273-XGB", "RF_2273", "XGB"),
    "RF - LSTM": ("RF_2273-PureLSTM", "RF_2273", "PureLSTM"),
    "RF - LSTM+QNN": ("RF_2273-LSTM_QNN", "RF_2273", "LSTM_QNN"),
    "XGBoost - LSTM+QNN": ("XGB-LSTM_QNN", "XGB", "LSTM_QNN"),
    "LSTM - LSTM+QNN": ("PureLSTM-LSTM_QNN", "PureLSTM", "LSTM_QNN"),
    "Linear-4D - LSTM+QNN": ("CTRL_LINEAR4D-LSTM_QNN", "CTRL_LINEAR4D", "LSTM_QNN"),
    "Fourier-2 - LSTM+QNN": ("FOURIER2-LSTM_QNN", "FOURIER2", "LSTM_QNN"),
}


def test_second_panel_table() -> None:
    rows = {k: v for k, v in table_rows(TEX, "tab:panel2").items() if k != "Comparison"}
    assert set(rows) == set(PANEL2_ROWS)
    assert (PANEL2["n_test"], PANEL2["K_years"]) == (884, 19)
    for label, (key, a, b) in PANEL2_ROWS.items():
        p = PANEL2["pairs"][key]
        first_panel = MODELS[a]["mean"] - MODELS[b]["mean"]
        expected = [p["delta"], first_panel, p["p_delong"], *p["ci_block"], *p["ci_two_way"], p["mde_delong"]]
        expected += [p["ratio"], p["phases_excluding_zero"]]
        assert_row(flat(rows[label]), expected, label)


# ------------------------------------------------------------------ Table 12: Hanley-McNeil MDE (library)
def test_hanley_mcneil_table() -> None:
    rows = {k: v for k, v in table_rows(TEX, "tab:hm_mde").items() if k != "N"}
    assert len(rows) == 5
    for n_str, cells in rows.items():
        n = numbers(n_str)[0].value
        expected = [paired_mde_hanley_mcneil(n, prevalence=0.41), paired_mde_hanley_mcneil(n, prevalence=0.10)]
        assert_row(flat(cells), expected, n_str)


# ------------------------------------------------------------------ Table 13: empirical resolution curve
def test_resolution_curve_table() -> None:
    rows = {k: v for k, v in table_rows(TEX, "tab:resolution").items() if k not in ("", "Years")}
    curve = T43["resolution_curve"]
    assert set(rows) == set(curve)
    for years, c in curve.items():
        expected = [c["N"], c["HM"], c["single_RF_LSTM"], c["single_RF_QNN"], c["mean_RF_QNN"]]
        assert_row(flat(rows[years]), expected, f"{years} years")
    # the full-test-set entry of the curve is the MDE of the paired comparison in Table 5
    assert curve["19"]["mean_RF_QNN"] == pytest.approx(PAIRS["RF-LSTM_QNN"]["mde_iid"], rel=1e-6)


# ------------------------------------------------------------------ Abstract
def test_abstract_claims() -> None:
    nb = T43["noise_budget"]
    ratios = [nb[m]["se_iid"] / nb[m]["sd"] for m in ("RF", "XGB", "PureLSTM")]
    assert round(min(ratios)) == 3 and round(max(ratios)) == 21  # "3--21 times the seed SD"
    mdes = [PAIRS[k]["mde_iid"] for k in ("RF-XGB", "RF-PureLSTM", "XGB-PureLSTM", "RF-LSTM_QNN")]
    assert (round(min(mdes), 3), round(max(mdes), 3)) == (0.022, 0.032)  # "0.022--0.032 AUC"
    gaps = [PAIRS[k]["gap"] for k in ("RF-LSTM_QNN", "RF-LSTM_QTL")]
    assert (round(min(gaps), 3), round(max(gaps), 3)) == (0.020, 0.024)  # "0.020--0.024 AUC below RF"
    assert round(SEAS5["pairs"]["SEAS5_QM_L1-CLIM"]["delta"], 3) == 0.043  # "+0.043 AUC"
    assert np.mean([nb[m]["se_iid"] for m in BUDGET_COLS]) == pytest.approx(0.024, abs=5e-4)  # "about 0.024"
    abstract = TEX.read_text(encoding="utf-8").split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    for claim in ("529 test station-months", "3--21 times", "0.022--0.032", "0.020--0.024", "$+0.043$"):
        assert claim in abstract, claim


# ------------------------------------------------------------------ Section 2 and Table 11: the monthly panels
def _panel(path: Path, meta: Path) -> dict[str, Any]:
    """Partition, P60 and test labels rebuilt from a tracked monthly panel (scripts/export_monthly_panel.py)."""
    p = pd.read_csv(path, parse_dates=["fecha"])
    year = p["fecha"].dt.year
    p60 = float(np.percentile(p.loc[(year <= 1996) & p["label_ok"], "precip_label"], 60))
    w = p[p["win_ok"]]
    wy = w["fecha"].dt.year
    test = w[wy >= 2008].assign(y=lambda d: (d["precip_label"] > p60).astype(int))
    ref = pd.read_csv(meta, parse_dates=["fecha"]).merge(test, on=["ciudad", "fecha"], how="outer", indicator=True)
    return {
        "n": [int((wy <= 1996).sum()), int(((wy >= 1997) & (wy <= 2007)).sum()), len(test)],
        "p60": p60,
        "prevalence": float(test["y"].mean()),
        "same_rows": bool((ref["_merge"] == "both").all()),
        "same_labels": bool((ref["y_te"] == ref["y"]).all()),
    }


def test_first_panel_reproduces_partition_and_labels() -> None:
    panel = _panel(RES / "panel_v2_monthly.csv", RES / "test_meta.csv")
    assert panel["same_rows"] and panel["same_labels"]
    assert panel["n"] == [2332, 516, 529]
    text = TEX.read_text(encoding="utf-8")
    assert "training 1938--1996 (2{,}332 windows), validation 1997--2007 (516) and test 2008--2026 (529;" in text
    assert "$P_{60}=\\qty{29.9}{\\milli\\meter}$" in text
    assert Printed(29.9, 1).matches(panel["p60"])
    assert Printed(40.6, 1).matches(100 * panel["prevalence"])


def test_second_panel_reproduces_partition_and_labels() -> None:
    panel = _panel(RES / "panel2" / "med2v2" / "panel_monthly.csv", RES / "panel2" / "med2v2" / "test_meta_pooled.csv")
    assert panel["same_rows"] and panel["same_labels"]
    assert panel["n"][0] == 1694 and panel["n"][2] == PANEL2["n_test"] == 884
    text = TEX.read_text(encoding="utf-8")
    assert "884 station-months (34.7\\,\\% positive)" in text and "1{,}694 trai" in text
    assert Printed(34.7, 1).matches(100 * panel["prevalence"])
