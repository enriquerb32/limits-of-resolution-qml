"""Every number in the main-text tables of the v2 manuscript must follow from the archived v2 results.

The printed values are read from ``paper/main.tex``; the reference values come from the JSON files the
analysis scripts wrote (``results/``). A printed value passes when the archived value rounds to it.
Sign conventions that differ between table and archive (``A - B`` printed, ``B - A`` archived) are
handled by :func:`flip`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _latex import Printed, numbers, table_rows

from qml_resolution import paired_mde_hanley_mcneil

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
SUPP = ROOT / "paper" / "supp.tex"
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
    for key in ("ci_delong", "ci_block", "ci_two_way", "dbs_ci", "ci_cr2"):
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
    """Delta, SE, z, p, primary CR2 CI and p, years x seeds CI, cluster/iid ratio, MDE (Tables 3, 6, 7)."""
    gap = pair["gap"]
    return [
        gap,
        pair[se_key],
        gap / pair[se_key],
        pair["p_delong"],
        *pair["ci_cr2"],
        pair["p_cr2"],
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


# ------------------------------------------------------------------ Tables 3, 6, 7: paired comparisons
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
        "Fourier-2 - LSTM+QNN": ("FOURIER2-LSTM_QNN", False),
        "Full Fourier-1 - LSTM+QNN": ("FOURIER_FULL1-LSTM_QNN", False),
        "Full Fourier-2 - LSTM+QNN": ("FOURIER_FULL2-LSTM_QNN", False),
        "Frozen random - LSTM+QNN": ("FROZEN_RANDOM-LSTM_QNN", False),
        "Frozen zero-angle - LSTM+QNN": ("FROZEN_IDENTITY-LSTM_QNN", False),
    },
}
# rows moved to the supplement (same columns as Table 7)
SUPP_PAIRED = {
    "tab:controls_more": {
        "Fourier-1 - LSTM+QNN": ("FOURIER1-LSTM_QNN", False),
        "Fourier-3 - LSTM+QNN": ("FOURIER3-LSTM_QNN", False),
        "Fourier-4 - LSTM+QNN": ("FOURIER4-LSTM_QNN", False),
        "Full Fourier-2 - Fourier-2": ("FOURIER_FULL2-FOURIER2", False),
    },
}
PAIRED_CASES = [(TEX, tab, row, key, f) for tab, rows in PAIRED.items() for row, (key, f) in rows.items()]
PAIRED_CASES += [(SUPP, tab, row, key, f) for tab, rows in SUPP_PAIRED.items() for row, (key, f) in rows.items()]


@pytest.mark.parametrize(("path", "table", "row", "key", "flipped"), PAIRED_CASES,
                         ids=[f"{c[1]}:{c[2]}" for c in PAIRED_CASES])
def test_paired_comparisons(path: Path, table: str, row: str, key: str, flipped: bool) -> None:
    rows = table_rows(path, table)
    expected_rows = {**PAIRED, **SUPP_PAIRED}[table]
    assert set(rows) >= set(expected_rows), f"{table}: rows {sorted(rows)}"
    pair = flip(PAIRS[key]) if flipped else PAIRS[key]
    assert_row(flat(rows[row]), paired_row(pair), f"{table} / {row}")


def test_paired_tables_have_no_unchecked_rows() -> None:
    for path, tables in ((TEX, PAIRED), (SUPP, SUPP_PAIRED)):
        for table, expected in tables.items():
            printed = {r for r in table_rows(path, table) if r != "Difference"}
            assert printed == set(expected), f"{table}: unchecked rows {printed ^ set(expected)}"


# ------------------------------------------------------------------ Table 5: 15-seed AUC of every model
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


@pytest.mark.parametrize(("path", "label_", "complete"), [(TEX, "tab:case", False), (SUPP, "tab:case_all", True)],
                         ids=["main", "supplement"])
def test_model_auc_table(path: Path, label_: str, complete: bool) -> None:
    """The main table shows a subset of the models; the supplementary table shows all of them."""
    rows = {_norm(k): v for k, v in table_rows(path, label_).items() if k != "Model"}
    if complete:
        assert set(rows) == set(CASE), f"rows {sorted(set(rows) ^ set(CASE))}"
    else:
        assert set(rows) <= set(CASE) and len(rows) >= 12, f"rows {sorted(set(rows) - set(CASE))}"
    qnn = MODELS["LSTM_QNN"]["mean"]
    assert all(s["n_seeds"] == 15 for k, s in MODELS.items() if k != "CLIM")
    for label in rows:
        m = MODELS[CASE[label]]
        expected = [m["mean"], m["mean"] - qnn] if CASE[label] == "CLIM" else [m["mean"], m["sd"], m["mean"] - qnn]
        assert_row(flat(rows[label]), expected, label)


# ------------------------------------------------------------------ Table 8: positive control (teacher circuit)
TEACHER = {
    "Full Fourier-2 (625 tensor features)": "FOURIER_FULL2",
    "Full Fourier-1 (81 tensor features)": "FOURIER_FULL1",
    "Trained circuit": "CIRCUIT",
    "Fourier-4 (separable)": "FOURIER4",
    "Fourier-2 (separable)": "FOURIER2",
    "Linear-4D map": "LINEAR4D",
}


def test_teacher_table() -> None:
    """Single-teacher control (moved to the supplement; the main-text Table 8 is the repeated control)."""
    teacher = _load("local_exp38_teacher_control_interface.json")
    rows = {k: v for k, v in table_rows(SUPP, "tab:teacher_single").items() if not k.startswith("Head")}
    assert set(rows) == {*TEACHER, "Oracle (teacher score)"}
    assert_row(flat(rows["Oracle (teacher score)"]), [teacher["oracle_auc_test"]], "oracle")
    for label, key in TEACHER.items():
        m = teacher["models"][key]
        assert len(m["auc"]) == 5
        assert_row(flat(rows[label])[-2:], [m["mean"], m["sd"]], label)
    # main text: "most of its signal (89 %) lies in interactions between angles"
    assert round(100 * teacher["interaction_share"]) == 89 and "(89\\,\\%)" in TEX.read_text(encoding="utf-8")


# ------------------------------------------------------------------ Table 9: Brier and calibration
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


# ------------------------------------------------------------------ Table 10: within-station AUC
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


# ------------------------------------------------------------------ Table 11: station x month label
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


# ------------------------------------------------------------------ Table 12: SEAS5
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


# ------------------------------------------------------------------ Table 13: second panel
PANEL2_ROWS = {
    "RF - XGBoost": ("RF_2273-XGB", "RF_2273", "XGB"),
    "RF - LSTM": ("RF_2273-PureLSTM", "RF_2273", "PureLSTM"),
    "RF - LSTM+QNN": ("RF_2273-LSTM_QNN", "RF_2273", "LSTM_QNN"),
    "XGBoost - LSTM+QNN": ("XGB-LSTM_QNN", "XGB", "LSTM_QNN"),
    "LSTM - LSTM+QNN": ("PureLSTM-LSTM_QNN", "PureLSTM", "LSTM_QNN"),
    "Linear-4D - LSTM+QNN": ("CTRL_LINEAR4D-LSTM_QNN", "CTRL_LINEAR4D", "LSTM_QNN"),
    "Fourier-2 - LSTM+QNN": ("FOURIER2-LSTM_QNN", "FOURIER2", "LSTM_QNN"),
}


def _first_panel_probs() -> dict[str, np.ndarray]:
    h = np.load(RES / "json" / "results_hybrids_trainonly_fixet0_p60_28.1_probs.npz")
    c = np.load(RES / "json" / "results_rerun_train_only_canonical_probs.npz")
    k = np.load(RES / "json" / "results_controls_trainonly_fixet0_p60_28.1_probs.npz")
    out = {m: h[f"{m}_probs"] for m in ("RF_2273", "PureLSTM", "LSTM_QNN", "CTRL_LINEAR4D")}
    out.update(XGB=c["train_only_XGB_probs"], FOURIER2=k["FOURIER2_probs"], y=h["y_te"].astype(int))
    return out


def test_second_panel_table() -> None:
    from qml_resolution import mde, paired_delong

    rows = {k: v for k, v in table_rows(TEX, "tab:panel2").items() if k != "Comparison"}
    assert set(rows) == set(PANEL2_ROWS)
    assert (PANEL2["n_test"], PANEL2["K_years"]) == (884, 19)
    fp = _first_panel_probs()
    scale = np.sqrt(S42["n_test"] / PANEL2["n_test"])
    cr2 = _load("local_panel2_cr2.json")
    for label, (key, a, b) in PANEL2_ROWS.items():
        p = PANEL2["pairs"][key]
        first_panel = MODELS[a]["mean"] - MODELS[b]["mean"]
        predicted = float(mde(paired_delong(fp["y"], fp[a], fp[b]).se)) * scale  # per-contrast N^{-1/2} scaling
        expected = [p["delta"], first_panel, p["p_delong"], *cr2[key]["ci_cr2"], cr2[key]["p_cr2"], *p["ci_two_way"],
                    p["mde_delong"], predicted]
        expected += [p["ratio"], p["phases_excluding_zero"]]
        assert_row(flat(rows[label]), expected, label)


# ------------------------------------------------------------------ Table 4: Hanley-McNeil MDE (library)
def test_hanley_mcneil_table() -> None:
    rows = {k: v for k, v in table_rows(TEX, "tab:hm_mde").items() if k != "N"}
    assert len(rows) == 5
    for n_str, cells in rows.items():
        n = numbers(n_str)[0].value
        expected = [paired_mde_hanley_mcneil(n, prevalence=0.41), paired_mde_hanley_mcneil(n, prevalence=0.10)]
        assert_row(flat(cells), expected, n_str)


# ------------------------------------------------------------------ Figure 2 / Table S-resolution: resolution curve
def test_resolution_curve_table() -> None:
    rows = {k: v for k, v in table_rows(SUPP, "tab:resolution").items() if k not in ("", "Years")}
    curve = T43["resolution_curve"]
    assert set(rows) == set(curve)
    for years, c in curve.items():
        expected = [c["N"], c["HM"], c["single_RF_LSTM"], c["single_RF_QNN"], c["mean_RF_QNN"]]
        assert_row(flat(rows[years]), expected, f"{years} years")
    # the full-test-set entry of the curve is the MDE of the paired comparison in Table 6
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
    assert all(r > 1 for r in ratios)  # "larger than the seed variability of the classical models"
    for claim in ("529 test station-months", "about 0.024", "0.022--0.032 AUC", "4-qubit circuit"):
        assert claim in abstract, claim
    words = re.sub(r"\$[^$]*\$", "X", abstract.split("\\bigskip")[0]).split()
    assert len(words) <= 200, len(words)


# ------------------------------------------------------------------ numbers added in the response to review
def test_xgboost_trees_in_text() -> None:
    it = _load("json/xgb_best_iterations.json")
    assert it["max_abs_diff_archived_probs"] == 0.0
    trees = it["trees_used"]
    assert (min(trees), max(trees), np.median(trees)) == (2, 121, 16)
    assert sum(t <= 3 for t in trees) == 5
    text = TEX.read_text(encoding="utf-8")
    assert "2 to 121 across the 15 seeds (median 16, and 2--3 trees in five seeds)" in text


def test_cluster_periods_in_text() -> None:
    per = _load("local_exp45_cluster_periods.json")
    text = TEX.read_text(encoding="utf-8")
    lstm, qnn = per["RF-LSTM"], per["RF-QNN"]
    assert Printed(-88, 0).matches(100 * lstm["2008-2012"]["cross_share"])
    assert Printed(-99, 0).matches(100 * qnn["2008-2012"]["cross_share"])
    assert Printed(-25, 0).matches(100 * lstm["2013-2026"]["cross_share"])
    assert Printed(-3, 0).matches(100 * qnn["2013-2026"]["cross_share"])
    assert Printed(0.65, 2).matches(lstm["2013-2026"]["ratio"]) and Printed(0.87, 2).matches(qnn["2013-2026"]["ratio"])
    assert Printed(-0.004, 3).matches(lstm["2008-2012"]["gap"]) and Printed(-0.0005, 4).matches(qnn["2008-2012"]["gap"])
    assert Printed(0.023, 3).matches(lstm["2013-2026"]["gap"]) and Printed(0.030, 3).matches(qnn["2013-2026"]["gap"])
    assert Printed(0.60, 2).matches(lstm["no2026"]["ratio"]) and Printed(0.73, 2).matches(qnn["no2026"]["ratio"])
    assert lstm["2008-2012"]["years"] == 5 and lstm["all"]["n"] - lstm["no2026"]["n"] == 9
    for claim in (
        "$-88$\\,\\% and $-99$\\,\\%",
        "gaps $-0.004$ and $-0.0005$",
        "$-25$\\,\\% and $-3$\\,\\%",
        "ratios 0.65 and 0.87",
        "$+0.023$ and $+0.030$",
        "unchanged (0.60 and 0.73)",
    ):
        assert claim in text, claim


def test_auc_correlations_in_text() -> None:
    cor = _load("local_exp46_auc_correlation.json")
    pr, val = cor["pairs"], cor["validation"]
    assert Printed(0.78, 2).matches(pr["RF-LSTM_QNN"]["r_median"])
    assert Printed(0.86, 2).matches(pr["RF-PureLSTM"]["r_median"])
    assert Printed(0.91, 2).matches(pr["RF-XGB"]["r_median"])
    assert Printed(0.045, 3).matches(pr["RF-LSTM_QNN"]["hm_mde_at_r"])
    assert Printed(0.036, 3).matches(pr["RF-PureLSTM"]["hm_mde_at_r"])
    curve = T43["resolution_curve"]["19"]
    assert Printed(0.046, 3).matches(curve["single_RF_QNN"]) and Printed(0.035, 3).matches(curve["single_RF_LSTM"])
    assert val["n"] == 516 and Printed(36.6, 1).matches(100 * val["prevalence"])
    assert Printed(0.025, 3).matches(val["hm_se_one_auc_at_0.70"])
    assert Printed(0.030, 3).matches(val["hm_mde_paired_rho_0.90"]) and Printed(0.048, 3).matches(
        val["hm_mde_paired_rho_0.75"]
    )
    text = TEX.read_text(encoding="utf-8")
    for claim in (
        "0.78 for Random Forest and \\texttt{LSTM+QNN}, 0.86",
        "0.91 for the two tree ensembles",
        "gives 0.045 and 0.036",
        "empirical MDEs of 0.046 and 0.035",
        "516 validation rows (36.6\\,\\% positive)",
        "about 0.025",
        "about 0.030--0.048 AUC",
    ):
        assert claim in text, claim


# ------------------------------------------------------------------ Section 2 and Table 13: the monthly panels
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


# ------------------------------------------------------------------ numbers quoted in the running text
def test_running_text_numbers() -> None:
    text = TEX.read_text(encoding="utf-8")
    clim_sm = T43["recal_stationmonth"]["CLIM"]["auc"]
    assert Printed(0.479, 3).matches(clim_sm)
    assert text.count("against 0.479 for the climatology") == 2 and "0.480 for the climatology" not in text
    sim = _load("json/results_exp15_panel_sim.json")
    design = sim["design"]
    assert sum(design["year_sizes"]) == 529 and Printed(0.406, 3).matches(design["prevalence"])
    supp_text = SUPP.read_text(encoding="utf-8")   # the earlier DeLong-vs-block simulation is in the supplement
    assert "(529 rows, prevalence 0.406)" in supp_text
    rows = sim["rows"]
    null = [r for r in rows if r["delta_design"] == 0]
    alt = [r for r in rows if r["delta_design"] > 0]
    dl = [r["rejection_delong"] for r in null]
    bl = [r["rejection_block"] for r in null]
    pw = [r["rejection_delong"] for r in alt] + [r["rejection_block"] for r in alt]
    assert (min(dl), max(dl), min(bl), max(bl)) == pytest.approx((0.048, 0.058, 0.064, 0.088), abs=5e-4)
    assert (round(100 * min(pw)), round(100 * max(pw))) == (17, 24)
    assert "type-I errors of 0.048--0.058 for DeLong and 0.064--0.088 for the block bootstrap" in supp_text
    assert "17--24\\,\\% power" in supp_text
