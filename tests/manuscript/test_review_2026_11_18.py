"""Numbers added in the revision after the review of 2026-11-18 must follow from the archived results.

Covers the lookahead audit (S1), the calibration of the primary procedure (exp48), the sensitivity to clusters of
consecutive years, the primary procedure for the SEAS5 contrasts, and the three checks of exp49.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _latex import Printed, numbers, table_rows

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
SUPP = ROOT / "paper" / "supp.tex"
RES = ROOT / "results"
MAIN_TEXT = TEX.read_text(encoding="utf-8")
SUPP_TEXT = SUPP.read_text(encoding="utf-8")


def _load(name: str) -> Any:
    return json.loads((RES / name).read_text(encoding="utf-8"))


A47 = _load("local_exp47_truncation_audit.json")
C48 = _load("local_exp48_primary_calibration.json")
C48_2 = _load("local_exp48_primary_calibration_2yr.json")
M49 = _load("local_exp49_review_minor.json")
BLK = _load("local_cr2_block_sensitivity.json")
S5 = _load("local_seas5_cr2.json")
S42 = _load("local_exp42_v2_summary.json")


def p(value: float, digits: int) -> Printed:
    return Printed(round(value, digits), digits)


def in_text(text: str, *claims: str) -> None:
    for c in claims:
        assert c in text, c


def test_no_unprotected_path_in_captions() -> None:
    """\\path is fragile: inside a \\caption it breaks the build unless protected (it did once)."""
    import re

    def captions(text: str) -> list[str]:
        out, i = [], 0
        while (i := text.find("\\caption{", i)) != -1:
            j, depth = i + len("\\caption{"), 1
            while depth:
                depth += {"{": 1, "}": -1}.get(text[j], 0) if text[j - 1] != "\\" else 0
                j += 1
            out.append(text[i + len("\\caption{"):j - 1]); i = j
        return out

    for name, text in (("main", MAIN_TEXT), ("supp", SUPP_TEXT)):
        caps = captions(text)
        assert len(caps) >= 10, name
        for cap in caps:
            assert not re.search(r"(?<!\\protect)\\path\{", cap), f"{name}: unprotected \\path in caption: {cap[:80]}"


# ------------------------------------------------------------------ lookahead audit
def test_truncation_test_and_reach() -> None:
    arc, cau = A47["truncation_archived"], A47["truncation_causal"]
    assert arc["test_rows_full"] == arc["rows_unchanged"] == 529
    assert arc["eligible_full_only"] == arc["eligible_cut_only"] == arc["rows_with_changed_inputs"] == 0
    assert cau["test_rows_full"] == cau["rows_unchanged"] == 533 and cau["rows_with_changed_inputs"] == 0
    reach = A47["interpolation_reach"]["archived"]
    assert reach["filled_station_months"] == 54
    assert reach["windows_with_filled_last_month"] == {"train": 24, "val": 2, "test": 0}
    in_text(SUPP_TEXT, "54 station-months", "24 training, 2 validation windows", "533 test windows instead of 529")
    in_text(MAIN_TEXT, "24 training and 2 validation windows", "all 529 test windows have the same inputs")


def test_label_completeness_and_p60_set() -> None:
    lab = A47["labels"]
    assert lab["test_rows_incomplete"] == 0
    p60 = lab["p60_month_set"]
    assert p60["n_months"] == p60["n_months_complete"] == 2847
    assert p60["by_station"] == {"Alicante": 699, "Barcelona": 809, "San_Javier": 629, "Valencia": 710}
    assert round(p60["p60"], 1) == round(p60["p60_complete_only"], 1) == 29.9
    reach = A47["interpolation_reach"]
    assert reach["labelled_months"]["val"] == 528
    assert {k: v for k, v in reach["labelled_months_with_missing_days"].items() if v} == {"test": 1}
    in_text(SUPP_TEXT, "2{,}847 labelled training months", "Alicante 699, Barcelona 809, San Javier 629, Valencia 710",
            "all 528 labelled validation months")


def test_lookahead_table() -> None:
    rows = {k: v for k, v in table_rows(SUPP, "tab:lookahead").items() if k != "Model"}
    src = {"Random Forest": A47["refit_trees"]["RF"], "XGBoost": A47["refit_trees"]["XGB"],
           "LSTM": A47["refit_neural"]["PureLSTM"], "LSTM+QNN": A47["refit_neural"]["LSTM_QNN"]}
    assert set(rows) == set(src)
    for name, r in src.items():
        expected = [r["auc_archived_common"], r["auc_causal_common"], r["delta_archived_minus_causal"], *r["ci95_year_block"]]
        printed = [n for c in rows[name] for n in numbers(c)]
        assert len(printed) == len(expected), name
        for i, (pr, e) in enumerate(zip(printed, expected, strict=True)):
            assert pr.matches(e), f"{name} {i}: printed {pr.value}, archive {e:.5f}"
    assert max(abs(r["delta_archived_minus_causal"]) for r in src.values()) < 0.0025   # "at most 0.002"


# ------------------------------------------------------------------ primary procedure
def _groups() -> dict[str, list[dict[str, Any]]]:
    sc = {s["name"]: s for s in C48["scenarios"]}
    rows = C48["rows"]
    return {
        "19 years, independent": [r for r in rows if r["mask"] == "real" and sc[r["scenario"]]["phi"] == 0],
        "10 years": [r for r in rows if r["mask"] == "K10"],
        "38 years": [r for r in rows if r["mask"] == "K38"],
        "2 or 4 stations every year": [r for r in rows if r["mask"] in ("m2", "m4")],
        "AR(1) years, scorer-specific effects": [r for r in rows if r["scenario"] == "specific_ar1"],
        "AR(1) years with year$\\times$station effects": [r for r in rows if r["scenario"] == "station_year_specific_ar1"],
    }


def test_primary_calibration_table() -> None:
    rows = table_rows(SUPP, "tab:primary_cal")
    methods = ["delong", "block_pct", "cr1_t", "cr2_bm", "jackknife_t", "wild_webb"]
    for name, cells in _groups().items():
        key = f"{name.replace('$', '')} ({len(cells)} cells)"            # the row label states the number of cells
        assert key in rows, (key, list(rows))
        printed = [n for c in rows[key] for n in numbers(c)]
        expected = [v for m in methods for v in (min(r["coverage"][m] for r in cells), max(r["coverage"][m] for r in cells))]
        for i, (pr, e) in enumerate(zip([abs(x.value) for x in printed], expected, strict=True)):
            assert round(e, 3) == pytest.approx(pr, abs=5e-4), f"{name} {i}: printed {pr}, archive {e:.4f}"


def test_primary_calibration_claims() -> None:
    sc = {s["name"]: s for s in C48["scenarios"]}
    indep = [r for r in C48["rows"] if sc[r["scenario"]]["phi"] == 0]
    cr2 = [r["coverage"]["cr2_bm"] for r in indep]
    assert (round(min(cr2), 3), round(max(cr2), 3)) == (0.935, 0.975)
    assert round(min(r["coverage"]["delong"] for r in indep), 2) == 0.85 or round(min(r["coverage"]["delong"] for r in indep), 3) == 0.845
    null_rej = [r["rejection"]["block_pct"] for r in C48["rows"] if r["delta"] == 0 and r["delta_true_mcse"] == 0]
    assert (round(100 * min(null_rej)), round(100 * max(null_rej))) == (6, 13)     # "6--13 %"
    in_text(MAIN_TEXT, "rejects a null difference in 6--13\\,\\% of replications")
    k19 = [r for r in C48["rows"] if r["mask"] in ("real", "m2", "m4")]
    assert round(max(r["p_ratio_le_0_59"] for r in k19), 3) == 0.012
    assert round(max(r["p_ratio_le_0_72"] for r in k19), 3) == 0.058
    df = {round(r["mean_df_bm"], 1) for r in C48["rows"] if r["mask"] == "real"}
    assert df == {15.9}
    in_text(MAIN_TEXT, "covers at 0.935--0.975", "$\\nu=15.9$", "more than 1.2\\,\\%")


def test_two_year_clusters() -> None:
    def cov(rows: list[dict[str, Any]], scen: str, d: float) -> float:
        return next(r for r in rows if r["scenario"] == scen and r["delta"] == d and r["mask"] == "real")["coverage"]["cr2_bm"]

    assert round(cov(C48["rows"], "specific_ar1", 0.0), 3) == 0.937 and round(cov(C48_2["rows"], "specific_ar1", 0.0), 3) == 0.953
    assert round(cov(C48["rows"], "station_year_specific_ar1", 0.0), 3) == 0.908
    assert round(cov(C48_2["rows"], "station_year_specific_ar1", 0.0), 3) == 0.915
    pw1 = next(r for r in C48["rows"] if r["scenario"] == "iid" and r["delta"] == 0.02)["rejection"]["cr2_bm"]
    pw2 = next(r for r in C48_2["rows"] if r["scenario"] == "iid" and r["delta"] == 0.02)["rejection"]["cr2_bm"]
    assert (round(pw1, 2), round(pw2, 2)) == (0.28, 0.22)
    pv = {k: {b: round(v[b]["p"], 3) for b in v} for k, v in BLK.items()}
    assert (pv["RF-PureLSTM"]["2yr"], pv["RF-LSTM_QNN"]["2yr"]) == (0.061, 0.051)
    assert pv["RF-LSTM_QTL"]["2yr"] == 0.027 and pv["RF-LSTM_QTL"]["3yr"] == 0.064
    assert max(max(v.values()) for k, v in pv.items() if k.startswith("FROZEN")) <= 0.018
    in_text(SUPP_TEXT, "($p=0.061$ and 0.051)", "($p=0.027$)", "($p=0.064$)", "($p\\le0.018$)")


def test_primary_mde_and_scope() -> None:
    m = _load("local_primary_mde.json")["pairs"]
    r3 = lambda ks, f: (round(min(m[k][f] for k in ks), 3), round(max(m[k][f] for k in ks), 3))  # noqa: E731
    classical = ["RF-XGB", "RF-PureLSTM", "XGB-PureLSTM"]
    complete = classical + ["RF-LSTM_QNN"]
    interface = ["LSTM_QNN-CTRL_LINEAR4D", "PureLSTM-LSTM_QNN", "LSTM_QTL-PureLSTM", "LSTM_QTL-QTL_CLASSICAL", "LSTM_QTL-QTL_TENSOR1"]
    assert r3(classical, "mde_cr2") == (0.018, 0.024) and r3(complete, "mde_cr2") == (0.018, 0.025)
    assert r3(classical, "mde_cr2_holm_first_step") == (0.021, 0.029)
    assert r3(interface, "mde_cr2") == (0.012, 0.019) and r3(interface, "mde_delong") == (0.011, 0.016)
    assert round(m["RF-LSTM_QNN"]["mde_cr2"], 3) == 0.025
    for k, v in S42["pairs"].items():                         # tables use the exp42 value; both computations agree
        if k in m:
            assert v["mde_cr2"] == pytest.approx(m[k]["mde_cr2"], rel=1e-9), k
    in_text(MAIN_TEXT, "is 0.018--0.025 AUC (0.022--0.032 AUC under independence)",
            "0.018--0.024 AUC for the classical models (0.022--0.031 under independence, and 0.021--0.029 at the first Holm step)",
            "the MDEs of these comparisons are 0.012--0.019 (0.011--0.016 under independence)")
    sm = _load("local_stationmonth_cr2.json")
    between = [v["p_cr2"] for k, v in sm.items() if not k.endswith("CLIM")]
    clim = [v["p_cr2"] for k, v in sm.items() if k.endswith("CLIM")]
    assert round(min(between), 2) == 0.72 and round(min(clim), 2) == 0.14
    in_text(MAIN_TEXT, "(CR2 $p\\ge0.72$ between models and $p\\ge0.14$ against the climatology")
    two = _load("local_two_station_subset.json")
    assert two["rows"] == 441 and two["years"] == 19
    assert all(v["resolved"] == v["resolved_all"] for v in two["pairs"].values())
    p = two["pairs"]
    assert (round(p["RF-PureLSTM"]["delta"], 3), round(p["RF-LSTM_QNN"]["delta"], 3)) == (0.019, 0.027)
    assert (round(p["RF-PureLSTM"]["p_cr2"], 3), round(p["RF-LSTM_QNN"]["p_cr2"], 3)) == (0.026, 0.014)
    in_text(MAIN_TEXT, "on Alicante and Barcelona alone, present in every test year (441 rows)", "(CR2 $p=0.026$ and 0.014)")


def test_leave_one_year() -> None:
    lo = _load("local_cr2_leave_one_year.json")
    assert len(lo["RF-PureLSTM"]["years_that_flip_verdict"]) == 9
    assert lo["RF-LSTM_QNN"]["years_that_flip_verdict"] == ["2020", "2023"] or lo["RF-LSTM_QNN"]["years_that_flip_verdict"] == [2020, 2023]
    assert [int(u) for u in lo["RF-XGB"]["years_that_flip_verdict"]] == [2020, 2022]
    for k in ("RF-LSTM_QTL", "FROZEN_RANDOM-LSTM_QNN", "FROZEN_IDENTITY-LSTM_QNN", "LSTM_QNN-CTRL_LINEAR4D"):
        assert lo[k]["years_that_flip_verdict"] == [], k
    r = lambda k, i: round(lo[k]["p_range"][i], 3)  # noqa: E731
    assert (r("RF-PureLSTM", 0), r("RF-PureLSTM", 1)) == (0.013, 0.079)
    assert (r("RF-LSTM_QNN", 0), r("RF-LSTM_QNN", 1)) == (0.021, 0.054)
    assert (r("RF-XGB", 0), r("RF-XGB", 1)) == (0.032, 0.232)
    assert r("RF-LSTM_QTL", 1) == 0.017 and max(r("FROZEN_RANDOM-LSTM_QNN", 1), r("FROZEN_IDENTITY-LSTM_QNN", 1)) == 0.037
    assert round(lo["LSTM_QNN-CTRL_LINEAR4D"]["p_range"][0], 2) == 0.34
    in_text(MAIN_TEXT, "for 9 of the 19 years")
    in_text(SUPP_TEXT, "($p$ from 0.013 to 0.079)", "(2020 and 2023; $p$ from 0.021 to 0.054)", "($p$ from 0.032 to 0.232)")


def test_primary_verdicts_in_text() -> None:
    pr = S42["pairs"]
    assert round(pr["RF-PureLSTM"]["p_cr2"], 3) == 0.041 and round(pr["RF-LSTM_QNN"]["p_cr2"], 3) == 0.031
    assert round(pr["RF-LSTM_QTL"]["p_cr2"], 3) == 0.008
    assert all(pr[k]["p_cr2"] > 0.05 / 3 for k in ("RF-XGB", "RF-PureLSTM", "XGB-PureLSTM"))       # none passes Holm
    assert all(pr[k]["p_cr2"] > 0.05 / 3 for k in ("RF-LSTM_QNN", "LSTM_QNN-CTRL_LINEAR4D", "PureLSTM-LSTM_QNN"))
    assert {round(pr[k]["p_cr2"], 3) for k in ("FROZEN_RANDOM-LSTM_QNN", "FROZEN_IDENTITY-LSTM_QNN")} == {0.014, 0.010}
    in_text(MAIN_TEXT, "$p=0.041$, above the Holm threshold", "(CR2 $p=0.031$)", "resolved by CR2 ($p=0.008$)", "$p=0.010$--0.014")


def test_seas5_primary() -> None:
    s = {k: (round(v["ci_cr2"][0], 3), round(v["ci_cr2"][1], 3), round(v["p_cr2"], 3)) for k, v in S5.items()}
    assert s["orig:SEAS5_QM_L1-CLIM"] == (-0.008, 0.095, 0.093)
    assert s["orig:RF_2273-SEAS5_QM_L1"] == (-0.102, 0.002, 0.056)
    assert s["orig:LSTM_QNN-SEAS5_QM_L1"][2] == 0.014
    assert s["stationmonth:SEAS5_QM_L1-CLIM"][:2] == (0.081, 0.237)
    in_text(MAIN_TEXT, "CR2 $[-0.008,+0.095]$, $p=0.093$", "CR2 $[-0.102,+0.002]$, $p=0.056$", "CR2 $[+0.081,+0.237]$")


def test_panel2_primary() -> None:
    p2 = _load("local_panel2_cr2.json")
    panel2 = _load("local_exp32_panel2_med2v2.json")["pooled"]["pairs"]
    assert len(p2) == 7
    assert set(p2) <= set(panel2)
    for k, v in p2.items():
        assert v["delta"] == pytest.approx(panel2[k]["delta"], abs=1e-9), k
    assert round(min(v["p_cr2"] for v in p2.values()), 3) == 0.107
    assert all(not (v["ci_cr2"][0] > 0 or v["ci_cr2"][1] < 0) for v in p2.values())
    assert {round(v["df"], 1) for v in p2.values()} == {17.7}
    in_text(MAIN_TEXT, "$\\nu=17.7$", "is resolved ($p\\ge0.107$")


# ------------------------------------------------------------------ exp52: emulation and common angles
def test_emulation_and_common_angles() -> None:
    import numpy as np

    r = _load("local_exp52_emulation_substitution.json")
    e, s = r["E"], r["S"]
    assert len(e["seeds"]) == 15 and e["n_coefficients_per_readout"] == 6561 and e["auc_equal_exact"]
    assert e["max_logit_err_exact"] < 3e-14 and round(e["max_logit_err_exact"] * 1e14, 1) == 2.6
    loss1 = [x["auc_circuit"] - x["k1"]["auc"] for x in e["seeds"]]
    loss2 = [x["auc_circuit"] - x["k2"]["auc"] for x in e["seeds"]]
    assert (round(float(np.mean(loss1)), 3), round(max(loss1), 3)) == (0.058, 0.215)
    assert (round(float(np.mean(loss2)), 3), round(max(loss2), 3)) == (0.011, 0.070)
    assert s["n_seeds"] == 15
    mean = {k: round(v, 3) for k, v in s["mean_auc"].items()}
    assert mean == {"ORIGINAL": 0.660, "CIRCUIT_NEW": 0.652, "LINEAR4D": 0.654, "FOURIER2": 0.664,
                    "FOURIER_FULL1": 0.668, "FOURIER_FULL2": 0.667}
    c = s["contrasts"]
    assert [round(c[f"CIRCUIT_NEW-{k}"]["p_cr2"], 3) for k in ("FOURIER2", "FOURIER_FULL1", "FOURIER_FULL2")] == [0.002, 0.001, 0.013]
    assert round(c["ORIGINAL-FOURIER_FULL1"]["p_cr2"], 3) == 0.035 and c["ORIGINAL-FOURIER_FULL1"]["ci_two_way"][1] > 0
    assert not c["CIRCUIT_NEW-LINEAR4D"]["resolved"] and not c["ORIGINAL-FOURIER2"]["resolved"]
    rows = table_rows(SUPP, "tab:common_angles")
    name = {"ORIGINAL": "Original", "CIRCUIT_NEW": "New circuit", "LINEAR4D": "Linear-4D", "FOURIER2": "Fourier-2",
            "FOURIER_FULL1": "Tensor-81", "FOURIER_FULL2": "Tensor-625"}
    for k, v in c.items():
        a, b = k.split("-")
        printed = [n for cell in rows[f"{name[a]} - {name[b]}"] for n in numbers(cell)]
        expected = [v["delta"], *v["ci_cr2"], v["p_cr2"], *v["ci_two_way"]]
        assert all(p.matches(x) for p, x in zip(printed, expected, strict=True)), k
    in_text(MAIN_TEXT, "(15-seed mean AUC 0.652)", "(0.664; CR2 $p=0.002$)", "(0.668 and 0.667; $p=0.001$ and 0.013)",
            "costs 0.058 AUC on average", "(CR2 $p=0.035$;")


# ------------------------------------------------------------------ exp53: external benchmark
def test_external_benchmark() -> None:
    e = _load("local_exp53_external_benchmark.json")
    assert e["n"] == {"train": 7669, "val": 2689, "test": 10282} and e["cells"] == {"train": 259, "val": 77, "test": 182}
    assert e["reference_population"] == {"cells": 39, "rows": 1170}
    assert e["predictions"] == {"P1": False, "P2": False, "P3": False, "P4": True, "P5": True}
    auc = [round(e["auc_all_test"][m], 3) for m in ("LR", "RF", "XGB", "MLP")]
    assert auc == [0.905, 0.921, 0.919, 0.914]
    rows = table_rows(SUPP, "tab:external")
    labels = {"RF - LR": "RF-LR", "XGB - RF": "XGB-RF", "MLP - RF": "MLP-RF"}
    ph = e["posthoc_all_cells"]
    assert ph["cells"] == 182 and round(ph["finite_population_factor"], 2) == 0.95
    for lab, key in labels.items():
        r, s, a = e["contrasts_reference"][key], e["small_tests"]["by_contrast"][key], e["contrasts_all_test"][key]
        p = ph["by_contrast"][key]
        assert p["reference"]["delta"] == pytest.approx(a["delta"], abs=1e-12)
        expected = [r["delta"], *r["ci_cr2"], s["resolved_cr2"], s["resolved_delong"], s["coverage_cr2"],
                    s["coverage_delong"], a["delta"], *a["ci_cr2"], p["coverage_cr2"], p["coverage_delong"]]
        printed = [n for c in rows[lab] for n in numbers(c)]
        assert len(printed) == len(expected), lab
        for i, (pr, ex) in enumerate(zip(printed, expected, strict=True)):
            assert pr.matches(ex), f"{lab} {i}: printed {pr.value}, archive {ex:.4f}"
    ratios = [v["se_cr2"] / v["se_delong"] for v in e["contrasts_all_test"].values()]
    assert (round(min(ratios), 1), round(max(ratios), 1)) == (4.4, 6.6)
    sm = e["small_tests"]
    assert sm["draws"] == 1000 and sm["rows_median"] == 570 and sm["rows_range"] == [423, 735]
    p5 = [v["mde_predicted"] / v["mde_empirical"] for v in sm["by_contrast"].values()]
    assert (round(min(p5), 2), round(max(p5), 2)) == (1.37, 1.40)
    cov_ph = {k: (v["coverage_cr2"], v["coverage_delong"]) for k, v in ph["by_contrast"].items()}
    assert (round(100 * min(c[1] for c in cov_ph.values())), round(100 * max(c[1] for c in cov_ph.values()))) == (41, 45)
    assert (round(100 * min(c[0] for c in cov_ph.values())), round(100 * max(c[0] for c in cov_ph.values()))) == (96, 98)
    in_text(MAIN_TEXT, "4.4--6.6 times the DeLong one", "61, 87 and 48\\,\\%", "only 55--63\\,\\% of draws",
            "CR2 covers in 94--99.8\\,\\%", "resolves in 0.2--10.5\\,\\%", "$\\nu=13.5$",
            "DeLong covers in 41--45\\,\\%, CR2 in 96--98\\,\\%", "in only 41--63\\,\\% of draws")
    assert "wrong sign in 16" not in MAIN_TEXT
    in_text(SUPP_TEXT, "1.37--1.40 times the empirical one")


# ------------------------------------------------------------------ exp50: repeated positive control
def test_teacher_grid() -> None:
    g = _load("local_exp50_teacher_grid.json")
    assert g["n_distinct"]["signal"] == 10 and g["n_distinct"]["null"] == 2 and g["selected_teacher"] == 0
    d = g["detection"]
    assert (d["LINEAR4D"]["circuit_above"], d["LINEAR4D"]["control_above"]) == (6, 0)
    assert (d["FOURIER2"]["circuit_above"], d["FOURIER2"]["control_above"]) == (0, 2)
    assert (d["FOURIER_FULL1"]["circuit_above"], d["FOURIER_FULL1"]["control_above"]) == (0, 10)
    assert all(v["null_detected"] == 0 for v in d.values())
    shares = [round(g["interaction_share"][str(c)], 2) for c in range(4)]
    assert shares == [0.83, 0.68, 0.57, 0.80]
    sig = [r for r in g["configs"].values() if r["teacher"] is not None]
    for head, (lo, hi) in {"CIRCUIT": (0.54, 0.75), "FOURIER2": (0.56, 0.76), "FOURIER_FULL1": (0.60, 0.79)}.items():
        vals = [r["auc"][head] for r in sig]
        assert (round(min(vals), 2), round(max(vals), 2)) == (lo, hi), head
    in_text(MAIN_TEXT, "above the Linear-4D map in 6 of the 10 configurations", "(which is above it in 2)",
            "below the 81-function tensor surrogate in all 10")
    # main-text Table 8: one row per distinct configuration, AUCs and markers from the archive
    rows = table_rows(TEX, "tab:teacher_main")
    uniq = {}
    for r in g["configs"].values():
        uniq.setdefault((r["teacher"], r["target"], r["draw"]), r)
    body = [k for k in rows if k.startswith(("Teacher", "Null"))]
    assert len(body) == 12
    heads = ("LINEAR4D", "FOURIER2", "FOURIER_FULL1")
    printed_cells = [rows[k] for k in body]
    expected_cells = []
    for r in sorted([r for r in uniq.values() if r["teacher"] is not None], key=lambda r: (r["teacher"], r["target"], r["draw"])) \
            + sorted([r for r in uniq.values() if r["teacher"] is None], key=lambda r: r["draw"]):
        expected_cells.append(r)
    assert len(expected_cells) == len(printed_cells) == 12
    marks_up = marks_down = 0
    for cells, r in zip(printed_cells, expected_cells, strict=True):
        nums = [n for c in cells for n in numbers(c)]
        expected = ([r["oracle_auc_test"]] if r["teacher"] is not None else []) + [r["auc"]["CIRCUIT"]] + [r["auc"][h] for h in heads]
        assert len(nums) == len(expected) and all(p.matches(x) for p, x in zip(nums, expected, strict=True)), cells
        for c, h in zip(cells[-3:], heads, strict=True):
            lo, hi = r["contrasts"][f"CIRCUIT-{h}"]["ci_cr2_bm"]
            assert ("downarrow" in c) == (lo > 0) and ("uparrow" in c and "downarrow" not in c) == (hi < 0), (c, h)
            marks_down += lo > 0; marks_up += hi < 0
    assert (marks_down, marks_up) == (6, 12)
    in_text(SUPP_TEXT, "0.83, 0.68, 0.57 and 0.80", "0.54 to 0.75 for the circuit, 0.56 to 0.76 for Fourier-2")


# ------------------------------------------------------------------ exp49
def test_within_cell_auc() -> None:
    w = M49["within_cell"]
    vals = [w[m]["within_cell_auc"] for m in ("RF_2273", "XGB", "PureLSTM", "CTRL_LINEAR4D", "LSTM_QNN")]
    assert (round(min(vals), 3), round(max(vals), 3)) == (0.449, 0.495)
    assert w["CLIM"]["within_cell_auc"] == pytest.approx(0.5) and round(w["CLIM"]["pooled_auc"], 3) == 0.685
    inc = [m for m in ("RF_2273", "XGB", "PureLSTM", "CTRL_LINEAR4D") if w[m]["ci_block"][0] <= 0.5 <= w[m]["ci_block"][1]]
    assert len(inc) == 4 and w["LSTM_QNN"]["ci_block"][1] < 0.5
    assert [round(x, 3) for x in w["LSTM_QNN"]["ci_block"]] == [0.423, 0.497]
    in_text(MAIN_TEXT, "Every model scores 0.449--0.495", "$[0.423,0.497]$")


def test_seas5_increment() -> None:
    s = M49["seas5_increment"]
    models = ("RF_2273", "XGB", "PureLSTM", "CTRL_LINEAR4D", "LSTM_QNN")
    for lead, (lo, hi) in (("L2", (-0.008, -0.001)), ("L1", (-0.003, 0.001))):
        d = [s[lead][m]["delta_auc"] for m in models]
        assert (round(min(d), 3), round(max(d), 3)) == (lo, hi), lead
        assert all(s[lead][m]["ci_block_auc"][0] <= 0 <= s[lead][m]["ci_block_auc"][1] for m in models)
        assert all(s[lead][m]["delta_brier"] > -0.001 for m in models)
    assert min(s[lead][m]["p_delong"] for lead in ("L1", "L2") for m in models) >= 0.31
    in_text(MAIN_TEXT, "by $-0.008$ to $-0.001$ at lead 2", "by $-0.003$ to $+0.001$ at lead 1", "DeLong $p\\ge0.31$")


def test_analytic_auc() -> None:
    a = M49["analytic_auc"]
    got = {k: round(a[k]["by_design_delta"]["0.03"]["delta"], 4) for k in "BCDE"}
    assert got == {"B": 0.0290, "C": 0.0280, "D": 0.0280, "E": 0.0238}
    for k in "BCDE":
        assert abs(got[k] - a[k]["reported_mc"]) < 0.001
    in_text(SUPP_TEXT, "0.0290 (B), 0.0280 (C and D) and 0.0238 (E)")
