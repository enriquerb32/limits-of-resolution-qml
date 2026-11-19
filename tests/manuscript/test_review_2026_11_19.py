"""Numbers added in the revision after the review of 2026-11-19 must follow from the archived results.

Covers the corrected few-cluster statement, the period of the P60 threshold, the causal-pipeline primary families
(exp53 A), the CR2 resolution of the second panel (exp53 B), the SEAS5 contrasts under the primary procedure (exp53 C)
and the simulation of clusters against rows (exp54).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _latex import numbers, table_rows

ROOT = Path(__file__).resolve().parents[2]
TEX, SUPP, RES = ROOT / "paper" / "main.tex", ROOT / "paper" / "supp.tex", ROOT / "results"
MAIN_TEXT = TEX.read_text(encoding="utf-8")
SUPP_TEXT = SUPP.read_text(encoding="utf-8")


def _load(name: str) -> Any:
    return json.loads((RES / name).read_text(encoding="utf-8"))


E53 = _load("local_exp55_review_closing.json")
E54 = _load("local_exp54_power_clusters_rows.json")
C48 = _load("local_exp48_primary_calibration.json")


def in_text(text: str, *claims: str) -> None:
    for c in claims:
        assert c in text, c


def r3(x: float) -> str:
    return f"{x:.3f}"


def rng(values, digits: int = 3) -> str:
    return f"{min(values):.{digits}f}--{max(values):.{digits}f}"


# ------------------------------------------------------------------ point 3 of the review
def test_ratio_shares_in_main_text() -> None:
    k19 = [r for r in C48["rows"] if r["mask"] in ("real", "m2", "m4")]
    assert round(100 * max(r["p_ratio_le_0_59"] for r in k19), 1) == 1.2
    assert round(100 * max(r["p_ratio_le_0_72"] for r in k19), 1) == 5.8
    in_text(MAIN_TEXT, "as low as 0.59 in more than 1.2\\,\\% of replications, or as low as 0.72 in more than 5.8\\,\\%")
    assert "as low as 0.60 or 0.72 in more than 1.2" not in MAIN_TEXT


def test_two_station_subset_holm_statement() -> None:
    two = _load("local_two_station_subset.json")["pairs"]
    assert two["RF-LSTM_QNN"]["p_cr2"] < 0.05 / 3 < _load("local_exp42_v2_summary.json")["pairs"]["RF-LSTM_QNN"]["p_cr2"]
    in_text(MAIN_TEXT, "the second $p$-value would also pass the first Holm step, which it does not on the full panel")


def test_p60_period() -> None:
    p = pd.read_csv(RES / "panel_v2_monthly.csv", parse_dates=["fecha"])
    lab = p[(p["label_ok"].astype(str).str.lower() == "true") & (p["fecha"].dt.year <= 1996)]
    assert len(lab) == 2847 and round(float(np.percentile(lab["precip_label"], 60)), 1) == 29.9
    first = lab.groupby("ciudad")["fecha"].min().dt.strftime("%Y-%m").to_dict()
    assert first == {"Alicante": "1938-10", "Barcelona": "1924-03", "San_Javier": "1944-08", "Valencia": "1937-11"}
    assert lab.groupby("ciudad").size().to_dict() == {"Alicante": 699, "Barcelona": 809, "San_Javier": 629, "Valencia": 710}
    in_text(MAIN_TEXT, "the pooled 60th percentile of the 2{,}847 labelled station-months up to 1996",
            "in 1924 at Barcelona and in 1937--1944 at the other stations")
    in_text(SUPP_TEXT, "Barcelona March 1924, Valencia November 1937, Alicante October 1938, San Javier August 1944",
            "all from data up to 1996")
    assert "all from 1938--1996" not in SUPP_TEXT


# ------------------------------------------------------------------ exp53 B: second panel, CR2 MDE
PANEL2 = {"RF - XGBoost": "RF_2273-XGB", "RF - LSTM": "RF_2273-PureLSTM", "RF - LSTM+QNN": "RF_2273-LSTM_QNN",
          "XGBoost - LSTM+QNN": "XGB-LSTM_QNN", "LSTM - LSTM+QNN": "PureLSTM-LSTM_QNN",
          "Linear-4D - LSTM+QNN": "CTRL_LINEAR4D-LSTM_QNN", "Fourier-2 - LSTM+QNN": "FOURIER2-LSTM_QNN"}


def test_panel2_cr2_mde_claims() -> None:
    b = E53["panel2_mde"]
    assert (b["n_first"], b["n_second"], b["years_first"], b["years_second"]) == (529, 884, 19, 19)
    complete = [v for k, v in b["pairs"].items() if not k.startswith("FOURIER2")]
    assert len(complete) == 6
    assert rng([v["mde_cr2_second"] for v in complete]) == "0.019--0.035"
    assert rng([v["ratio_observed_predicted"] for v in complete], 2) == "1.36--1.67"
    assert all(v["mde_cr2_second"] > v["mde_cr2_first"] for v in complete)
    assert r3(b["pairs"]["FOURIER2-LSTM_QNN"]["mde_cr2_second"]) == "0.006"
    in_text(MAIN_TEXT, "The CR2 MDEs of the six comparisons between complete models are 0.019--0.035 on the second panel, "
                       "1.36--1.67 times their $N^{-1/2}$ predictions", "only that of \\texttt{Fourier-2} against the hybrid (0.006)")


def test_panel2_table_cr2_mde_column() -> None:
    rows = {k: v for k, v in table_rows(TEX, "tab:panel2").items() if k != "Comparison"}
    assert set(rows) == set(PANEL2)
    for label, key in PANEL2.items():
        v = E53["panel2_mde"]["pairs"][key]
        cell = numbers(rows[label][6])            # "MDE CR2 (pred.)" column
        assert [c.value for c in cell] == [round(v["mde_cr2_second"], 3), round(v["mde_cr2_predicted"], 3)], label


# ------------------------------------------------------------------ exp53 C: SEAS5
def test_seas5_text_claims() -> None:
    c = E53["seas5"]
    inc2 = [v for k, v in c["increment"]["L2"].items() if isinstance(v, dict)]
    inc1 = [v for k, v in c["increment"]["L1"].items() if isinstance(v, dict)]
    assert not any(v["resolved"] for v in inc1 + inc2)
    assert min(v["p_cr2"] for v in inc2) >= 0.25 and min(v["p_cr2"] for v in inc1) >= 0.62
    assert rng([v["mde_cr2"] for v in inc2]) == "0.010--0.019"
    ref = c["reference"]
    assert [round(x, 3) for x in ref["SEAS5_QM_L2-CLIM"]["ci_cr2"]] == [-0.038, 0.019]
    assert (round(ref["RF_2273-SEAS5_QM_L2"]["delta"], 3), round(ref["LSTM_QNN-SEAS5_QM_L2"]["delta"], 3)) == (0.003, -0.017)
    assert (round(ref["RF_2273-SEAS5_QM_L2"]["p_cr2"], 2), round(ref["LSTM_QNN-SEAS5_QM_L2"]["p_cr2"], 2)) == (0.84, 0.25)
    in_text(MAIN_TEXT, "SEAS5 is not distinguished from the climatology ($-0.010$, CR2 $[-0.038,+0.019]$)",
            "($+0.003$ and $-0.017$; CR2 $p=0.84$ and 0.25", "CR2 $p\\ge0.25$ at lead 2 and $p\\ge0.62$ at lead 1",
            "an increment of 0.010--0.019 AUC at lead 2", "We did not detect incremental skill")
    assert "The statistical models add nothing to it" not in MAIN_TEXT


def _check_rows(rows: dict, mapping: dict, entry) -> None:
    for label, key in mapping.items():
        expected = entry(key)
        got = [c.value for cell in rows[label] for c in numbers(cell)]
        assert len(got) == len(expected), (label, got, expected)
        for g, (e, d) in zip(got, expected):
            assert round(e, d) == g, (label, g, e)


def test_reference_table() -> None:
    e42 = _load("local_exp42_v2_summary.json")["pairs"]
    pm = _load("local_primary_mde.json")["pairs"]
    s33 = _load("local_exp33_seas5_orig.json")["pairs"]
    e49 = _load("local_exp49_review_minor.json")["seas5_increment"]
    ref, inc = E53["seas5"]["reference"], E53["seas5"]["increment"]
    rows = {k: v for k, v in table_rows(TEX, "tab:seas5").items() if k != "Comparison"}
    clim = {"Random Forest - climatology": "RF-CLIM", "XGBoost - climatology": "XGB-CLIM",
            "Standalone LSTM - climatology": "PureLSTM-CLIM", "Linear-4D control - climatology": "CTRL_LINEAR4D-CLIM",
            "LSTM+QNN - climatology": "LSTM_QNN-CLIM"}
    _check_rows(rows, clim, lambda k: [(e42[k]["gap"], 4), (e42[k]["ci_cr2"][0], 3), (e42[k]["ci_cr2"][1], 3),
                                       (e42[k]["p_cr2"], 3), (e42[k]["p_delong"], 3), (e42[k]["ci_block"][0], 3),
                                       (e42[k]["ci_block"][1], 3), (pm[k]["mde_cr2"], 3)])
    for L in (2, 1):
        sea = {f"SEAS5 - climatology (L{L})": f"SEAS5_QM_L{L}-CLIM", f"Random Forest - SEAS5 (L{L})": f"RF_2273-SEAS5_QM_L{L}",
               f"LSTM+QNN - SEAS5 (L{L})": f"LSTM_QNN-SEAS5_QM_L{L}"}
        _check_rows(rows, sea, lambda k: [(ref[k]["delta"], 4), (ref[k]["ci_cr2"][0], 3), (ref[k]["ci_cr2"][1], 3),
                                          (ref[k]["p_cr2"], 3), (s33[k]["p_delong"], 3), (s33[k]["ci_block"][0], 3),
                                          (s33[k]["ci_block"][1], 3), (ref[k]["mde_cr2"], 3)])
    names = {"Random Forest": "RF_2273", "XGBoost": "XGB", "Standalone LSTM": "PureLSTM",
             "Linear-4D control": "CTRL_LINEAR4D", "LSTM+QNN": "LSTM_QNN"}
    for L, path, table in ((2, TEX, "tab:seas5"), (1, SUPP, "tab:seas5_blends")):
        rr = {k: v for k, v in table_rows(path, table).items() if k != "Comparison"}
        _check_rows(rr, {f"Increment {n} (L{L})": m for n, m in names.items()},
                    lambda m: [(inc[f"L{L}"][m]["delta"], 4), (inc[f"L{L}"][m]["ci_cr2"][0], 3),
                               (inc[f"L{L}"][m]["ci_cr2"][1], 3), (inc[f"L{L}"][m]["p_cr2"], 3),
                               (e49[f"L{L}"][m]["p_delong"], 3), (e49[f"L{L}"][m]["ci_block_auc"][0], 3),
                               (e49[f"L{L}"][m]["ci_block_auc"][1], 3), (inc[f"L{L}"][m]["mde_cr2"], 3)])


def test_replacement_blends_in_supplement() -> None:
    s33 = _load("local_exp33_seas5_orig.json")["pairs"]
    rep = E53["seas5"]["replacement"]
    rows = {k: v for k, v in table_rows(SUPP, "tab:seas5_blends").items() if k != "Comparison"}
    blends = {"Blend RF - blend climatology": "BLEND_RF_2273_L1-BLEND_CLIM_L1",
              "Blend XGBoost - blend climatology": "BLEND_XGB_L1-BLEND_CLIM_L1",
              "Blend LSTM - blend climatology": "BLEND_PureLSTM_L1-BLEND_CLIM_L1",
              "Blend Linear-4D - blend climatology": "BLEND_CTRL_LINEAR4D_L1-BLEND_CLIM_L1",
              "Blend LSTM+QNN - blend climatology": "BLEND_LSTM_QNN_L1-BLEND_CLIM_L1",
              "Blend LSTM+QNN - blend Linear-4D": "BLEND_LSTM_QNN_L1-BLEND_CTRL_LINEAR4D_L1"}
    for k, key in blends.items():
        assert abs(rep[key]["delta"] - s33[key]["delta"]) < 1e-9, key
    _check_rows(rows, blends, lambda k: [(rep[k]["delta"], 4), (rep[k]["ci_cr2"][0], 3), (rep[k]["ci_cr2"][1], 3),
                                         (rep[k]["p_cr2"], 3), (s33[k]["p_delong"], 3), (s33[k]["ci_block"][0], 3),
                                         (s33[k]["ci_block"][1], 3), (rep[k]["mde_cr2"], 3)])


# ------------------------------------------------------------------ exp53 A: causal pipeline
def test_causal_primary_families() -> None:
    a = E53["causal"]
    f = a["families"]
    arc = _load("local_exp42_v2_summary.json")["pairs"]
    for fam in ("classical", "hybrid"):                       # the archived column reproduces the primary analysis
        for k, v in f["archived_529"][fam].items():
            assert abs(v["p_cr2"] - arc[k]["p_cr2"]) < 1e-9, k
    cc, own = f["causal_common_529"], f["causal_own_533"]
    assert (a["n_causal"], a["n_common"]) == (533, 529)
    for name in ("causal_common_529", "causal_own_533"):
        res = {k: v["resolved"] for fam in f[name].values() for k, v in fam.items()}
        assert res == {"RF-XGB": False, "RF-PureLSTM": False, "XGB-PureLSTM": False, "RF-LSTM_QNN": True,
                       "LSTM_QNN-CTRL_LINEAR4D": False, "PureLSTM-LSTM_QNN": False}, name
        assert f[name]["hybrid"]["RF-LSTM_QNN"]["resolved_holm"]
    assert r3(cc["classical"]["RF-PureLSTM"]["p_cr2"]) == "0.070" and r3(cc["hybrid"]["RF-LSTM_QNN"]["p_cr2"]) == "0.009"
    assert r3(a["blocks"]["causal_common_529"]["RF-LSTM_QNN"]["3yr"]) == "0.076"
    deltas = [abs(a["refit_linear4d"]["delta_archived_minus_causal"])]
    deltas += [abs(v["delta_archived_minus_causal"]) for v in _load("local_exp47_truncation_audit.json")["refit_neural"].values()]
    deltas += [abs(v["delta_archived_minus_causal"]) for v in _load("local_exp47_truncation_audit.json")["refit_trees"].values() if isinstance(v, dict)]
    assert max(deltas) < 0.0025                               # "changes their test AUC by at most 0.002"
    in_text(MAIN_TEXT, "Random Forest against the LSTM is no longer resolved (CR2 $p=0.070$",
            "passes the Holm correction ($p=0.009$)", "though not with clusters of three consecutive years ($p=0.076$")
    in_text(SUPP_TEXT, "\\label{tab:causal}")


# ------------------------------------------------------------------ exp54: clusters against rows
def _fit(sc: str) -> tuple[float, float]:
    rows = [r for r in E54["rows"] if r["scenario"] == sc and r["delta"] == 0.02]
    X = np.array([[1 / (r["K"] * r["m"]), 1 / r["K"]] for r in rows])
    v = np.array([r["sd_delta_hat"] ** 2 for r in rows])
    (a, b), *_ = np.linalg.lstsq(X, v, rcond=None)
    return float(a), float(b)


def test_clusters_rows_claims() -> None:
    rows = E54["rows"]
    assert len(rows) == 128 and E54["reps"] == 1000
    assert rng([r["coverage"]["cr2_bm"] for r in rows]) == "0.930--0.967"
    null = [r for r in rows if r["delta"] == 0]
    assert rng([100 * r["rejection"]["cr2_bm"] for r in null], 1) == "3.3--7.0"
    assert round(100 * max(r["rejection"]["delong"] for r in null if r["scenario"] == "station_year"), 1) == 17.6
    a, b = _fit("station_year")
    assert round(a / b, -1) == 110
    others = [_fit(s) for s in ("specific", "specific_ar1")]
    assert (round(min(x / y for x, y in others), -1), round(max(x / y for x, y in others), -1)) == (290, 460)
    assert _fit("iid")[1] <= 0 or _fit("iid")[0] / _fit("iid")[1] > 1000      # no cluster term with independent cases
    in_text(MAIN_TEXT, "$a/b$, is about 110 with year$\\times$station effects and 290--460",
            "CR2 covers at 0.930--0.967 and rejects a null difference in 3.3--7.0\\,\\% of replications in all 128 cells",
            "DeLong rejects it in up to 17.6\\,\\%", "Independent information, not row count, sets the resolution.")
    assert "Clusters, not rows" not in MAIN_TEXT and "not of rows" not in MAIN_TEXT


def test_seas5_retrieval_date() -> None:
    in_text(MAIN_TEXT, "(data retrieved through the CDS API on 9--10 November 2026)")
    in_text(SUPP_TEXT, "was downloaded through the CDS API on 9--10 November 2026")
    assert "add the date on which the SEAS5 fields were retrieved" not in MAIN_TEXT
