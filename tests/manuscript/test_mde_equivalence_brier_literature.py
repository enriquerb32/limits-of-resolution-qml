"""Numbers in the manuscript must follow from the archived results.

Covers the uncertainty of the CR2 MDE and the equivalence tests at the circuit interface (exp57), the tables kept
in the supplement, the summary of verdicts, the Brier inference, the literature reanalysis and the cancellation
simulation.
"""

from __future__ import annotations

from typing import Any

from _archive import MAIN_TEXT, RES, ROOT, SUPP, SUPP_TEXT, TEX, in_text, load
from _latex import numbers, table_rows

E57 = load("local_exp57_mde_ci_equivalence.json")

MDE_ROWS = {
    "RF - XGBoost": "RF-XGB",
    "RF - LSTM": "RF-PureLSTM",
    "XGBoost - LSTM": "XGB-PureLSTM",
    "RF - LSTM+QNN": "RF-LSTM_QNN",
    "Linear-4D - LSTM+QNN": "CTRL_LINEAR4D-LSTM_QNN",
    "LSTM - LSTM+QNN": "PureLSTM-LSTM_QNN",
    "Fourier-2 - LSTM+QNN": "FOURIER2-LSTM_QNN",
    "Full Fourier-1 - LSTM+QNN": "FOURIER_FULL1-LSTM_QNN",
    "Full Fourier-2 - LSTM+QNN": "FOURIER_FULL2-LSTM_QNN",
    "LSTM+QTL - QTL-classical": "QTL_FIXED-QTL_CLASSICAL",
    "LSTM+QTL - QTL tensor-81": "QTL_FIXED-QTL_TENSOR1",
}
TOST_ROWS = {k: v for k, v in MDE_ROWS.items() if v in E57["tost"]}


def _check(cells: list[str], expected: list[float], label: str) -> None:
    printed = [n for c in cells for n in numbers(c)]
    assert len(printed) == len(expected), f"{label}: {cells}"
    for p, e in zip(printed, expected, strict=True):
        assert p.matches(e), f"{label}: printed {p.value}, archived {e}"


def test_mde_ci_table() -> None:
    rows = {k: v for k, v in table_rows(SUPP, "tab:mde_ci").items() if k != "Difference"}
    assert set(rows) == set(MDE_ROWS)
    for label, key in MDE_ROWS.items():
        v = E57["mde_ci"][key]
        _check(rows[label], [v["delta"], v["mde_cr2"], *v["ci95_chi2"], *v["ci95_year_boot"]], label)


def test_tost_table() -> None:
    rows = {k: v for k, v in table_rows(SUPP, "tab:tost").items() if k != "Difference"}
    assert set(rows) == set(TOST_ROWS) and E57["margin"] == 0.01
    for label, key in TOST_ROWS.items():
        v = E57["tost"][key]
        _check(rows[label], [v["delta"], *v["ci90_cr2"], v["p_tost"], v["smallest_margin"]], label)
        assert v["equivalent"] == (key == "QTL_FIXED-QTL_CLASSICAL"), key


def test_mde_ci_and_tost_in_text() -> None:
    m = E57["mde_ci"]
    ratios = [r for v in m.values() for r in (v["ci95_chi2"][0] / v["mde_cr2"], v["ci95_chi2"][1] / v["mde_cr2"])]
    assert (round(min(ratios), 2), round(max(ratios), 2)) == (0.74, 1.52)
    boot = [
        r for v in m.values() for r in (v["ci95_year_boot"][0] / v["mde_cr2"], v["ci95_year_boot"][1] / v["mde_cr2"])
    ]
    assert (round(min(boot), 2), round(max(boot), 2)) == (0.45, 1.44)
    pw = m["RF-LSTM_QNN"]["power_at_mde_if_se_at_chi2_bounds"]
    assert (round(pw[0], 2), round(pw[1], 2)) == (0.44, 0.96)
    in_text(
        MAIN_TEXT,
        "for Random Forest against \\texttt{LSTM+QNN} 0.025 $[0.018,0.038]$",
        "gives $[0.011,0.035]$",
        "between 0.44 and 0.96",
        "from about 0.7 to 1.5 times its point value",
    )
    in_text(SUPP_TEXT, "0.74--1.52 times the point MDE", "0.45--1.44 times the point MDE", "would be 0.44 or 0.96")
    qnn = [v for k, v in E57["tost"].items() if k.endswith("LSTM_QNN")]
    assert (round(min(v["p_tost"] for v in qnn), 2), round(max(v["p_tost"] for v in qnn), 2)) == (0.11, 0.39)
    assert (round(min(v["smallest_margin"] for v in qnn), 3), round(max(v["smallest_margin"] for v in qnn), 3)) == (
        0.013,
        0.018,
    )
    assert not any(v["equivalent"] for v in qnn)
    in_text(
        MAIN_TEXT,
        "(TOST $p=0.11$--0.39)",
        "the smallest margins the data would accept are 0.013--0.018 AUC",
        "(TOST $p=0.047$; post hoc, unadjusted",
    )


def test_moved_tables_and_summary() -> None:
    for label in ("tab:case", "tab:teacher_main", "tab:station_coverage"):
        assert f"\\label{{{label}}}" not in MAIN_TEXT, label
    for label in (
        "tab:case_all",
        "tab:teacher_main",
        "tab:station_coverage",
        "tab:mde_ci",
        "tab:tost",
        "app:noise_budget",
    ):
        assert f"\\label{{{label}}}" in SUPP_TEXT, label
    assert "\\appendix" not in MAIN_TEXT
    rows = table_rows(TEX, "tab:verdicts")
    assert (
        len([k for k in rows if k != "Statement"]) == 11
    )  # the frozen-circuit row split by statement (models vs procedure)
    status = {k: v[-1] for k, v in rows.items() if k != "Statement"}
    assert sum("Exploratory" in s for s in status.values()) == 1
    assert status["Training the circuit helps (a statement about the procedure)"] == "Not resolved"
    assert status["The 15 trained circuits score above the 15 frozen ones"] == "Resolved for these models (secondary)"
    in_text(
        MAIN_TEXT,
        "Which interval decides which statement",
        "is not resolved (\\Cref{sec:multiplicity})",
        "an interval that resamples both seeds and test clusters excludes zero",
        "just below the 0.78--0.91 measured between the models of this paper",
    )
    in_text(SUPP_TEXT, "Pre-Analysis Plan Template")
    assert (ROOT / "docs" / "pre_analysis_plan_template.md").exists()
    in_text(
        MAIN_TEXT,
        "not a new estimator but a design-and-reporting workflow",
        "so it is a re-test, not an independent replication",
        "a test of coverage and of false decisions, not of power",
    )


# ------------------------------------------------------------------ Brier score, interface Holm, OpenData, literature
E58 = load("local_exp58_brier_inference.json")
BRIER_ROWS = {
    "RF - XGBoost": ("classical", "RF-XGB"),
    "RF - LSTM": ("classical", "RF-PureLSTM"),
    "XGBoost - LSTM": ("classical", "XGB-PureLSTM"),
    "RF - LSTM+QNN": ("hybrid", "RF-LSTM_QNN"),
    "LSTM+QNN - Linear-4D": ("hybrid", "LSTM_QNN-CTRL_LINEAR4D"),
    "LSTM - LSTM+QNN": ("hybrid", "PureLSTM-LSTM_QNN"),
    "Fourier-2 - LSTM+QNN": ("interface", "FOURIER2-LSTM_QNN"),
    "Full Fourier-1 - LSTM+QNN": ("interface", "FOURIER_FULL1-LSTM_QNN"),
    "Full Fourier-2 - LSTM+QNN": ("interface", "FOURIER_FULL2-LSTM_QNN"),
    **{
        f"{n} - climatology": ("climatology", f"{k}-CLIM")
        for n, k in [
            ("RF", "RF"),
            ("XGBoost", "XGB"),
            ("LSTM", "PureLSTM"),
            ("Linear-4D", "CTRL_LINEAR4D"),
            ("LSTM+QNN", "LSTM_QNN"),
        ]
    },
}


def _brier_entry(part: str, key: str, recal: bool) -> dict[str, Any] | None:
    if part == "interface":
        return None if recal else E58["interface"][key]
    if part == "climatology":
        return (E58["recalibrated"]["climatology"] if recal else E58["climatology"])[key]
    return (E58["recalibrated"][part] if recal else E58["families"][part])[key]


def test_brier_inference_table() -> None:
    rows = {k: v for k, v in table_rows(SUPP, "tab:brier_inference").items() if k in BRIER_ROWS}
    assert set(rows) == set(BRIER_ROWS)
    for label, (part, key) in BRIER_ROWS.items():
        cells = rows[label]
        for j, recal in ((0, False), (3, True)):
            v = _brier_entry(part, key, recal)
            if v is None:
                assert cells[j : j + 3] == ["--"] * 3, label
                continue
            p = [v["p_cr2"]]  # "<0.001" is parsed as the bound 0.001
            if round(v["p_cr2"], 3) == 0:
                assert "<0.001" in cells[j + 1], label
                p = [0.001]
            _check(cells[j : j + 3], [v["delta"], *v["ci_cr2"], *p, v["mde_cr2"]], f"{label} recal={recal}")
            assert ("$^{*}$" in cells[j + 1] or "$^{**}$" in cells[j + 1]) == v["resolved"], label
            if "resolved_after_holm" in v:
                assert ("$^{**}$" in cells[j + 1]) == v["resolved_after_holm"], label


def test_brier_claims_in_text() -> None:
    rc = E58["recalibrated"]
    assert not any(v["resolved_after_holm"] for f in ("classical", "hybrid") for v in rc[f].values())
    q = rc["hybrid"]["LSTM_QNN-CTRL_LINEAR4D"]
    assert round(q["p_cr2"], 2) == 0.44 and not q["resolved"]
    raw = E58["families"]["hybrid"]["LSTM_QNN-CTRL_LINEAR4D"]
    assert round(raw["delta"], 4) == 0.0046 and round(raw["p_cr2"], 3) == 0.016 and raw["resolved_after_holm"]
    worse = {k for k, v in rc["climatology"].items() if v["ci_cr2"][0] > 0}
    assert worse == {"PureLSTM-CLIM", "CTRL_LINEAR4D-CLIM", "LSTM_QNN-CLIM"}
    ps = [rc["climatology"][k]["p_cr2"] for k in worse]
    assert (round(min(ps), 3), round(max(ps), 3)) == (0.010, 0.028)
    in_text(MAIN_TEXT, "excludes zero ($p=0.010$--0.028)")
    in_text(SUPP_TEXT, "(worse by 0.0046)", "($+0.0012$, $p=0.44$)")  # in S


def test_interface_holm_sensitivity() -> None:
    """Holm over the four interface contrasts of LSTM+QNN, from the CR2 p-values printed in Tables 4 and 5."""
    rows = {**table_rows(TEX, "tab:qnn_paired"), **table_rows(SUPP, "tab:controls")}
    keys = ["Linear-4D - LSTM+QNN", "Fourier-2 - LSTM+QNN", "Full Fourier-1 - LSTM+QNN", "Full Fourier-2 - LSTM+QNN"]
    ps = [numbers(rows[k][3])[-1].value for k in keys]
    assert min(ps) == 0.158 and min(ps) > 0.05 / len(ps)
    in_text(MAIN_TEXT, "resolves none of them (smallest CR2 $p=0.158$)")


def test_aemet_opendata_pressure_check() -> None:
    import csv

    rows = list(csv.DictReader((RES / "local_exp59_aemet_opendata_pressure.csv").open(encoding="utf-8")))
    by = {(r["ciudad"], int(r["year"])): r for r in rows}
    for (st, y), r in by.items():
        if 2008 <= y <= 2025:  # 2026: the panel was downloaded on 1 June, before the last days were released
            assert float(r["pressure_days_opendata"]) == float(r["pressure_days_panel"]), (st, y)
            if r["max_abs_diff_pmax"]:
                assert float(r["max_abs_diff_pmax"]) == 0.0 and float(r["max_abs_diff_pmin"]) == 0.0, (st, y)
    assert all(float(by[("San_Javier", y)]["pressure_days_opendata"]) == 0 for y in range(2011, 2026))
    assert all(float(by[("Valencia", y)]["pressure_days_opendata"]) == 0 for y in range(2013, 2026))
    assert float(by[("Valencia", 2012)]["pressure_days_opendata"]) == 60
    in_text(MAIN_TEXT, "agree exactly with the AEMET OpenData service")


def test_literature_table() -> None:
    e60 = load("local_exp60_literature_reanalysis.json")
    printed = [
        r
        for r in MAIN_TEXT.split("\\label{tab:literature}")[1].split("\\bottomrule")[0].splitlines()
        if r.startswith("\\citet")
    ]
    assert len(printed) == len(e60["contrasts"]) == 8
    for line, c in zip(printed, e60["contrasts"], strict=True):
        cells = [x.strip() for x in line.rstrip("\\ ").split("&")]
        assert numbers(cells[4])[0].matches(c["delta"]), line
        ci = numbers(cells[5])
        assert ci[0].matches(c["ci95_0.75"][0]) and ci[1].matches(c["ci95_0.75"][1]), line
        mg = numbers(cells[6])
        assert mg[0].matches(c["smallest_margin_0.75"]) and mg[1].matches(c["smallest_margin_0.5"]), line
        assert numbers(cells[7])[0].matches(c["mde_pooled"]["0.75"]), line
    assert (e60["n_similarity"], e60["n_contrasts"], e60["n_studies"]) == (6, 8, 5)
    lo, hi = e60["margin_range_similarity"]
    lo5, hi5 = e60["margin_range_similarity_rho05"]
    assert (round(lo, 3), round(hi, 3), round(lo5, 3), round(hi5, 3)) == (0.010, 0.040, 0.015, 0.046)
    assert e60["similarity_ci_excludes_zero_rho075"] == ["wu2021_tth", "grossi2022"]
    assert e60["similarity_ci_excludes_zero_rho05"] == [] and e60["superiority_ci_excludes_zero_both"]
    n_class = e60["n_studies"] + len(e60["screened"])  # Terashi's VQC row (MDE only) is the same study as its QCL entry
    assert n_class == 14 and n_class + len(e60["regression_only"]) == 18
    sim = [c for c in e60["contrasts"] if c["key"] not in ("heredge2021", "kulkarni2023")]
    worst = [c["smallest_margin_worst_rounding_0.75"] for c in sim]
    assert (round(min(worst), 3), round(max(worst), 3)) == (0.020, 0.046)
    cons = e60["consistency"]
    assert all(abs(x - 0.042) <= 0.0006 for x in cons["wu2021"]["hm_se_one_auc_one_set"])  # 0.0419 and 0.0425
    assert [round(x, 3) for x in cons["belis2021"]["hm_se_one_auc_one_set"]] == [0.02, 0.02]
    assert round(cons["heredge2021"]["hm_se_one_auc_one_set"][0], 3) == 0.004
    assert round(cons["heredge2021"]["printed_spread"][0], 3) == 0.003
    assert round(cons["grossi2022"]["hm_se_one_auc_one_set"][0], 3) == 0.013
    h = next(c for c in e60["contrasts"] if c["key"] == "heredge2021")
    one_set_se = h["se_0.75"] * (h["n_sets"] ** 0.5)
    assert h["delta"] - 1.96 * one_set_se > 0  # "with one test set of 10,000 the interval still excludes zero"
    in_text(
        SUPP_TEXT,
        "Wu et al.\\ give standard deviations of 0.03--0.06",
        "against a Hanley--McNeil standard error of 0.042",
        "against 0.020 and 0.013",
        "implies 0.003 per test set of 10{,}000, against 0.004",
        "Two studies were checked against their published versions",
        "its AUC tables could not be extracted",
        "whether the trials share one test set is not stated",
        "margins of $\\pm0.020$ to $\\pm0.046$",
        "with one test set of 10{,}000 the interval still excludes zero",
    )
    marks = {"wu2021": "X", "belis2021": "P", "grossi2022": "X", "heredge2021": "P", "kulkarni2023": "X"}
    for line, c in zip(printed, e60["contrasts"], strict=True):
        key = c["key"].split("_")[0]
        assert line.startswith(f"\\citet{{{key}}}$^{{\\mathrm{{{marks[key]}}}}}$"), line
        assert c["checked_against"].startswith("published") == (marks[key] == "P"), key
    abstract = MAIN_TEXT.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    assert "imply unstated margins up to $\\pm0.04$ AUC" in abstract
    in_text(
        MAIN_TEXT,
        "the most favourable reading for a claim of similarity, judged by its margin, and the least conservative "
        "for excluding zero",
        "four of the five studies report per-model uncertainties",
        "and $\\pm0.020$ to $\\pm0.046$ in the worst case of rounding",
    )
    w = next(c for c in e60["contrasts"] if c["key"] == "wu2021_tth")
    w_one = w["se_0.75"] * (w["n_sets"] ** 0.5)
    assert (
        w["delta"] - 1.96 * w_one < 0 < w["delta"] + 1.96 * w_one
    )  # "with one test set of 100 the ttH interval would include zero"
    in_text(
        SUPP_TEXT,
        "within the printed range (Wu et al.) or slightly above it (Heredge et al.)",
        "with one test set of 100 the ttH interval would include zero",
    )
    in_text(
        MAIN_TEXT,
        "margins of $\\pm0.010$ to $\\pm0.040$ AUC ($\\pm0.015$ to $\\pm0.046$ with a correlation of 0.5",
        "In two of them the interval excludes zero in favour of the classical model at a correlation of 0.75",
        "we screened 18 QML studies",
        "of the 14 binary-classification studies, 5 (four of them in particle physics)",
        "equivalence margins of up to $\\pm0.04$ AUC",
        "none of the eight comparisons reports an interval",
    )
    assert "6 of 8" not in MAIN_TEXT and "none reports it" not in MAIN_TEXT
    in_text(
        SUPP_TEXT, "illustrative and the search was targeted, not systematic", "No bibliographic database was searched"
    )
    assert SUPP_TEXT.split("\\label{tab:literature_screened}")[1].split("\\bottomrule")[0].count("arXiv") == len(
        e60["screened"]
    )


# ------------------------------------------------------------------ within-year cancellation (exp61, exp62)
def test_cancellation_simulation_claims() -> None:
    e61 = load("local_exp61_cancellation_calibration.json")
    e62 = load("local_exp62_floored_cr2_observed.json")
    assert round(e61["icc_lower_bound"], 3) == -0.037 and round(e61["icc_implied"]["RF-PureLSTM"], 3) == -0.024
    s = e61["search"]
    assert len(s) == 13 and max(x["p_le_0_72"] for x in s) <= 0.033
    assert (round(min(x["median_ratio"] for x in s), 2), round(max(x["median_ratio"] for x in s), 2)) == (1.05, 2.45)
    full = e61["full"]
    seas = [c["median_ratio"] for c in full if c["scenario"].startswith("seasonal")]
    assert (round(min(seas), 2), round(max(seas), 2)) == (0.83, 0.97)

    def rng(lab: str, m: str) -> tuple[float, float]:
        v = [c[lab]["coverage"][m] for c in full if c[lab]["n"]]
        return round(min(v), 2), round(max(v), 2)

    assert len({c["scenario"] for c in full}) == 10
    assert (
        round(min(c["all"]["coverage"]["cr2_bm"] for c in full), 3),
        round(max(c["all"]["coverage"]["cr2_bm"] for c in full), 3),
    ) == (0.947, 0.955)
    assert rng("ratio_le_0_72", "cr2_bm") == (0.78, 0.90) and rng("ratio_le_0_59", "cr2_bm") == (0.70, 0.86)
    assert rng("ratio_le_0_72", "cr2_floor") == (0.92, 0.99) and rng("all", "cr2_floor") == (0.96, 0.99)
    assert e62["changed"] == ["RF-PureLSTM", "RF-LSTM_QNN"]
    assert round(e62["RF-PureLSTM"]["p_floor"], 3) == 0.204 and round(e62["RF-LSTM_QNN"]["p_floor"], 3) == 0.107
    assert e62["FROZEN_RANDOM-LSTM_QNN"]["resolved_floor"] and e62["RF-QTL_FIXED"]["resolved_floor"]
    in_text(
        MAIN_TEXT,
        "close to its lower bound of $-0.037$",
        "lowers its median only to 0.83",
        "coverage 0.947--0.955 overall",
        "0.78--0.90 among replications with a ratio $\\le0.72$",
        "0.70--0.86 with a ratio $\\le0.59$",
        "keeps 0.92--0.99 there",
        "conservative overall (0.96--0.99)",
        "no longer resolved ($p=0.204$ and 0.107)",
        "decide with the standard error floored at the iid one",
    )
    in_text(
        SUPP_TEXT,
        "Simulation of Within-Year Cancellation",
        "give median ratios of 1.05--2.45",
        "is at most 0.033",
        "lowers the median ratio to 0.83--0.97",
    )


def test_protocol_as_main_contribution() -> None:
    abstract = MAIN_TEXT.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    assert "needs only archived predictions, applies to any metric with per-case influence values" in abstract
    in_text(
        MAIN_TEXT,
        "The main contribution is \\textbf{a reporting protocol and the package \\texttt{qml\\_resolution}}",
        "The protocol needs only archived predictions, labels, cluster labels and seeds, and applies to any metric "
        "with per-case influence values; \\texttt{qml\\_resolution} implements its inferential items",
    )
    for label in ("tab:controls", "fig:architecture"):  # in S
        assert f"\\label{{{label}}}" in SUPP_TEXT and f"\\label{{{label}}}" not in MAIN_TEXT, label
