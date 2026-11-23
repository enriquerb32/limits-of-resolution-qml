"""Numbers in the manuscript must follow from the archived results.

Covers the trigger of the floored CR2 interval (protocol item 1) and the MDE of the procedure that decides (exp63),
the wording of the equivalence margin and of the scope of the claims, and statements kept in the supplement.
"""

from __future__ import annotations

from _archive import MAIN_TEXT, SUPP_TEXT, in_text, load

E63 = load("local_exp63_floor_threshold.json")


def _range(values: list[float], nd: int = 2) -> tuple[float, float]:
    return round(min(values), nd), round(max(values), nd)


def test_floor_trigger() -> None:
    thr = E63["threshold"]
    assert thr["reps"] == 20000 and round(thr["q05"], 3) == 0.708 and round(thr["q05"], 2) == 0.71
    assert round(thr["median"], 3) == 0.975 and round(thr["q01"], 2) == 0.61 and round(thr["q10"], 2) == 0.76
    in_text(
        MAIN_TEXT,
        "at or below the 5th percentile of its distribution simulated on this test design with independent cases, 0.71",
        "(here at or below its 5th percentile, 0.71, recalibrated for each design)",
        "decide with the standard error floored at the iid one",
    )
    in_text(SUPP_TEXT, "0.708 in 20{,}000 replications (median 0.975; 1st and 10th percentiles 0.61 and 0.76)")


def test_floor_operating_characteristics() -> None:
    o = E63["operating"]
    assert len(o) == 20 and all(r["reps"] == 5000 for r in o)
    assert _range([r["all"]["rule"] for r in o]) == (0.95, 0.98)
    assert _range([r["all"]["cr2"] for r in o]) == (0.95, 0.96)
    assert _range([r["all"]["always_floored"] for r in o]) == (0.96, 0.99)
    assert _range([r["triggered"]["rule"] for r in o]) == (0.89, 0.99)
    assert _range([r["triggered"]["cr2"] for r in o]) == (0.73, 0.89)
    assert (min(r["triggered"]["n"] for r in o), max(r["triggered"]["n"] for r in o)) == (45, 981)
    assert _range([r["q05_ratio"] for r in o]) == (0.61, 0.84)
    assert _range([100 * r["share_triggered"] for r in o], 1) == (0.9, 19.6)
    in_text(
        MAIN_TEXT,
        "this rule covers at 0.95--0.98 overall and at 0.89--0.99 among the replications that trigger it, "
        "where CR2 covers at 0.73--0.89",
    )
    in_text(
        SUPP_TEXT,
        "the 5th percentile of the ratio is 0.61--0.84 and the rule triggers in 0.9--19.6\\,\\%",
        "covers at 0.95--0.98 overall, against 0.95--0.96 for CR2 and 0.96--0.99",
        "(45 to 981 per cell)",
        "where CR2 covers at 0.73--0.89",
    )


def test_floor_on_the_contrasts() -> None:
    c = E63["contrasts"]
    assert c["triggered"] == ["RF-PureLSTM"] and c["changed"] == ["RF-PureLSTM"]
    lstm, qnn, qtl = c["RF-PureLSTM"], c["RF-LSTM_QNN"], c["RF-QTL_FIXED"]
    assert round(lstm["ratio_cr1_iid"], 3) == 0.594 and round(lstm["p_decision"], 3) == 0.204
    assert round(lstm["mde_decision"], 3) == 0.033 and round(lstm["mde_delong"], 3) == 0.031
    assert round(qnn["ratio_cr1_iid"], 3) == 0.723 and not qnn["triggered"] and qnn["resolved_decision"]
    assert round(qnn["ratio_cr1_iid"] - E63["threshold"]["q05"], 3) == 0.014
    assert round(qtl["ratio_cr1_iid"], 3) == 0.779 and qtl["resolved_decision"]
    # floored MDE of RF - LSTM+QNN, had it triggered: (t_0.975 + t_0.80) SE_iid; both contrasts share nu = 15.9, so the
    # factor (t_0.975 + t_0.80) / 2.80 of RF - LSTM carries over to the DeLong MDE of RF - LSTM+QNN
    assert abs(lstm["df"] - qnn["df"]) < 1e-9
    floored_qnn = lstm["mde_decision"] / lstm["mde_delong"] * qnn["mde_delong"]
    assert round(floored_qnn, 3) == 0.034
    in_text(
        MAIN_TEXT,
        "would \\emph{not be resolved} (floored $p=0.204$; MDE of the floored procedure 0.033)",
        "The verdicts of the paper are those of the unfloored CR2 interval",
        "Random Forest against \\texttt{LSTM+QNN} (ratio 0.72) lies just above the trigger",
        "floored $p=0.204$, MDE 0.033",
        "(floored, $p=0.107$)",
    )
    in_text(SUPP_TEXT, "has MDE 0.033 (DeLong 0.031)", "floored its MDE would be 0.034", "falls 0.014 above it")
    # a trigger of 0.75 (sensitivity of the trigger): which AUC contrasts it would floor, and what changes
    e62 = load("local_exp62_floored_cr2_observed.json")
    at75 = sorted(k for k, v in c.items() if isinstance(v, dict) and v["ratio_cr1_iid"] <= 0.75)
    assert at75 == ["RF-LSTM_QNN", "RF-PureLSTM", "RF-XGB"]
    assert round(e62["RF-LSTM_QNN"]["p_floor"], 3) == 0.107 and not e62["RF-XGB"]["resolved_cr2"]
    assert (round(e62["RF-XGB"]["p_cr2"], 3), round(e62["RF-XGB"]["p_floor"], 3)) == (0.110, 0.224)
    in_text(
        MAIN_TEXT,
        "with a tree ensemble (0.033 for Random Forest against the LSTM with the floor of the sensitivity check)",
    )
    in_text(
        SUPP_TEXT,
        "a Brier-specific trigger would be computed in the same way",
        "another test set needs its own calibration",
    )
    in_text(MAIN_TEXT, "a trigger of 0.75 would floor it too ($p=0.107$), a verdict that was already exploratory")
    in_text(SUPP_TEXT, "A trigger of 0.75 would also floor", "($p$ from 0.110 to 0.224; not resolved either way)")


def test_section_openings_and_literature_table() -> None:
    in_text(
        MAIN_TEXT,
        "On a second panel the null results held, the ranking did not",
        "iid inference covered a fixed reference difference in fewer than two thirds of small test sets",
    )
    lit = MAIN_TEXT.split("\\label{tab:literature}")[1].split("\\end{table}")[0]
    assert "adjustbox" not in lit and "\\begin{tabularx}{\\textwidth}" in lit  # legible at text size, not scaled


def test_floor_on_brier() -> None:
    b = E63["brier"]
    assert b["triggered"] == ["recal/classical/RF-PureLSTM", "recal/climatology/PureLSTM-CLIM"]
    assert b["changed"] == ["recal/climatology/PureLSTM-CLIM"]
    v = b["recal/climatology/PureLSTM-CLIM"]
    assert round(v["ratio_cr1_iid"], 2) == 0.66 and round(v["p_cr2"], 3) == 0.016 and round(v["p_decision"], 3) == 0.095
    in_text(MAIN_TEXT, "the LSTM no longer qualifies (ratio 0.66, $p=0.095$)")
    in_text(SUPP_TEXT, "the LSTM is not ($p=0.095$", "(ratio 0.66; $p$ from 0.016 to 0.095)")


def test_claim_wording() -> None:
    abstract = MAIN_TEXT.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    assert "AUC differences of a few hundredths" in abstract
    assert "independently of these results" not in MAIN_TEXT and "independently of these results" not in SUPP_TEXT
    in_text(
        MAIN_TEXT,
        "not chosen from the equivalence results but not blind to them",
        "In the published comparisons we reanalysed illustratively",
        "the floor of protocol item 1 were fixed after the main analysis",
        "is a warning, not a gain in resolution",
        "Nothing in the protocol is specific to precipitation",
    )
    in_text(SUPP_TEXT, "it was not chosen from the equivalence results, but it was not blind to them")
    for marker in ("XXALL", "XXTRIG", "XXCR2", "XXBRIER"):
        assert marker not in MAIN_TEXT, marker


def test_primary_floor_separation_and_scope() -> None:
    abstract = MAIN_TEXT.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    assert "nor equivalence of \\texttt{LSTM+QNN} established" in abstract  # LSTM+QTL is nominally equivalent (S)
    assert "covered a fixed reference difference in 55--63\\,\\% of draws, CR2 in 94--99.8\\,\\%" in abstract
    assert "41--63" not in MAIN_TEXT
    fig1 = MAIN_TEXT.split("\\node[prim=13.6cm, below=of holm] (dec)")[1].split("\\draw")[0]
    dec, rob = fig1.split("(rob)")
    assert "floored" not in dec and "CR2 floored at DeLong" in rob  # the floor is robustness, not the decision
    in_text(
        MAIN_TEXT,
        "the unfloored CR2 interval decides and the floor is a sensitivity check",
        "proposes, for analyses planned with it",
        "recalibrated for each design",
        "no advantage over the interface controls was detected in this design",
        "the benchmark tested coverage, not power",
    )
    assert "could have been detected" not in MAIN_TEXT
    in_text(SUPP_TEXT, "In this paper the unfloored CR2 interval, fixed before the rule existed, decides")
