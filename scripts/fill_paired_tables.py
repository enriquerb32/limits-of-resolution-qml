#!/usr/bin/env python3
"""fill_paired_tables.py - rewrite the rows of Tables 3, 6 and 7 of paper/main.tex from results/local_exp42_v2_summary.json.

Columns: difference, DeLong SE, z (p), primary CR2 interval with Bell-McCaffrey df (p), years x seeds interval,
cluster/iid ratio (CR1 / DeLong), nominal DeLong MDE. The row map is the one checked by tests/manuscript.

  python scripts/fill_paired_tables.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "manuscript"))
from test_v2_tables import PAIRED, PANEL2_ROWS, SUPP_PAIRED, flip  # noqa: E402

TEX = ROOT / "paper" / "main.tex"
SUPP = ROOT / "paper" / "supp.tex"


def s(x, d):
    return f"{x:+.{d}f}"


def row_cells(p):
    return (f"${s(p['gap'], 4)}$ & {p['se_iid']:.4f} & ${p['gap'] / p['se_iid']:.2f}$ ({p['p_delong']:.3f}) & "
            f"$[{s(p['ci_cr2'][0], 3)},{s(p['ci_cr2'][1], 3)}]$ ({p['p_cr2']:.3f}) & "
            f"$[{s(p['ci_two_way'][0], 3)},{s(p['ci_two_way'][1], 3)}]$ & {p['ratio_year_iid']:.2f} & {p['mde_iid']:.3f} \\\\")


def main():
    pairs = json.loads((ROOT / "results" / "local_exp42_v2_summary.json").read_text(encoding="utf-8"))["pairs"]
    supp = SUPP.read_text(encoding="utf-8")                 # rows of Table 7 moved to the supplement
    for table, rows in SUPP_PAIRED.items():
        start = supp.index(f"\\label{{{table}}}"); end = supp.index("\\end{tabular}", start)
        body = supp[start:end]
        for label, (key, flipped) in rows.items():
            p = flip(pairs[key]) if flipped else pairs[key]
            tl = re.escape(label.replace(" - ", " $-$ "))
            body, n = re.subn(rf"^({tl}) & .*\\\\$", lambda m, p=p: f"{m.group(1)} & {row_cells(p)}", body, flags=re.M)
            assert n == 1, (table, label, n)
        supp = supp[:start] + body + supp[end:]
    SUPP.write_text(supp, encoding="utf-8")
    tex = TEX.read_text(encoding="utf-8")
    for table, rows in PAIRED.items():
        start = tex.index(f"\\label{{{table}}}"); end = tex.index("\\end{tabular}", start)
        body = tex[start:end]
        for label, (key, flipped) in rows.items():
            p = flip(pairs[key]) if flipped else pairs[key]
            tl = re.escape(label.replace(" - ", " $-$ "))
            body, n = re.subn(rf"^({tl}) & .*\\\\$", lambda m: f"{m.group(1)} & {row_cells(p)}", body, flags=re.M)
            assert n == 1, (table, label, n)
        body = body.replace("Year-block 95\\,\\% CI", "CR2 95\\,\\% CI ($p$)")
        tex = tex[:start] + body + tex[end:]
    # Table 13 (second panel): the fifth cell becomes the primary CR2 interval with its p-value
    p2 = json.loads((ROOT / "results" / "local_panel2_cr2.json").read_text(encoding="utf-8"))
    start = tex.index("\\label{tab:panel2}"); end = tex.index("\\end{tabular}", start)
    body = tex[start:end]
    for label, (key, _, _) in PANEL2_ROWS.items():
        v = p2[key]
        tl = re.escape(label.replace(" - ", " $-$ "))
        cr2 = f"$[{s(v['ci_cr2'][0], 3)},{s(v['ci_cr2'][1], 3)}]$ ({v['p_cr2']:.3f})"

        def repl(m, cr2=cr2):
            cells = m.group(0)[:-2].split(" & ")
            cells[4] = cr2
            return " & ".join(cells) + "\\\\"
        body, n = re.subn(rf"^{tl} & .*\\\\$", repl, body, flags=re.M)
        assert n == 1, (label, n)
    body = body.replace("Year-block 95\\,\\% CI", "CR2 95\\,\\% CI ($p$)")
    tex = tex[:start] + body + tex[end:]
    TEX.write_text(tex, encoding="utf-8")
    print("rewrote", sum(len(r) for r in PAIRED.values()) + len(PANEL2_ROWS), "rows")


if __name__ == "__main__":
    main()
