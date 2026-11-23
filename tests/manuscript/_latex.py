"""Minimal reader for the tables of the manuscript.

The tests read the numbers *as printed* in ``paper/main.tex`` (not copies of them), so that any edit
of the manuscript that breaks agreement with the archived results fails the suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Printed:
    """A number as printed in the manuscript, with the number of decimals it was rounded to."""

    value: float
    decimals: int

    def matches(self, archived: float) -> bool:
        """True if ``archived`` rounds to the printed value (half a unit in the last printed place)."""
        return abs(archived - self.value) <= 0.5 * 10.0**-self.decimals + 1e-9


@cache
def _read(tex: Path) -> str:
    return tex.read_text(encoding="utf-8")


def _strip(cell: str) -> str:
    cell = re.sub(r"\\(?:textbf|texttt|emph|textit)\{([^{}]*)\}", r"\1", cell)
    cell = cell.replace("{,}", "").replace("\\,", " ").replace("\\%", "%").replace("$-$", "-")
    cell = cell.replace("$", "").replace("\\\\", "").replace("\\hline", "")
    return " ".join(cell.split())


def numbers(cell: str) -> list[Printed]:
    """All numbers of a table cell, in order (``1.27 (0.205)`` gives two, ``[-0.002,+0.020]`` gives two)."""
    out = []
    for tok in _NUMBER.findall(_strip(cell)):
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        out.append(Printed(float(tok), decimals))
    return out


def table_rows(tex: Path, label: str) -> dict[str, list[str]]:
    """Rows of the table carrying ``\\label{label}``, keyed by their cleaned first cell."""
    text = _read(tex)
    pos = text.index(f"\\label{{{label}}}")
    start = text.rindex("\\begin{table", 0, pos)
    end = text.index("\\end{table", pos)
    rows: dict[str, list[str]] = {}
    for line in text[start:end].splitlines():
        if "&" not in line or "\\caption" in line or "multicolumn" in line:
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)&", line.split("\\\\")[0])]
        rows[_strip(cells[0])] = cells[1:]
    return rows
