"""Paths, manuscript text and archived results shared by the manuscript tests."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
SUPP = ROOT / "paper" / "supp.tex"
RES = ROOT / "results"
MAIN_TEXT = TEX.read_text(encoding="utf-8")
SUPP_TEXT = SUPP.read_text(encoding="utf-8")


@cache
def _read(name: str) -> str:
    return (RES / name).read_text(encoding="utf-8")


def load(name: str) -> Any:
    """Parse an archived JSON file of ``results/`` (a fresh object on every call)."""
    return json.loads(_read(name))


def in_text(text: str, *claims: str) -> None:
    """Assert that every claim appears verbatim in ``text``."""
    for c in claims:
        assert c in text, c
