"""
tests/test_reproducibility.py
==============================
End-to-end reproducibility checks (slow):
  1. Parameter counts of the hybrid and of the Linear-4D control (2,613 / 24 quantum; 2,609).
  2. Test partition of the v2 manuscript (Table 1) recomputed from the archived test metadata.
  3. v2 claims: scripts/build_traceability_v2.py (claims copied from the v2 manuscript) passes.
  4. v1 claims: scripts/build_traceability.py (claims of the v1 manuscript) passes on results_v1/.
  5. The v2 summary and tables (exp42, exp43) are regenerated from the archived probabilities bit for bit.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.slow


def _run(script, *args, cwd=ROOT):
    res = subprocess.run(
        [sys.executable, os.path.join(ROOT, *script.split("/")), *args], capture_output=True, text=True, cwd=cwd
    )
    assert res.returncode == 0, f"{script} failed:\n{res.stderr}\n{res.stdout}"
    return res.stdout


def test_parameter_counts():
    """Analytical parameter counts (TR-021, TR-022); tests/quantum checks the same counts on the model classes."""
    stdout = _run("scripts/verify_parameter_counts.py")
    assert "LSTM+QNN TOTAL: 2613 (0.92% quantum)" in stdout
    assert "Linear-4D TOTAL: 2609" in stdout


def test_partition_counts(tmp_path):
    """Table 1 of the v2 manuscript: 529 test windows (215 positive) in 19 years, 516 validation windows."""
    stdout = _run("scripts/make_partition_table.py")
    assert "Total          529              40.6% (215 / 529)" in stdout
    assert "validation windows: 516; test windows: 529 in 19 years (2008-2026)" in stdout


def test_traceability_v2():
    stdout = _run("scripts/build_traceability_v2.py")
    m = re.search(r"(\d+)/(\d+) verified", stdout)
    assert m and m.group(1) == m.group(2), stdout


def test_traceability_v1(tmp_path):
    """Every claim of the v1 audit matrix passes on the v1 archives, without mismatches."""
    stdout = _run("scripts/build_traceability.py", "--out", str(tmp_path / "v1.csv"))
    m = re.search(r"(\d+) claims: (\d+) verified, (\d+) read-only, (\d+) unverified, (\d+) mismatches", stdout)
    assert m, f"summary line not found:\n{stdout}"
    total, verified, read_only, unverified, mismatches = map(int, m.groups())
    assert total > 0
    assert mismatches == 0 and unverified == 0
    assert verified + read_only == total
    tracked = Path(ROOT, "audit", "01_traceability_v1.csv").read_text(encoding="utf-8")
    assert (tmp_path / "v1.csv").read_text(encoding="utf-8") == tracked


def _same(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (u, v) in enumerate(zip(a, b, strict=True)):
            _same(u, v, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        assert math.isnan(b), path
    elif isinstance(a, float):
        assert abs(a - b) <= 1e-12, f"{path}: {a} vs {b}"
    else:
        assert a == b, path


@pytest.mark.parametrize(
    "script, output",
    [
        ("exp43_v2_tables.py", "local_exp43_v2_tables.json"),
        ("exp42_v2_summary.py", "local_exp42_v2_summary.json"),  # about 10 min (B = 2,000 per pair, 31 pairs)
    ],
)
def test_v2_tables_regenerate_from_archived_probabilities(tmp_path, script, output):
    res = Path(ROOT, "results")
    (tmp_path / "json").mkdir()
    for f in (res / "json").glob("*.npz"):
        shutil.copy(f, tmp_path / "json" / f.name)
    shutil.copy(res / "test_meta.csv", tmp_path / "test_meta.csv")
    _run(f"code/local_experiments/{script}", str(tmp_path))
    new = json.loads((tmp_path / output).read_text(encoding="utf-8"))
    archived = json.loads((res / output).read_text(encoding="utf-8"))
    _same(archived, new)
