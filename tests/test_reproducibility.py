"""
tests/test_reproducibility.py
==============================
Minimal pytest verification of core reproducibility invariants:
  1. Parameter counts (TR-021: 2,613 total / 24 quantum; TR-022: 2,609 / 20 params)
  2. Dataset partition sizes (TR-016..19: 2,273 train / 514 val / 528 test)
  3. Traceability matrix integrity (82 verified claims)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_parameter_counts():
    """Verify exact parameter counts matching paper claims TR-021 and TR-022."""
    script = os.path.join(ROOT, "scripts", "verify_parameter_counts.py")
    res = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, f"verify_parameter_counts failed:\n{res.stderr}"
    stdout = res.stdout
    assert "LSTM+QNN TOTAL: 2613 (0.92% quantum)" in stdout
    assert "Linear-4D TOTAL: 2609" in stdout


def test_partition_counts():
    """Verify exact sample split sizes (2,273 train, 514 val, 528 test)."""
    script = os.path.join(ROOT, "scripts", "make_partition_table.py")
    res = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, f"make_partition_table failed:\n{res.stderr}"
    assert "2273     514      528      3315" in res.stdout


def test_traceability_matrix():
    """Verify that all 82 claims in the audit matrix pass without mismatches."""
    script = os.path.join(ROOT, "scripts", "build_traceability.py")
    res = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, f"build_traceability failed:\n{res.stderr}\n{res.stdout}"
    assert "0 mismatches" in res.stdout
    assert "82 claims" in res.stdout
