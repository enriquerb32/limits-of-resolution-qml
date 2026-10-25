"""
tests/test_exp10_reproducibility.py
Verify that exp10_paired_ci.py reproduces Table 4 metrics from archived probabilities.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_exp10_paired_ci_reproduction():
    """Verify that running exp10 on canonical probabilities matches archived Table 4 results."""
    probs_path = os.path.join(ROOT, "results", "json", "results_rerun_train_only_canonical_probs.npz")
    archived_json = os.path.join(ROOT, "results", "json", "results_exp10_paired_ci.json")

    assert os.path.exists(probs_path), f"Missing {probs_path}"
    assert os.path.exists(archived_json), f"Missing {archived_json}"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_out = tmp.name

    try:
        script = os.path.join(ROOT, "code", "local_experiments", "exp10_paired_ci.py")
        cmd = [
            sys.executable,
            script,
            "--probs", probs_path,
            "--prefix", "train_only",
            "--B", "200",
            "--out", tmp_out,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        assert res.returncode == 0, f"exp10 failed:\n{res.stderr}\n{res.stdout}"

        with open(tmp_out, "r", encoding="utf-8") as f:
            recomputed = json.load(f)["pairs"]
        with open(archived_json, "r", encoding="utf-8") as f:
            archived = json.load(f)["pairs"]

        for pair in ["RF-XGB", "RF-LSTM", "XGB-LSTM"]:
            assert pair in recomputed, f"Pair {pair} missing from recomputed output"
            # Analytical delta and DeLong standard errors must match exactly
            assert abs(recomputed[pair]["delta"] - archived[pair]["delta"]) < 1e-4
            assert abs(recomputed[pair]["se_delong"] - archived[pair]["se_delong"]) < 1e-4
            assert abs(recomputed[pair]["p_delong"] - archived[pair]["p_delong"]) < 1e-4
            # Bootstrap SE with B=200 within Monte Carlo sampling tolerance
            assert abs(recomputed[pair]["se_block"] - archived[pair]["se_block"]) < 0.005

    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
