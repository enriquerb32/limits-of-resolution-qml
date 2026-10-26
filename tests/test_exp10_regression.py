import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_exp10_reproduces_archived_table(tmp_path):
    out = tmp_path / "exp10.json"
    cmd = [sys.executable, os.path.join(ROOT, "code", "local_experiments", "exp10_paired_ci.py"),
           "--probs", os.path.join(ROOT, "results", "json", "results_rerun_train_only_canonical_probs.npz"),
           "--prefix", "train_only",
           "--extra", os.path.join(ROOT, "results", "json", "results_qml_core_trainonly_probs.npz"),
           "--B", "50", "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    new = json.loads(out.read_text())["pairs"]
    ref = json.load(open(os.path.join(ROOT, "results", "json", "results_exp10_paired_ci.json")))["pairs"]
    assert sorted(new) == sorted(ref)
    for k in ref:
        assert abs(new[k]["delta"] - ref[k]["delta"]) < 1e-9, k
        assert abs(new[k]["se_delong"] - ref[k]["se_delong"]) < 1e-9, k
