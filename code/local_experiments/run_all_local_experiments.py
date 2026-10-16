"""
code/local_experiments/run_all_local_experiments.py
Master runner for all WP7 local experiments:
  1. Month x Station Climatology Baseline
  2. Persistence Baseline (Continuous and Binary)
  3. Brier Score and Reliability/Calibration Analysis
  4. Year-Block Bootstrap Analysis
  5. PCA-4 Bottleneck Controls vs QKSVM
  6. TOST and ROPE Equivalence Testing
"""

import subprocess
import sys
import os

EXPERIMENTS = [
    ("Exp 1: Climatology Baseline", "code/local_experiments/exp1_climatology_baseline.py"),
    ("Exp 2: Persistence Baseline", "code/local_experiments/exp2_persistence_baseline.py"),
    ("Exp 3: Brier & Reliability", "code/local_experiments/exp3_brier_and_reliability.py"),
    ("Exp 4: Year-Block Bootstrap", "code/local_experiments/exp4_year_block_bootstrap.py"),
    ("Exp 5: PCA-4 Controls", "code/local_experiments/exp5_pca4_controls.py"),
    ("Exp 6: TOST & ROPE Equivalence", "code/local_experiments/exp6_tost_rope_equivalence.py"),
]

def main():
    py_exec = sys.executable
    print("="*75)
    print("EXECUTING ALL WP7 LOCAL EXPERIMENTS (scikit-learn / scipy, no retraining)")
    print("="*75)

    for title, script_path in EXPERIMENTS:
        print(f"\n>>> Running {title} ({script_path})...")
        res = subprocess.run([py_exec, script_path], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"ERROR in {script_path}:")
            print(res.stderr)
            sys.exit(res.returncode)
        else:
            print(res.stdout.strip())

    print("\n" + "="*75)
    print("ALL WP7 LOCAL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*75)

if __name__ == '__main__':
    main()
