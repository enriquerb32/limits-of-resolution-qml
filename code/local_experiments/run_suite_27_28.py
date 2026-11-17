#!/usr/bin/env python3
"""
Sequential runner for exp27 and exp28 (orig and stationmonth).
"""
import os
import subprocess
import sys
import time

PYTHON = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

commands = [
    ("exp27_controls", [PYTHON, "-u", os.path.join(HERE, "exp27_controls.py")]),
    ("exp28_orig", [PYTHON, "-u", os.path.join(HERE, "exp28_labels_recalibration.py"), "--label", "orig"]),
    ("exp28_orig_analyze", [PYTHON, "-u", os.path.join(HERE, "exp28_labels_recalibration.py"), "--label", "orig", "--analyze"]),
    ("exp28_stationmonth", [PYTHON, "-u", os.path.join(HERE, "exp28_labels_recalibration.py"), "--label", "stationmonth"]),
    ("exp28_stationmonth_analyze", [PYTHON, "-u", os.path.join(HERE, "exp28_labels_recalibration.py"), "--label", "stationmonth", "--analyze"]),
]

env = os.environ.copy()
env["MPLCONFIGDIR"] = os.environ.get("TEMP", "C:\\Temp")

for name, cmd in commands:
    print(f"\n========================================================", flush=True)
    print(f"STARTING {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"========================================================\n", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, env=env)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"\nFAILED {name} with return code {res.returncode} after {dt/60:.1f} min", flush=True)
        sys.exit(res.returncode)
    print(f"\nCOMPLETED {name} in {dt/60:.1f} min", flush=True)

print(f"\nALL EXPERIMENTS COMPLETED SUCCESSFULLY at {time.strftime('%Y-%m-%d %H:%M:%S')}!", flush=True)

