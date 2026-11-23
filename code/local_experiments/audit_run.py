#!/usr/bin/env python3
"""
audit_run.py - run a script while logging every file it opens inside the repository (refit v3, date audit)
=========================================================================================================
  python code/local_experiments/audit_run.py <script.py> [args...]
Every 'open' audit event (Python's sys.addaudithook) whose path lies under the working directory is appended to the
file named by LOR_OPEN_LOG as '<mode or flags>\t<absolute path>'. scripts/refit_audit.py then checks that no file
read under results/, checkpoints/ or logs_v2/ predates the start of the run.
"""
import os
import runpy
import sys

ROOT = os.path.abspath(os.getcwd())
LOG = open(os.environ['LOR_OPEN_LOG'], 'a', encoding='utf-8')


def _hook(event, args):
    if event != 'open' or not args:
        return
    path = args[0]
    if isinstance(path, int):
        return
    try:
        p = os.path.abspath(os.fsdecode(path))
    except (TypeError, ValueError):
        return
    if p.startswith(ROOT):
        mode = args[1] if len(args) > 1 else ''
        LOG.write(f'{mode}\t{p}\n')
        LOG.flush()


sys.addaudithook(_hook)
script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name='__main__')
