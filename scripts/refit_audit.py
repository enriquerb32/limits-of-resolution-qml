#!/usr/bin/env python3
"""
refit_audit.py - date audit after each phase of refit v3 (audit/refit_plan_v3.txt, section 1)
=============================================================================================
  python scripts/refit_audit.py --phase <name> --since <ISO time the phase started> --open-log <log>
Checks, in the clean worktree:
  1. every file under results/, checkpoints/ and logs_v2/ was written after the run started (refit_v3_run_start.txt);
  2. every file this phase wrote (modified at or after --since) is listed;
  3. every file the phase's processes opened for reading under results/, checkpoints/ or logs_v2/ (audit_run.py log)
     was written by this run (after the run start), so no archived prediction, checkpoint, tuning or resume file
     was read.
Writes results/refit_v3/audit_<phase>.json and exits 1 on any failure.
"""
import argparse
import datetime as dt
import json
import os
import sys

ROOT = os.path.abspath(os.getcwd())
WATCH = ('results', 'checkpoints', 'logs_v2')


def mtime(p):
    return dt.datetime.fromtimestamp(os.path.getmtime(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True); ap.add_argument('--since', required=True)
    ap.add_argument('--open-log', required=True)
    a = ap.parse_args()
    start = dt.datetime.fromisoformat(open(os.path.join(ROOT, 'refit_v3_run_start.txt')).read().strip())
    since = dt.datetime.fromisoformat(a.since)
    files = [os.path.join(d, f) for w in WATCH for d, _, fs in os.walk(os.path.join(ROOT, w)) for f in fs]
    old = [os.path.relpath(p, ROOT) for p in files if mtime(p) < start]
    written = sorted(os.path.relpath(p, ROOT) for p in files if mtime(p) >= since)
    reads, bad_reads = set(), []
    if os.path.exists(a.open_log):
        for line in open(a.open_log, encoding='utf-8'):
            mode, _, p = line.rstrip('\n').partition('\t')
            rel = os.path.relpath(p, ROOT)
            if not rel.split(os.sep)[0] in WATCH or 'w' in mode or 'a' in mode or 'x' in mode:
                continue
            reads.add(rel)
    for rel in sorted(reads):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p) and mtime(p) < start:
            bad_reads.append(rel)
    rep = dict(phase=a.phase, run_start=start.isoformat(), phase_start=since.isoformat(),
               checked_at=dt.datetime.now().isoformat(timespec='seconds'), files_checked=len(files),
               files_before_run_start=old, written_this_phase=written, read_under_watched_dirs=sorted(reads),
               reads_of_files_older_than_run=bad_reads, passed=not old and not bad_reads)
    os.makedirs(os.path.join(ROOT, 'results', 'refit_v3'), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, 'results', 'refit_v3', f'audit_{a.phase}.json'), 'w'), indent=2)
    print(f"audit {a.phase}: {'PASS' if rep['passed'] else 'FAIL'} | {len(files)} files, {len(written)} written in "
          f"this phase, {len(reads)} read under {'/'.join(WATCH)}, {len(old)} older than the run, "
          f"{len(bad_reads)} old files read")
    sys.exit(0 if rep['passed'] else 1)


if __name__ == '__main__':
    main()
