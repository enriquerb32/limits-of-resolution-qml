#!/usr/bin/env python3
"""
refit_v3_run.py - runs one phase of refit v3 in the clean worktree, then its date audit
======================================================================================
  python code/local_experiments/refit_v3_run.py --phase trees|search|final|teacher [--workers 6]
Every job runs through audit_run.py (files opened are logged to logs_v2/open_<phase>.log), with LOR_DATASET=v2,
V2_CAUSAL=1, PYTHONHASHSEED=0 and its own log in logs_v2/refit_<phase>_<job>.log. Jobs start when their dependencies
have finished; a failed job stops further launches. After the last job, scripts/refit_audit.py checks the dates.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.getcwd())
L = os.path.join('code', 'local_experiments')
PY = sys.executable
SEARCHED = ['PureLSTM', 'LSTM_QNN', 'QTL_FIXED', 'CTRL_LINEAR4D', 'FOURIER1', 'FOURIER2', 'FOURIER3', 'FOURIER4',
            'FOURIER_FULL1', 'FOURIER_FULL2', 'FOURIER_FULL4']
FINAL = SEARCHED + ['FROZEN_RANDOM', 'FROZEN_IDENTITY', 'QTL_CLASSICAL', 'QTL_TENSOR1']
E66 = os.path.join(L, 'exp66_refit_v3.py')
E67 = os.path.join(L, 'exp67_teacher_v3.py')


def jobs(phase):
    if phase == 'trees':
        return {'tune': ([os.path.join(L, 'exp40_v2_tune.py'), '--trials', '50'], [], 6),
                'classical': ([os.path.join(L, 'exp41_v2_classical.py'), '--seeds', '15'], ['tune'], 6)}
    if phase == 'search':
        return {m: ([E66, '--step', 'search', '--models', m], ['PureLSTM'] if m == 'QTL_FIXED' else [], 1)
                for m in SEARCHED}
    if phase == 'final':
        return {m: ([E66, '--step', 'final', '--models', m], ['PureLSTM'] if m.startswith('QTL_') else [], 1)
                for m in FINAL}
    if phase == 'teacher':
        heads = ('CIRCUIT', 'LINEAR4D', 'FOURIER2', 'FOURIER_FULL1', 'FOURIER_FULL4')
        J = {f'teacher_{h}': ([E67, '--heads', h], [], 1) for h in heads}
        J['teacher_combine'] = ([E67, '--combine'], [f'teacher_{h}' for h in heads], 1)
        return J
    raise SystemExit(phase)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--phase', required=True)
    ap.add_argument('--workers', type=int, default=6); a = ap.parse_args()
    if not os.path.exists('refit_v3_run_start.txt'):
        raise SystemExit('not a refit v3 worktree (refit_v3_run_start.txt missing)')
    os.makedirs('logs_v2', exist_ok=True)
    since = dt.datetime.now().isoformat(timespec='seconds')
    open_log = os.path.join('logs_v2', f'open_{a.phase}.log')
    J = jobs(a.phase)
    done, failed, running = set(), set(), {}
    print(f'[{since}] phase {a.phase}: {len(J)} jobs, {a.workers} workers', flush=True)
    while len(done) + len(failed) < len(J):
        for name, (args, deps, threads) in J.items():
            if name in done or name in failed or name in running or failed:
                continue
            if len(running) >= a.workers or not all(d in done for d in deps):
                continue
            env = dict(os.environ, LOR_DATASET='v2', V2_CAUSAL='1', PYTHONHASHSEED='0', PYTHONUTF8='1',
                       OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), LOR_OPEN_LOG=open_log)
            own = E66 in args or E67 in args
            cmd = [PY, os.path.join(L, 'audit_run.py')] + args + (['--threads', str(threads)] if own else [])
            log = open(os.path.join('logs_v2', f'refit_{a.phase}_{name}.log'), 'w', encoding='utf-8')
            running[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log, time.time())
            print(f'  start {name}', flush=True)
        if not running and failed:
            break
        time.sleep(5)
        for name in list(running):
            p, log, t0 = running[name]
            if p.poll() is not None:
                log.close(); del running[name]
                (done if p.returncode == 0 else failed).add(name)
                print(f'  {"done" if p.returncode == 0 else "FAILED"} {name} ({(time.time() - t0) / 60:.1f} min, '
                      f'exit {p.returncode})', flush=True)
    if failed:
        raise SystemExit(f'phase {a.phase}: failed jobs {sorted(failed)}')
    rc = subprocess.call([PY, os.path.join('scripts', 'refit_audit.py'), '--phase', a.phase, '--since', since,
                          '--open-log', open_log])
    print(f'[{dt.datetime.now().isoformat(timespec="seconds")}] phase {a.phase} finished, audit exit {rc}', flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
