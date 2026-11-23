#!/usr/bin/env python3
"""Paired uncertainty for the difference between the seed-mean AUCs of two models.

Estimand (conditional on the trained models and the fixed 528-row test set)::

    Delta = mean_s AUC(p_{A,s}) - mean_s AUC(p_{B,s})

Three uncertainty estimates for the same quantity:

1. DeLong (iid rows) on seed-averaged structural components.
2. Calendar-year block bootstrap: the same resampled years for both models.
3. Two-way bootstrap: years and, independently within each model, seeds.

MDE = 2.80 * SE (alpha 0.05, power 0.80, normal approximation): a design sensitivity,
not a significance threshold. The statistics come from ``qml_resolution``; this script
only loads the archived probabilities and writes the table.

Inputs
------
--probs  npz with keys '<prefix>_<MODEL>_probs' (S x N) and 'y_te'
--hybrid json with models[k]['probs'] and 'y_te' (historical runs)
--extra  npz with keys '<NAME>_probs' (S x N) and 'y_te' (e.g. the core QML experiment)
--years  years_te.npy (default results/years_te.npy)
--pairs  e.g. RF-XGB RF-LSTM RF_2273-LSTM_QNN

Usage
-----
python exp10_paired_ci.py --probs results/json/results_rerun_train_only_canonical_probs.npz \
    --prefix train_only --pairs RF-XGB RF-LSTM XGB-LSTM --B 2000 --out out/results_exp10_paired_ci.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # use the repository copy if the package is not installed
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import rankdata  # noqa: E402

from qml_resolution import cluster_bootstrap_delta, delong_components, percentile_ci  # noqa: E402
from qml_resolution import paired_delong as delong_test  # noqa: E402

MDE_FACTOR = 2.80  # (z_0.975 + z_0.80) rounded, as in the paper
DEFAULT_PAIRS = ["RF-XGB", "RF-LSTM", "XGB-LSTM"]
DEFAULT_PAIRS_EXTRA = [*DEFAULT_PAIRS, "RF_2273-LSTM_QNN", "CTRL_LINEAR4D-LSTM_QNN", "PureLSTM-LSTM_QNN"]


# --------------------------------------------------------------------------------------------
# Legacy interface imported by exp13, exp16 and exp19 (kept so that they run unchanged).
# --------------------------------------------------------------------------------------------
def auc_fast(y: np.ndarray, p: np.ndarray) -> float:
    """Rank AUC of one score vector, without input checks (used inside resampling loops)."""
    r = rankdata(p)
    m = int(y.sum())
    n = len(y) - m
    return float((r[y == 1].sum() - m * (m + 1) / 2) / (m * n))


class Model:
    """Per-seed AUCs and seed-averaged DeLong components of one model."""

    def __init__(self, probs: np.ndarray, y: np.ndarray) -> None:
        self.P = np.atleast_2d(np.asarray(probs, dtype=float))
        self.S = self.P.shape[0]
        comp = delong_components(y, self.P)
        mean = comp.seed_mean()
        self.auc = comp.auc
        self.v10, self.v01 = mean.v10[0], mean.v01[0]

    @property
    def mean_auc(self) -> float:
        return float(self.auc.mean())


def paired_delong(A: Model, B: Model, y: np.ndarray) -> tuple[float, float, float, float]:  # noqa: N803
    """Return (delta, se, z, p) of the paired DeLong test of the seed means of A and B."""
    test = delong_test(y, A.P, B.P)
    return test.delta, test.se, test.z, test.p_value


def boot(A: Model, B: Model, y: np.ndarray, years: np.ndarray, B_rep: int,  # noqa: N803
         rng: np.random.Generator, resample_seeds: bool) -> np.ndarray:
    """Year-block (optionally years x seeds) bootstrap replicates of the seed-mean difference."""
    return cluster_bootstrap_delta(y, A.P, B.P, years, B_rep, rng, resample_seeds)


def load_models(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Collect the (S, N) probability matrices of every model and check that labels agree."""
    models: dict[str, np.ndarray] = {}
    y: np.ndarray | None = None

    def set_labels(new: np.ndarray, source: str) -> np.ndarray:
        if y is not None and not np.array_equal(new, y):
            raise ValueError(f"{source}: y_te differs from previously loaded labels")
        return new

    if args.probs:
        z = np.load(args.probs)
        y = set_labels(z["y_te"].astype(int), "--probs")
        prefix = args.prefix + "_"
        for k in z.files:
            if k.startswith(prefix) and k.endswith("_probs"):
                models[k[len(prefix) : -len("_probs")]] = z[k]
    if args.hybrid:
        with open(args.hybrid, encoding="utf-8") as fh:
            payload = json.load(fh)
        y = set_labels(np.array(payload["y_te"]), "--hybrid")
        for k, v in payload["models"].items():
            models[k] = np.array(v["probs"])
    if args.extra:
        z = np.load(args.extra)
        y = set_labels(z["y_te"].astype(int), "--extra")
        for k in z.files:
            if k.endswith("_probs"):
                models[k[: -len("_probs")]] = z[k]
    if y is None or not models:
        raise SystemExit("no models loaded: pass --probs, --hybrid or --extra")
    return models, y


def load_years(path: str | None, n: int) -> np.ndarray:
    """Calendar year of every test row (the cluster of the block bootstrap)."""
    if path:
        years = np.load(path)
    elif (ROOT / "results" / "years_te.npy").exists():
        years = np.load(ROOT / "results" / "years_te.npy")
    else:
        from noise_utils import load_test_partition

        *_, years, _ = load_test_partition()
    if len(years) != n:
        raise ValueError(f"years length ({len(years)}) != test labels length ({n})")
    return years


def split_pair(pair: str, names: set[str]) -> tuple[str, str]:
    """Split 'A-B' into two known model names; names may themselves contain '-'."""
    for i, ch in enumerate(pair):
        if ch == "-" and pair[:i] in names and pair[i + 1 :] in names:
            return pair[:i], pair[i + 1 :]
    raise KeyError(f"cannot split pair {pair!r} into two of {sorted(names)}")


def compare(
    y: np.ndarray, pa: np.ndarray, pb: np.ndarray, years: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict[str, object]:
    """All statistics of one pair. The bootstrap draws consume ``rng`` in a fixed order."""
    test = delong_test(y, pa, pb)
    block = cluster_bootstrap_delta(y, pa, pb, years, n_boot, rng)
    two_way = cluster_bootstrap_delta(y, pa, pb, years, n_boot, rng, resample_seeds=True)
    se_block = float(block.std(ddof=1))
    return dict(
        S_A=int(np.atleast_2d(pa).shape[0]),
        S_B=int(np.atleast_2d(pb).shape[0]),
        delta=test.delta,
        se_delong=test.se,
        z=test.z,
        p_delong=test.p_value,
        ci_delong=[test.delta - 1.96 * test.se, test.delta + 1.96 * test.se],
        ci_block=list(percentile_ci(block)),
        se_block=se_block,
        ci_two_way=list(percentile_ci(two_way)),
        se_two_way=float(two_way.std(ddof=1)),
        mde_delong=MDE_FACTOR * test.se,
        mde_block=MDE_FACTOR * se_block,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--probs")
    ap.add_argument("--prefix", default="train_only")
    ap.add_argument("--hybrid")
    ap.add_argument("--extra")
    ap.add_argument("--years")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "out" / "results_exp10_paired_ci.json"))
    args = ap.parse_args()

    models, y = load_models(args)
    years = load_years(args.years, len(y))
    pairs = args.pairs or (DEFAULT_PAIRS_EXTRA if args.extra else DEFAULT_PAIRS)
    rng = np.random.default_rng(args.seed)
    print("models:", {k: v.shape for k, v in models.items()})
    print(f"test years {years.min()}-{years.max()} | clusters {len(np.unique(years))}")

    header = f"\n{'pair':<26}{'Delta':>8}{'SE_DL':>8}{'z':>6}{'p':>7} | {'block 95% CI':^20}{'SE_blk':>8}"
    print(header + f" | {'2-way 95% CI':^20} | MDE_iid MDE_blk")
    results = {}
    for pair in pairs:
        a, b = split_pair(pair, set(models))
        r = compare(y, models[a], models[b], years, args.B, rng)
        results[pair] = r
        ci, ci2 = r["ci_block"], r["ci_two_way"]
        print(
            f"{pair:<26}{r['delta']:8.4f}{r['se_delong']:8.4f}{r['z']:6.2f}{r['p_delong']:7.3f}"
            f" | [{ci[0]:7.4f},{ci[1]:7.4f}]{r['se_block']:8.4f} | [{ci2[0]:7.4f},{ci2[1]:7.4f}]"
            f" | {r['mde_delong']:.3f}   {r['mde_block']:.3f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(n_test=len(y), n_clusters=int(len(np.unique(years))), B=args.B, pairs=results)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
