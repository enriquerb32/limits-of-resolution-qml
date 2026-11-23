"""Protocol report for paired model comparisons on a small, clustered test set.

Implements the inferential items of the reporting protocol of the paper (Table 14): from archived test
predictions, labels, cluster labels and seeds it returns, for each declared pair, the paired difference of
seed-mean AUCs with

* the primary interval and p-value (CR2 with Bell-McCaffrey degrees of freedom over the clusters),
* DeLong as the reference under independence,
* a two-way (clusters x seeds) bootstrap interval, with seeds drawn jointly only for pairs declared as
  sharing a fitted component,
* the cluster/iid ratio, the nominal DeLong MDE and the MDE of the primary procedure,
* Holm-adjusted primary p-values within a declared family,
* a power curve under declared assumptions: the primary standard error, optionally rescaled to another test
  size by ``sqrt(n / n_target)`` (which assumes that added cases bring independent information and that the
  covariance between models is unchanged).

Command line (scores in one ``.npz``: ``y``, a cluster array and one ``(S, n)`` array per model)::

    python -m qml_resolution.report scores.npz --clusters years --pair RF:LSTM --pair RF:QNN \
        --family RF:LSTM --family RF:QNN --shared QTL:LSTM --deltas 0.01 0.02 --n-target 2100 --out report.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import nct
from scipy.stats import t as student_t

from qml_resolution.auc import paired_delong
from qml_resolution.power import mde
from qml_resolution.resampling import cluster_bootstrap_delta, cluster_robust_delta, percentile_ci

Pair = tuple[str, str]


@dataclass(frozen=True)
class Assumptions:
    """Declared settings of a report; stored with every report so that it can be reproduced."""

    alpha: float = 0.05
    power: float = 0.80
    n_boot: int = 2000
    random_state: int = 0
    power_deltas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03)
    n_target: int | None = None


@dataclass(frozen=True)
class PairReport:
    """Everything the protocol asks for one comparison ``A - B``."""

    pair: str
    delta: float
    se_delong: float
    p_delong: float
    se_cr2: float
    df: float
    ci_primary: tuple[float, float]
    p_primary: float
    resolved: bool
    ci_two_way: tuple[float, float]
    seeds_resampled: str
    ratio_cluster_iid: float
    mde_delong: float
    mde_primary: float
    p_holm: float | None = None
    resolved_after_holm: bool | None = None
    power: dict[str, dict[str, float]] = field(default_factory=dict)


def power_t(delta: float, se: float, df: float, alpha: float = 0.05) -> float:
    """Power of a two-sided t test of zero when the true difference is ``delta`` (noncentral t)."""
    if se <= 0:
        raise ValueError("se must be positive")
    crit = float(student_t.ppf(1 - alpha / 2, df))
    nc = delta / se
    return float(nct.sf(crit, df, nc) + nct.cdf(-crit, df, nc))


def mde_t(se: float, df: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """``(t_{1-alpha/2} + t_{power}) * se`` with ``df`` degrees of freedom.

    This is the usual approximation of the noncentral t; :func:`power_t` gives the exact power.
    """
    return float((student_t.ppf(1 - alpha / 2, df) + student_t.ppf(power, df)) * se)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def compare_pair(
    y: ArrayLike,
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    clusters: ArrayLike,
    name: str = "A-B",
    shared_seeds: bool = False,
    assumptions: Assumptions = Assumptions(),
) -> PairReport:
    """Protocol quantities for one paired comparison."""
    a = assumptions
    cr = cluster_robust_delta(y, scores_a, scores_b, clusters)
    dl = paired_delong(y, scores_a, scores_b)
    lo, hi = cr.ci(1 - a.alpha)
    reps = cluster_bootstrap_delta(
        y,
        scores_a,
        scores_b,
        clusters,
        n_boot=a.n_boot,
        random_state=a.random_state,
        resample_seeds=True,
        shared_seeds=shared_seeds,
    )
    two_way = percentile_ci(reps, 1 - a.alpha)
    n = int(np.asarray(y).size)
    power: dict[str, dict[str, float]] = {}
    for d in a.power_deltas:
        row = {"at_this_test_set": power_t(d, cr.se_cr2, cr.df, a.alpha)}
        if a.n_target:
            row[f"at_n_{a.n_target}"] = power_t(d, cr.se_cr2 * np.sqrt(n / a.n_target), cr.df, a.alpha)
        power[f"{d:g}"] = row
    return PairReport(
        pair=name,
        delta=cr.delta,
        se_delong=dl.se,
        p_delong=dl.p_value,
        se_cr2=cr.se_cr2,
        df=cr.df,
        ci_primary=(lo, hi),
        p_primary=cr.p_value,
        resolved=bool(lo > 0 or hi < 0),
        ci_two_way=two_way,
        seeds_resampled="joint" if shared_seeds else "independent",
        ratio_cluster_iid=cr.se_cr1 / cr.se_iid,
        mde_delong=float(mde(dl.se, a.alpha, a.power)),
        mde_primary=mde_t(cr.se_cr2, cr.df, a.alpha, a.power),
        power=power,
    )


def report(
    y: ArrayLike,
    scores: Mapping[str, ArrayLike],
    clusters: ArrayLike,
    pairs: Iterable[Pair],
    shared: Iterable[Pair] = (),
    family: Iterable[Pair] = (),
    assumptions: Assumptions = Assumptions(),
) -> dict[str, Any]:
    """Protocol report for several pairs: per-pair quantities and Holm adjustment within ``family``."""
    pairs = list(pairs)
    shared_set = {frozenset(p) for p in shared}
    fam = [f"{a}-{b}" for a, b in family]
    out: dict[str, PairReport] = {}
    for a_name, b_name in pairs:
        for m in (a_name, b_name):
            if m not in scores:
                raise KeyError(f"model {m!r} not in scores")
        key = f"{a_name}-{b_name}"
        out[key] = compare_pair(
            y, scores[a_name], scores[b_name], clusters, key, frozenset((a_name, b_name)) in shared_set, assumptions
        )
    missing = [k for k in fam if k not in out]
    if missing:
        raise KeyError(f"family pairs not among the pairs: {missing}")
    if fam:
        adj = holm([out[k].p_primary for k in fam])
        for k, p in zip(fam, adj, strict=True):
            out[k] = replace(out[k], p_holm=p, resolved_after_holm=bool(p < assumptions.alpha))
    labels = np.asarray(y)
    return {
        "assumptions": asdict(assumptions),
        "n_test": int(labels.size),
        "prevalence": float(labels.mean()),
        "n_clusters": int(np.unique(np.asarray(clusters)).size),
        "pairs": {k: asdict(v) for k, v in out.items()},
    }


def _pair(text: str) -> Pair:
    a, sep, b = text.partition(":")
    if not sep or not a or not b:
        raise argparse.ArgumentTypeError(f"expected MODEL_A:MODEL_B, got {text!r}")
    return a, b


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: read an archive of scores, print the report and optionally write it as JSON."""
    ap = argparse.ArgumentParser(prog="python -m qml_resolution.report", description=__doc__.split("\n\n")[0])
    ap.add_argument("npz", help="archive with labels, cluster labels and one (S, n) score array per model")
    ap.add_argument("--labels", default="y", help="key of the 0/1 labels (default y)")
    ap.add_argument("--clusters", required=True, help="key of the cluster labels (e.g. test years)")
    ap.add_argument("--pair", type=_pair, action="append", required=True, help="A:B, repeatable")
    ap.add_argument("--shared", type=_pair, action="append", default=[], help="pairs that share a fitted component")
    ap.add_argument("--family", type=_pair, action="append", default=[], help="pairs of the Holm family")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03])
    ap.add_argument("--n-target", type=int, default=None, help="test size for the rescaled power curve")
    ap.add_argument("--out", default=None, help="write the report as JSON")
    a = ap.parse_args(argv)
    z = np.load(a.npz)
    models = {k: z[k] for k in z.files if k not in (a.labels, a.clusters)}
    res = report(
        z[a.labels],
        models,
        z[a.clusters],
        a.pair,
        a.shared,
        a.family,
        Assumptions(a.alpha, a.power, a.n_boot, a.seed, tuple(a.deltas), a.n_target),
    )
    print(f"n={res['n_test']} prevalence={res['prevalence']:.3f} clusters={res['n_clusters']}")
    for k, r in res["pairs"].items():
        holm_txt = "" if r["p_holm"] is None else f" Holm p={r['p_holm']:.3f}"
        lo, hi = r["ci_primary"]
        print(
            f"{k:28s} d={r['delta']:+.4f}  CR2 [{lo:+.3f},{hi:+.3f}] p={r['p_primary']:.3f}"
            f"{holm_txt}  DeLong p={r['p_delong']:.3f}  ratio {r['ratio_cluster_iid']:.2f}"
            f"  MDE {r['mde_primary']:.3f} (DeLong {r['mde_delong']:.3f})"
        )
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
