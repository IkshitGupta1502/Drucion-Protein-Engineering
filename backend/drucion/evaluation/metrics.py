"""Benchmark metrics for closed-loop protein optimization.

Every metric is computed from the learning curve of *measured* fitness values
plus the pool's ground-truth statistics. Ground truth enters only here, after
all selection decisions have already been made and the budget spent.

Primary metric
--------------
``best_measured`` as a function of ``n_measured`` — best experimentally
measured fitness discovered versus cumulative experimental evaluations. The
goal of the system is to move that curve up and to the left.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RunMetrics:
    """Metrics for a single (method, seed) run."""

    method: str
    seed: int
    n_measured: int
    #: Best measured fitness discovered within budget.
    best_measured: float
    best_mutant: str
    #: Normalised area under the best-so-far curve: average of
    #: best-so-far over the budget, rescaled so 0 = pool mean and 1 = pool max.
    auc_normalised: float
    #: Simple regret: pool maximum minus best discovered.
    final_regret: float
    #: Fraction of the pool's top-1% variants that were measured.
    top1pct_recall: float
    #: Fraction of the pool's top-10% variants that were measured.
    top10pct_recall: float
    #: Measurements needed to first reach the threshold, or None if never.
    evals_to_threshold: int | None
    threshold: float
    #: Mean held-out Spearman of the surrogate across rounds (None if no
    #: surrogate predictions were recorded).
    mean_surrogate_spearman: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def normalised_auc(
    n_measured: np.ndarray,
    best_so_far: np.ndarray,
    pool_mean: float,
    pool_max: float,
) -> float:
    """Budget-weighted average of the best-so-far curve, rescaled to [0, 1].

    Uses the trapezoid rule over cumulative measurements, so a method that
    reaches a high value *earlier* scores higher than one that ends in the
    same place. 0 corresponds to a curve sitting at the pool mean, 1 to a
    curve that is at the pool maximum from the first measurement.
    """
    n = np.asarray(n_measured, dtype=float)
    b = np.asarray(best_so_far, dtype=float)
    if n.size < 2:
        return float("nan")
    area = np.trapezoid(b, n)
    span = n[-1] - n[0]
    if span <= 0:
        return float("nan")
    mean_level = area / span
    denom = pool_max - pool_mean
    if abs(denom) < 1e-12:
        return float("nan")
    return float((mean_level - pool_mean) / denom)


def evals_to_threshold(
    n_measured: np.ndarray, best_so_far: np.ndarray, threshold: float
) -> int | None:
    """First cumulative measurement count at which best-so-far >= threshold."""
    b = np.asarray(best_so_far, dtype=float)
    hits = np.flatnonzero(b >= threshold)
    if hits.size == 0:
        return None
    return int(np.asarray(n_measured)[hits[0]])


def compute_run_metrics(
    method: str,
    seed: int,
    n_measured: np.ndarray,
    best_so_far: np.ndarray,
    measured_indices: list[int],
    pool_stats: dict[str, float],
    top1pct: set[int],
    top10pct: set[int],
    best_mutant: str,
    threshold_quantile: float = 0.99,
    surrogate_spearmans: list[float] | None = None,
) -> RunMetrics:
    """Assemble every metric for one run."""
    threshold = pool_stats[f"p{int(threshold_quantile * 100)}"]
    measured = set(measured_indices)
    rhos = [r for r in (surrogate_spearmans or []) if r is not None]
    return RunMetrics(
        method=method,
        seed=seed,
        n_measured=int(n_measured[-1]),
        best_measured=float(best_so_far[-1]),
        best_mutant=best_mutant,
        auc_normalised=normalised_auc(
            n_measured, best_so_far, pool_stats["pool_mean"], pool_stats["pool_max"]
        ),
        final_regret=float(pool_stats["pool_max"] - best_so_far[-1]),
        top1pct_recall=(len(measured & top1pct) / len(top1pct)) if top1pct else 0.0,
        top10pct_recall=(
            (len(measured & top10pct) / len(top10pct)) if top10pct else 0.0
        ),
        evals_to_threshold=evals_to_threshold(n_measured, best_so_far, threshold),
        threshold=float(threshold),
        mean_surrogate_spearman=(
            float(np.mean(rhos)) if rhos else None
        ),
    )


def aggregate(runs: list[RunMetrics]) -> dict[str, dict]:
    """Mean and standard deviation of each metric, grouped by method.

    ``evals_to_threshold`` is summarised as a hit rate plus the mean over the
    runs that reached the threshold, because averaging a censored value would
    silently flatter methods that usually fail to get there.
    """
    out: dict[str, dict] = {}
    methods = sorted({r.method for r in runs}, key=lambda m: m)
    for method in methods:
        rows = [r for r in runs if r.method == method]
        summary: dict = {"n_seeds": len(rows)}
        for field in (
            "best_measured",
            "auc_normalised",
            "final_regret",
            "top1pct_recall",
            "top10pct_recall",
        ):
            vals = np.array([getattr(r, field) for r in rows], dtype=float)
            summary[f"{field}_mean"] = float(np.nanmean(vals))
            summary[f"{field}_std"] = float(np.nanstd(vals))
        reached = [
            r.evals_to_threshold for r in rows if r.evals_to_threshold is not None
        ]
        summary["threshold_hit_rate"] = len(reached) / len(rows) if rows else 0.0
        summary["evals_to_threshold_mean"] = (
            float(np.mean(reached)) if reached else None
        )
        summary["evals_to_threshold_std"] = (
            float(np.std(reached)) if reached else None
        )
        rhos = [
            r.mean_surrogate_spearman
            for r in rows
            if r.mean_surrogate_spearman is not None
        ]
        summary["surrogate_spearman_mean"] = float(np.mean(rhos)) if rhos else None
        out[method] = summary
    return out


#: Metrics where a *lower* value is better.
LOWER_IS_BETTER = {"final_regret", "evals_to_threshold"}


def paired_comparison(
    runs: list[RunMetrics],
    baseline: str = "random",
    metrics: tuple[str, ...] = (
        "best_measured",
        "auc_normalised",
        "top1pct_recall",
        "evals_to_threshold",
    ),
) -> dict:
    """Seed-paired comparison of each method against ``baseline``.

    The benchmark gives every method the *same* initial observations for a
    given seed, so seeds are paired and a paired test is the statistically
    appropriate one — it removes the large between-seed variance that makes
    the unpaired standard deviations look uninformative.

    Reports, per method and metric: number of seeds the method wins, the mean
    paired difference, and a two-sided Wilcoxon signed-rank p-value. The
    p-value is ``None`` when the test is undefined (e.g. every difference is
    zero) or fewer than 5 paired seeds are available. With ~10 seeds these are
    indicative, not confirmatory: they are reported as-is, including when a
    method loses.
    """
    from scipy.stats import wilcoxon

    by_method: dict[str, dict[int, RunMetrics]] = {}
    for r in runs:
        by_method.setdefault(r.method, {})[r.seed] = r
    if baseline not in by_method:
        return {}

    out: dict = {"baseline": baseline, "methods": {}}
    base = by_method[baseline]
    for method, rows in by_method.items():
        if method == baseline:
            continue
        shared = sorted(set(rows) & set(base))
        entry: dict = {"n_paired_seeds": len(shared), "metrics": {}}
        for metric in metrics:
            a, b = [], []
            for seed in shared:
                va = getattr(rows[seed], metric)
                vb = getattr(base[seed], metric)
                if va is None or vb is None:
                    continue
                a.append(float(va))
                b.append(float(vb))
            if not a:
                continue
            arr_a, arr_b = np.array(a), np.array(b)
            diff = arr_a - arr_b
            if metric in LOWER_IS_BETTER:
                wins = int((diff < 0).sum())
            else:
                wins = int((diff > 0).sum())
            p = None
            if len(diff) >= 5 and np.any(diff != 0):
                try:
                    p = float(wilcoxon(arr_a, arr_b).pvalue)
                except ValueError:
                    p = None
            entry["metrics"][metric] = {
                "n": len(diff),
                "wins": wins,
                "ties": int((diff == 0).sum()),
                "mean_difference": float(diff.mean()),
                "wilcoxon_p": p,
                "lower_is_better": metric in LOWER_IS_BETTER,
            }
        out["methods"][method] = entry
    return out


def budget_to_match(
    curves: dict, baseline: str = "random"
) -> dict:
    """How many measurements each method needs to match the baseline's final result.

    This is the experimental-efficiency claim in its most direct form: if the
    baseline's whole budget buys fitness ``F``, at what budget does each method
    first reach ``F``?
    """
    if baseline not in curves:
        return {}
    base_mean = np.asarray(curves[baseline]["mean"], dtype=float)
    target = float(base_mean[-1])
    base_n = np.asarray(curves[baseline]["n_measured"], dtype=float)
    full_budget = float(base_n[-1])

    out: dict = {"baseline": baseline, "target": target, "methods": {}}
    for method, c in curves.items():
        y = np.asarray(c["mean"], dtype=float)
        n = np.asarray(c["n_measured"], dtype=float)
        hits = np.flatnonzero(y >= target)
        reached = int(n[hits[0]]) if hits.size else None
        out["methods"][method] = {
            "budget_to_match": reached,
            "savings_fraction": (
                (full_budget - reached) / full_budget if reached else None
            ),
            "final_value": float(y[-1]),
        }
    return out


def curve_band(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean +/- 1 std of a set of best-so-far curves on a shared x grid.

    All runs in a benchmark share the same budget schedule, so the x grids are
    identical; this validates that assumption rather than interpolating over it.
    """
    if not curves:
        raise ValueError("No curves to aggregate")
    x0 = curves[0][0]
    for x, _ in curves:
        if not np.array_equal(x, x0):
            raise ValueError(
                "Learning curves have mismatched budget grids; refusing to "
                "aggregate across different experimental budgets."
            )
    Y = np.vstack([y for _, y in curves])
    return np.asarray(x0), Y.mean(axis=0), Y.std(axis=0)
