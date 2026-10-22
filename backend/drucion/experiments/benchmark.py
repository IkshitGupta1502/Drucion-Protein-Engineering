"""Controlled comparison of selection strategies under identical conditions.

Fairness contract enforced here
-------------------------------
1. Every method sees the **same candidate pool** (one assay load, reused).
2. Every method uses the **same feature matrix** (computed once, label-free).
3. For a given seed, every method starts from the **identical initial
   observations** (drawn once per seed and passed in).
4. Every method gets the **same experimental budget**
   (``n_initial + batch_size * n_rounds``).
5. Each run gets a **fresh oracle**, so budget counters never bleed across
   methods, and no method can observe another's reveals.

Only the selection rule differs. That is the whole experiment.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ALL_METHODS, RESULTS_DIR, ExperimentConfig
from ..data.proteingym import load_assay
from ..embeddings.features import build_features
from ..evaluation.metrics import (
    RunMetrics,
    aggregate,
    budget_to_match,
    compute_run_metrics,
    curve_band,
    paired_comparison,
)
from ..optimization.loop import ClosedLoopOptimizer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Everything needed to plot and report a benchmark."""

    assay_id: str
    config: dict
    methods: list[str]
    seeds: list[int]
    pool_stats: dict
    runs: list[RunMetrics] = field(default_factory=list)
    #: method -> {"n_measured", "mean", "std"} for the best-so-far curve.
    curves: dict = field(default_factory=dict)
    #: method -> {"n_measured", "mean", "std"} for holdout Spearman per round.
    surrogate_curves: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    #: Seed-paired comparison of each method against random selection.
    paired: dict = field(default_factory=dict)
    #: Budget each method needs to match random's final result.
    efficiency: dict = field(default_factory=dict)
    feature_info: dict = field(default_factory=dict)
    wall_time_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "assay_id": self.assay_id,
            "config": self.config,
            "methods": self.methods,
            "seeds": self.seeds,
            "pool_stats": self.pool_stats,
            "feature_info": self.feature_info,
            "runs": [r.to_dict() for r in self.runs],
            "curves": self.curves,
            "surrogate_curves": self.surrogate_curves,
            "summary": self.summary,
            "paired": self.paired,
            "efficiency": self.efficiency,
            "wall_time_s": self.wall_time_s,
            "notes": self.notes,
        }


def run_benchmark(
    config: ExperimentConfig,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    holdout_size: int = 500,
    progress=None,
) -> BenchmarkResult:
    """Run every method across every seed and aggregate the results."""
    methods = list(methods or ALL_METHODS)
    seeds = list(seeds if seeds is not None else range(5))
    t_start = time.time()

    # --- shared, label-free setup -------------------------------------
    base_assay = load_assay(config.assay_id, pool_size=config.pool_size)
    features = build_features(base_assay.pool, config.embedding)
    pool_stats = base_assay.oracle.ground_truth_summary()
    top1 = base_assay.oracle.top_quantile_indices(0.99)
    top10 = base_assay.oracle.top_quantile_indices(0.90)
    pool_size = len(base_assay.pool)

    # Fixed evaluation holdout, drawn once without looking at any label and
    # shared by every run: excluded from the selectable pool so it is never
    # trained on, never measured, and never charged to any method's budget.
    holdout: list[int] = []
    if holdout_size:
        hrng = np.random.default_rng(12345)
        n_hold = min(holdout_size, max(0, pool_size - config.total_budget - 10))
        if n_hold > 0:
            holdout = sorted(
                hrng.choice(pool_size, size=n_hold, replace=False).tolist()
            )

    # Initial designs: one per seed, shared by all methods for that seed, and
    # drawn from outside the holdout.
    selectable = np.setdiff1d(np.arange(pool_size), np.asarray(holdout, dtype=int))
    initial_by_seed = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        n_init = min(config.loop.n_initial, selectable.size)
        initial_by_seed[seed] = sorted(
            rng.choice(selectable, size=n_init, replace=False).tolist()
        )

    runs: list[RunMetrics] = []
    curves_by_method: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        m: [] for m in methods
    }
    surr_by_method: dict[str, list[list[float]]] = {m: [] for m in methods}
    notes: list[str] = []
    total = len(methods) * len(seeds)
    done = 0

    for method in methods:
        for seed in seeds:
            run_cfg = copy.deepcopy(config)
            run_cfg.loop.acquisition = method
            run_cfg.loop.seed = seed

            # Fresh oracle per run: no cross-run budget contamination.
            assay = load_assay(config.assay_id, pool_size=config.pool_size)
            opt = ClosedLoopOptimizer(
                assay,
                run_cfg,
                features=features,
                zero_shot_scores=features.zero_shot,
                initial_indices=initial_by_seed[seed],
                holdout_indices=holdout,
            )
            opt.run()

            n, best = opt.learning_curve()
            if opt.oracle.n_queries != config.total_budget:
                notes.append(
                    f"{method}/seed{seed}: consumed {opt.oracle.n_queries} "
                    f"measurements, expected {config.total_budget}"
                )
            if opt.surrogate.fallback_reason:
                notes.append(
                    f"{method}/seed{seed}: surrogate fell back to "
                    f"{opt.surrogate.kind} ({opt.surrogate.fallback_reason})"
                )
            runs.append(
                compute_run_metrics(
                    method=method,
                    seed=seed,
                    n_measured=n,
                    best_so_far=best,
                    measured_indices=opt.observed,
                    pool_stats=pool_stats,
                    top1pct=top1,
                    top10pct=top10,
                    best_mutant=opt.best_mutant,
                    surrogate_spearmans=[
                        r.holdout_spearman for r in opt.history[1:]
                    ],
                )
            )
            curves_by_method[method].append((n, best))
            surr_by_method[method].append(
                [
                    (r.holdout_spearman if r.holdout_spearman is not None else np.nan)
                    for r in opt.history[1:]
                ]
            )
            done += 1
            logger.info(
                "[%d/%d] %s seed=%d best=%.4f (%s)",
                done,
                total,
                method,
                seed,
                opt.best_measured,
                opt.best_mutant,
            )
            if progress:
                progress(done, total, method, seed, runs[-1])

    curves = {}
    for method, cs in curves_by_method.items():
        x, mean, std = curve_band(cs)
        curves[method] = {
            "n_measured": x.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    surrogate_curves = {}
    for method, rows in surr_by_method.items():
        if not rows or not rows[0]:
            continue
        M = np.array(rows, dtype=float)
        rounds = list(range(1, M.shape[1] + 1))
        n_grid = [
            config.loop.n_initial + config.loop.batch_size * (r - 1) for r in rounds
        ]
        with np.errstate(invalid="ignore"):
            surrogate_curves[method] = {
                "round": rounds,
                "n_measured": n_grid,
                "mean": np.nanmean(M, axis=0).tolist(),
                "std": np.nanstd(M, axis=0).tolist(),
            }

    return BenchmarkResult(
        assay_id=config.assay_id,
        config=config.to_dict(),
        methods=methods,
        seeds=seeds,
        pool_stats=pool_stats,
        runs=runs,
        curves=curves,
        surrogate_curves=surrogate_curves,
        summary=aggregate(runs),
        paired=paired_comparison(runs),
        efficiency=budget_to_match(curves),
        feature_info=features.to_dict() | {"holdout_size": len(holdout)},
        wall_time_s=time.time() - t_start,
        notes=notes,
    )


def save_benchmark(
    result: BenchmarkResult, out_dir: Path | None = None, tag: str = ""
) -> dict[str, Path]:
    """Write JSON, per-run CSV and curve CSV. Returns the paths written."""
    import pandas as pd

    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.assay_id}{('_' + tag) if tag else ''}"

    json_path = out_dir / f"{stem}_benchmark.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2))

    runs_path = out_dir / f"{stem}_runs.csv"
    pd.DataFrame([r.to_dict() for r in result.runs]).to_csv(runs_path, index=False)

    rows = []
    for method, c in result.curves.items():
        for n, m, s in zip(c["n_measured"], c["mean"], c["std"]):
            rows.append(
                {"method": method, "n_measured": n, "best_mean": m, "best_std": s}
            )
    curves_path = out_dir / f"{stem}_curves.csv"
    pd.DataFrame(rows).to_csv(curves_path, index=False)

    summary_path = out_dir / f"{stem}_summary.csv"
    pd.DataFrame(result.summary).T.to_csv(summary_path)

    return {
        "json": json_path,
        "runs": runs_path,
        "curves": curves_path,
        "summary": summary_path,
    }
