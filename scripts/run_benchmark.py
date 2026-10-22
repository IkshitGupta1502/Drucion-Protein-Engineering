#!/usr/bin/env python
"""Run the Drucion benchmark on a real ProteinGym assay and save results.

Example::

    python scripts/run_benchmark.py --assay A4GRB6_PSEAI_Chen_2020 --seeds 5
"""

from __future__ import annotations

import argparse
import logging

from drucion.config import ALL_METHODS, ExperimentConfig
from drucion.evaluation.plots import plot_benchmark
from drucion.experiments.benchmark import run_benchmark, save_benchmark


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assay", default=ExperimentConfig().assay_id)
    p.add_argument("--methods", nargs="+", default=list(ALL_METHODS))
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0..n-1)")
    p.add_argument("--n-initial", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--pool-size", type=int, default=None)
    p.add_argument("--surrogate", default="gp", choices=["gp", "rf"])
    p.add_argument("--featurizer", default="esm2")
    p.add_argument("--tag", default="")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = ExperimentConfig(assay_id=args.assay, pool_size=args.pool_size)
    cfg.loop.n_initial = args.n_initial
    cfg.loop.batch_size = args.batch_size
    cfg.loop.n_rounds = args.rounds
    cfg.surrogate.kind = args.surrogate
    cfg.embedding.featurizer = args.featurizer

    print(
        f"Assay={cfg.assay_id} budget={cfg.total_budget} "
        f"methods={args.methods} seeds={args.seeds}",
        flush=True,
    )

    result = run_benchmark(cfg, methods=args.methods, seeds=list(range(args.seeds)))
    paths = save_benchmark(result, tag=args.tag)

    print("\n=== Summary (mean +/- std across seeds) ===")
    print(f"pool max = {result.pool_stats['pool_max']:.4f}  "
          f"p99 = {result.pool_stats['p99']:.4f}  "
          f"mean = {result.pool_stats['pool_mean']:.4f}")
    hdr = f"{'method':<14}{'best':>18}{'AUC':>16}{'regret':>16}{'top1%rec':>14}{'evals→p99':>14}"
    print(hdr)
    for method in result.methods:
        s = result.summary[method]
        e = s["evals_to_threshold_mean"]
        etxt = f"{e:.0f} ({s['threshold_hit_rate']*100:.0f}%)" if e else "never"
        print(
            f"{method:<14}"
            f"{s['best_measured_mean']:>10.4f}±{s['best_measured_std']:<7.4f}"
            f"{s['auc_normalised_mean']:>9.3f}±{s['auc_normalised_std']:<6.3f}"
            f"{s['final_regret_mean']:>9.4f}±{s['final_regret_std']:<6.4f}"
            f"{s['top1pct_recall_mean']*100:>9.1f}%    "
            f"{etxt:>12}"
        )
    if result.notes:
        print("\nNotes:")
        for n in dict.fromkeys(result.notes):
            print(" -", n)

    if not args.no_plots:
        figs = plot_benchmark(result)
        for f in figs:
            print("plot:", f)
    for k, v in paths.items():
        print(f"{k}: {v}")
    print(f"wall time: {result.wall_time_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
