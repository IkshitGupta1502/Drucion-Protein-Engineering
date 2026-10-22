"""Publication-quality benchmark figures.

Figure 1 is the headline result: best measured fitness discovered versus
cumulative experimental evaluations, with +/-1 std bands across seeds and the
pool maximum drawn as the unreachable ceiling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import RESULTS_DIR

if TYPE_CHECKING:  # pragma: no cover
    from ..experiments.benchmark import BenchmarkResult

METHOD_LABELS = {
    "random": "Random selection",
    "greedy": "Greedy surrogate",
    "ucb": "Bayesian AL (UCB)",
    "ei": "Bayesian AL (EI)",
    "evolutionary": "ESM-guided evolutionary",
}

# Colour-blind-safe qualitative palette (Okabe-Ito).
METHOD_COLORS = {
    "random": "#999999",
    "evolutionary": "#E69F00",
    "greedy": "#56B4E9",
    "ei": "#009E73",
    "ucb": "#D55E00",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )
    return plt


def plot_learning_curves(
    result: "BenchmarkResult", out_dir: Path | None = None, tag: str = ""
) -> Path:
    plt = _style()
    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in result.methods:
        c = result.curves[method]
        x = np.asarray(c["n_measured"], dtype=float)
        m = np.asarray(c["mean"], dtype=float)
        s = np.asarray(c["std"], dtype=float)
        color = METHOD_COLORS.get(method, None)
        ax.plot(
            x, m, label=METHOD_LABELS.get(method, method), color=color, linewidth=2.0
        )
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.16, linewidth=0)

    pool_max = result.pool_stats["pool_max"]
    ax.axhline(
        pool_max,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"Pool maximum ({pool_max:.2f})",
    )
    ax.axhline(
        result.pool_stats["p99"],
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.5,
        label=f"Pool 99th pct ({result.pool_stats['p99']:.2f})",
    )

    n_seeds = len(result.seeds)
    ax.set_xlabel("Cumulative experimental measurements revealed")
    ax.set_ylabel("Best measured fitness (DMS score)")
    ax.set_title(
        f"{result.assay_id}\nBest measured fitness vs. experimental budget "
        f"(mean ±1 s.d., {n_seeds} seeds)",
        fontsize=10.5,
    )
    ax.legend(loc="lower right", fontsize=8.5, ncols=2)
    fig.tight_layout()
    path = out_dir / f"{result.assay_id}{('_' + tag) if tag else ''}_learning_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metric_bars(
    result: "BenchmarkResult", out_dir: Path | None = None, tag: str = ""
) -> Path:
    plt = _style()
    out_dir = Path(out_dir or RESULTS_DIR)
    fields = [
        ("auc_normalised", "Normalised AUC\n(higher is better)"),
        ("final_regret", "Final regret\n(lower is better)"),
        ("top1pct_recall", "Top-1% variants found\n(higher is better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (field, title) in zip(axes, fields):
        means = [result.summary[m][f"{field}_mean"] for m in result.methods]
        stds = [result.summary[m][f"{field}_std"] for m in result.methods]
        colors = [METHOD_COLORS.get(m) for m in result.methods]
        ax.bar(
            range(len(result.methods)),
            means,
            yerr=stds,
            color=colors,
            capsize=3,
            error_kw={"linewidth": 1.0},
        )
        ax.set_xticks(range(len(result.methods)))
        ax.set_xticklabels(
            [METHOD_LABELS.get(m, m).replace(" ", "\n") for m in result.methods],
            fontsize=7.5,
        )
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{result.assay_id} — summary metrics", fontsize=10.5)
    fig.tight_layout()
    path = out_dir / f"{result.assay_id}{('_' + tag) if tag else ''}_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_benchmark(
    result: "BenchmarkResult", out_dir: Path | None = None, tag: str = ""
) -> list[Path]:
    return [
        plot_learning_curves(result, out_dir, tag),
        plot_metric_bars(result, out_dir, tag),
    ]
