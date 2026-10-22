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


# ---------------------------------------------------------------------------
# Representation comparison (agent prior and its controls)
# ---------------------------------------------------------------------------

#: Display order and labels for the agent-prior representation check. The
#: order is the argument the figure makes: what we had, what the prior adds,
#: and what the controls rule out.
AGENT_ROWS: tuple[tuple[str, str, str], ...] = (
    ("esm2", "ESM-2 embeddings", "baseline"),
    ("esm2+zero_shot", "+ ESM-2 zero-shot  (current setup)", "baseline"),
    ("esm2+zero_shot+agent_blind", "+ LLM prior, blind", "agent"),
    ("esm2+zero_shot+agent", "+ LLM prior, told the protein", "agent"),
    ("esm2+zero_shot+agent_random_positions", "control: random positions", "control"),
    ("esm2+zero_shot+agent_shuffled", "control: shuffled scores", "control"),
)

#: Two chromatic slots (validated for colour-vision deficiency) plus a
#: deliberately recessive grey for the controls. Every bar is directly
#: labelled, so identity never rests on colour alone.
AGENT_COLORS = {"baseline": "#0072B2", "agent": "#D55E00", "control": "#999999"}

AGENT_ROLE_LABELS = {
    "baseline": "Existing features",
    "agent": "LLM structural prior",
    "control": "Ablation control",
}


def plot_representation_check(
    payload: dict, out_dir: Path | None = None, tag: str = ""
) -> Path:
    """Horizontal bars of held-out Spearman for each representation.

    ``payload`` is the JSON written by ``scripts/representation_check.py``.
    The reference line is the configuration the benchmark currently uses, so
    a bar's position relative to it reads directly as "better or worse than
    what we have now".
    """
    plt = _style()
    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = payload.get("results", {})
    rows = [
        (label, role, results[key])
        for key, label, role in AGENT_ROWS
        if key in results and "spearman_mean" in results[key]
    ]
    if not rows:
        raise ValueError("No usable representation rows in payload")

    labels = [r[0] for r in rows]
    roles = [r[1] for r in rows]
    means = np.array([r[2]["spearman_mean"] for r in rows], dtype=float)
    stds = np.array([r[2].get("spearman_std", 0.0) for r in rows], dtype=float)

    # Drawn bottom-up, so reverse to keep AGENT_ROWS reading top-down.
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(
        y,
        means,
        xerr=stds,
        height=0.62,
        color=[AGENT_COLORS[r] for r in roles],
        capsize=3,
        error_kw={"linewidth": 1.0, "ecolor": "#444444"},
        zorder=3,
    )

    baseline = results.get("esm2+zero_shot", {}).get("spearman_mean")
    if baseline is not None:
        ax.axvline(
            baseline,
            color="#333333",
            linestyle="--",
            linewidth=1.0,
            zorder=2,
        )
        ax.annotate(
            f"current setup  ρ = {baseline:.3f}",
            xy=(baseline, y.max() + 0.52),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
            va="center",
        )

    # Direct value labels on every bar: the controls are intentionally grey,
    # so the numbers carry identity rather than the fill.
    for yi, m, s in zip(y, means, stds):
        ax.annotate(
            f"{m:.3f}",
            xy=(m + s, yi),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8.5,
            va="center",
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Held-out Spearman ρ  (higher is better)")
    ax.set_xlim(0, max(means + stds) * 1.26)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    seen: list[str] = []
    handles = []
    for role in roles:
        if role in seen:
            continue
        seen.append(role)
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=8,
                color=AGENT_COLORS[role],
                label=AGENT_ROLE_LABELS[role],
            )
        )
    # Below the axes: inside the plot it collided with the shortest bar's
    # value label.
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=len(handles),
        fontsize=8.5,
    )

    assay = payload.get("assay", "")
    n_train = payload.get("n_train", "")
    fig.suptitle(
        f"{assay} — does an LLM structural prior improve the surrogate?",
        fontsize=10.5,
    )
    ax.set_title(
        f"fit on {n_train} label-free variants, evaluated on the disjoint "
        "remainder; error bars ±1 std over 3 seeds",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    path = out_dir / f"{assay}{('_' + tag) if tag else ''}_agent_prior.png"
    fig.savefig(path)
    plt.close(fig)
    return path


#: Baseline vs agent-prior comparison colours (CVD-validated pair).
COMPARISON_COLORS = {"baseline": "#0072B2", "prior": "#D55E00"}


def plot_agent_prior_comparison(
    baseline: dict,
    agent: dict,
    out_dir: Path | None = None,
    tag: str = "agentprior",
) -> Path:
    """Two panels: what the prior changed, and what it did not.

    The left panel is the surrogate's final held-out Spearman; the right is the
    fraction of the pool's top 1% actually measured. Separate panels rather
    than one chart with two scales, because they are different quantities —
    and because the point of the figure is that they moved independently.
    """
    plt = _style()
    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in baseline["methods"] if m in agent["summary"]]
    labels = [METHOD_LABELS.get(m, m).replace(" ", "\n") for m in methods]
    x = np.arange(len(methods))
    width = 0.38

    def _rho(payload: dict, method: str) -> float:
        return float(payload["surrogate_curves"][method]["mean"][-1])

    def _recall(payload: dict, method: str) -> float:
        return float(payload["summary"][method]["top1pct_recall_mean"]) * 100

    panels = (
        ("Surrogate quality", "Final held-out Spearman ρ", _rho, None),
        (
            "Optimization outcome",
            "Top-1% variants found (%)",
            _recall,
            _recall(baseline, "random"),
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    for ax, (title, ylabel, fn, ref) in zip(axes, panels):
        base_vals = [fn(baseline, m) for m in methods]
        prior_vals = [fn(agent, m) for m in methods]
        ax.bar(
            x - width / 2,
            base_vals,
            width,
            label="ESM-2 + zero-shot",
            color=COMPARISON_COLORS["baseline"],
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            prior_vals,
            width,
            label="+ blind LLM prior",
            color=COMPARISON_COLORS["prior"],
            zorder=3,
        )
        if ref is not None:
            # Labelled via the legend rather than an inline annotation: at
            # this scale every inline position collided with a bar.
            ax.axhline(
                ref,
                color="#333333",
                linestyle="--",
                linewidth=1.0,
                zorder=2,
                label=f"random selection ({ref:.1f}%)",
            )
        for xi, (b, p) in enumerate(zip(base_vals, prior_vals)):
            for offset, value in ((-width / 2, b), (width / 2, p)):
                ax.annotate(
                    f"{value:.2f}" if ref is None else f"{value:.1f}",
                    xy=(xi + offset, value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#222222",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(base_vals + prior_vals) * 1.22)

    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)
    assay = baseline.get("assay_id", "")
    fig.suptitle(
        f"{assay} — the prior improves the model, not the outcome",
        fontsize=10.5,
    )
    fig.tight_layout()
    path = out_dir / f"{assay}_{tag}_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path
