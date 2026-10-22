"""Acquisition functions scored over a *finite* pool of measured candidates.

Because ProteinGym can only reveal fitness for variants that were actually
assayed, acquisition is a ranking problem over the measured candidate pool —
not continuous optimisation in embedding space. Every strategy below therefore
maps ``(mean, std)`` predictions plus loop state to a score per candidate, and
selection is a diversity-filtered top-k over those scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class AcquisitionContext:
    """State an acquisition function may legitimately depend on."""

    #: Best fitness revealed so far (the incumbent). Known to the agent
    #: because it has already paid to measure it.
    best_observed: float
    #: UCB exploration weight.
    beta: float = 2.0
    #: EI improvement margin.
    xi: float = 0.01


def ucb(mean: np.ndarray, std: np.ndarray, ctx: AcquisitionContext) -> np.ndarray:
    """Upper confidence bound: ``mu + beta * sigma``."""
    return mean + ctx.beta * std


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, ctx: AcquisitionContext
) -> np.ndarray:
    """Analytic expected improvement over the current incumbent."""
    std = np.maximum(std, 1e-9)
    z = (mean - ctx.best_observed - ctx.xi) / std
    return std * (z * norm.cdf(z) + norm.pdf(z))


def greedy(mean: np.ndarray, std: np.ndarray, ctx: AcquisitionContext) -> np.ndarray:
    """Pure exploitation: rank by posterior mean, ignoring uncertainty."""
    return mean


ACQUISITIONS = {
    "ucb": ucb,
    "ei": expected_improvement,
    "greedy": greedy,
}


def select_batch(
    scores: np.ndarray,
    available: np.ndarray,
    batch_size: int,
    features: np.ndarray | None = None,
    min_distance: float = 0.0,
) -> list[int]:
    """Pick ``batch_size`` distinct candidates by score, with a diversity filter.

    Parameters
    ----------
    scores:
        Acquisition score for every candidate in the pool.
    available:
        Boolean mask of candidates that have not yet been measured. Already
        measured variants are never re-selected, which is what keeps budget
        accounting honest.
    features:
        Pool feature matrix, used only to enforce in-batch diversity.
    min_distance:
        Minimum Euclidean feature distance between two picks in the same
        batch. A candidate closer than this to an already-chosen pick is
        skipped in favour of the next best. ``0`` disables the filter.

    Falls back to plain top-k if the diversity constraint cannot be satisfied,
    so a batch is always filled when enough candidates remain.
    """
    if batch_size <= 0:
        return []
    pool_indices = np.flatnonzero(available)
    if pool_indices.size == 0:
        return []
    batch_size = min(batch_size, int(pool_indices.size))

    order = pool_indices[np.argsort(-scores[pool_indices], kind="stable")]

    if features is None or min_distance <= 0.0:
        return order[:batch_size].tolist()

    chosen: list[int] = []
    for idx in order.tolist():
        if len(chosen) >= batch_size:
            break
        if chosen:
            d = np.linalg.norm(features[chosen] - features[idx], axis=1)
            if float(d.min()) < min_distance:
                continue
        chosen.append(idx)

    # Top up with the best remaining candidates if diversity starved the batch.
    if len(chosen) < batch_size:
        for idx in order.tolist():
            if len(chosen) >= batch_size:
                break
            if idx not in chosen:
                chosen.append(idx)
    return chosen


def pairwise_distance_quantile(
    features: np.ndarray, q: float, max_sample: int = 500, seed: int = 0
) -> float:
    """Estimate the ``q``-quantile of pairwise feature distances in the pool.

    Used to turn the configured ``diversity_quantile`` into an absolute
    distance threshold that adapts to the geometry of each assay's features.
    """
    if q <= 0.0:
        return 0.0
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    if n > max_sample:
        features = features[rng.choice(n, max_sample, replace=False)]
    from scipy.spatial.distance import pdist

    d = pdist(features)
    if d.size == 0:
        return 0.0
    return float(np.quantile(d, q))
