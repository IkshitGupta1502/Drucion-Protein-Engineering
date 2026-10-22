"""Acquisition scoring and batch-selection invariants."""

from __future__ import annotations

import numpy as np
import pytest

from drucion.optimization.acquisition import (
    ACQUISITIONS,
    AcquisitionContext,
    expected_improvement,
    greedy,
    pairwise_distance_quantile,
    select_batch,
    ucb,
)


@pytest.fixture
def ctx():
    return AcquisitionContext(best_observed=1.0, beta=2.0)


class TestAcquisitionFunctions:
    def test_ucb_rewards_uncertainty(self, ctx):
        mean = np.array([1.0, 1.0])
        std = np.array([0.1, 0.9])
        scores = ucb(mean, std, ctx)
        assert scores[1] > scores[0]

    def test_ucb_matches_formula(self, ctx):
        mean = np.array([0.5, -1.0])
        std = np.array([0.2, 1.5])
        assert ucb(mean, std, ctx) == pytest.approx(mean + 2.0 * std)

    def test_greedy_ignores_uncertainty(self, ctx):
        mean = np.array([1.0, 2.0])
        a = greedy(mean, np.array([0.0, 0.0]), ctx)
        b = greedy(mean, np.array([5.0, 5.0]), ctx)
        assert a == pytest.approx(b)
        assert a == pytest.approx(mean)

    def test_ei_is_non_negative(self, ctx):
        mean = np.linspace(-5, 5, 50)
        std = np.full(50, 0.5)
        assert np.all(expected_improvement(mean, std, ctx) >= 0)

    def test_ei_prefers_promising_candidates(self, ctx):
        std = np.array([0.5, 0.5])
        mean = np.array([0.0, 3.0])  # incumbent is 1.0
        scores = expected_improvement(mean, std, ctx)
        assert scores[1] > scores[0]

    def test_ei_rewards_uncertainty_at_the_incumbent(self, ctx):
        mean = np.array([1.0, 1.0])
        std = np.array([0.05, 1.0])
        scores = expected_improvement(mean, std, ctx)
        assert scores[1] > scores[0]

    def test_ei_handles_zero_uncertainty(self, ctx):
        scores = expected_improvement(
            np.array([5.0, 0.0]), np.array([0.0, 0.0]), ctx
        )
        assert np.all(np.isfinite(scores))

    @pytest.mark.parametrize("name", sorted(ACQUISITIONS))
    def test_all_return_finite_scores_of_right_shape(self, name, ctx):
        rng = np.random.default_rng(0)
        mean = rng.standard_normal(30)
        std = np.abs(rng.standard_normal(30)) + 1e-3
        scores = ACQUISITIONS[name](mean, std, ctx)
        assert scores.shape == (30,)
        assert np.all(np.isfinite(scores))


class TestSelectBatch:
    def test_picks_highest_scores(self):
        scores = np.array([0.1, 0.9, 0.5, 0.7])
        available = np.ones(4, dtype=bool)
        assert select_batch(scores, available, 2) == [1, 3]

    def test_never_selects_unavailable_candidates(self):
        scores = np.array([9.0, 0.1, 0.2, 0.3])
        available = np.array([False, True, True, True])
        chosen = select_batch(scores, available, 3)
        assert 0 not in chosen
        assert len(chosen) == 3

    def test_returns_no_duplicates(self):
        rng = np.random.default_rng(0)
        scores = rng.random(100)
        available = np.ones(100, dtype=bool)
        chosen = select_batch(scores, available, 25)
        assert len(set(chosen)) == 25

    def test_caps_at_available_count(self):
        scores = np.arange(5.0)
        available = np.array([True, True, False, False, False])
        assert len(select_batch(scores, available, 10)) == 2

    def test_empty_pool_returns_empty(self):
        assert select_batch(np.arange(3.0), np.zeros(3, dtype=bool), 2) == []

    def test_zero_batch_size_returns_empty(self):
        assert select_batch(np.arange(3.0), np.ones(3, dtype=bool), 0) == []

    def test_handles_tied_scores_deterministically(self):
        scores = np.ones(10)
        available = np.ones(10, dtype=bool)
        a = select_batch(scores, available, 4)
        b = select_batch(scores, available, 4)
        assert a == b

    def test_diversity_filter_spreads_picks(self):
        """Near-duplicate top scorers should not monopolise a batch."""
        # Three tight clusters; scores decrease across clusters.
        features = np.array(
            [[0.0, 0.0], [0.01, 0.0], [0.02, 0.0],  # cluster A (best scores)
             [5.0, 0.0], [5.01, 0.0],               # cluster B
             [10.0, 0.0]]                            # cluster C
        )
        scores = np.array([1.0, 0.99, 0.98, 0.5, 0.49, 0.2])
        available = np.ones(6, dtype=bool)

        plain = select_batch(scores, available, 3, min_distance=0.0)
        diverse = select_batch(
            scores, available, 3, features=features, min_distance=1.0
        )
        assert plain == [0, 1, 2]  # all from cluster A
        assert diverse == [0, 3, 5]  # one per cluster

    def test_diversity_filter_still_fills_the_batch(self):
        """If diversity starves the batch, top up rather than under-deliver."""
        features = np.zeros((5, 2))  # every candidate identical
        scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        available = np.ones(5, dtype=bool)
        chosen = select_batch(
            scores, available, 3, features=features, min_distance=5.0
        )
        assert len(chosen) == 3
        assert len(set(chosen)) == 3


class TestDistanceQuantile:
    def test_zero_quantile_disables_filter(self):
        assert pairwise_distance_quantile(np.random.rand(20, 3), 0.0) == 0.0

    def test_returns_positive_distance(self):
        rng = np.random.default_rng(0)
        d = pairwise_distance_quantile(rng.standard_normal((50, 4)), 0.1)
        assert d > 0

    def test_is_deterministic(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((800, 4))
        a = pairwise_distance_quantile(X, 0.05, seed=1)
        b = pairwise_distance_quantile(X, 0.05, seed=1)
        assert a == b
