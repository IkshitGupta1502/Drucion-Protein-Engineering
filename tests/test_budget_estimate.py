import itertools
import numpy as np
import pytest
from drucion.evaluation.budget import estimate_random_budget


def test_expected_budget_matches_exhaustive_random_orders():
    # Same initial failure; two successes in five remaining candidates.
    scores = [0, 0, 1, 0, 1, 0]
    for batch in (1, 2, 3):
        costs = []
        for order in itertools.permutations(range(1, 6)):
            first = next(i+1 for i, idx in enumerate(order) if scores[idx] >= 1)
            costs.append(1 + min(5, ((first+batch-1)//batch)*batch))
        result = estimate_random_budget(scores, [0], [], 1, 3, batch)
        assert result['random_expected_measurements'] == pytest.approx(np.mean(costs))


def test_starting_sample_already_matches_has_no_savings():
    result = estimate_random_budget([1, 0, 2], [0], [], 1, 1, 2)
    assert result['saved_measurements'] == 0


def test_holdout_is_excluded_and_losses_are_not_hidden():
    result = estimate_random_budget([0, 1, 1, 0], [0], [1], 1, 3, 1)
    assert result['random_expected_measurements'] == 2.5
    assert result['saved_measurements'] == -.5


def test_unreachable_target_returns_none():
    assert estimate_random_budget([0, 1], [0], [1], 1, 2, 1) is None
