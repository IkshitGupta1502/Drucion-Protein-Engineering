"""Post-hoc random-budget comparison; never used to choose candidates."""
import numpy as np


def estimate_random_budget(scores, initial_indices, excluded_indices, target, actual_measurements, batch_size):
    """Expected cost to first reach target, with the same start and batch size.

    For N remaining candidates and K successes, probability of no success
    after t random draws is C(N-K,t)/C(N,t). Sum survival at batch boundaries
    to obtain expected measurements, including a possibly partial last batch.
    This uses known retrospective labels, not surrogate predictions.
    """
    scores = np.asarray(scores, dtype=float)
    initial = np.asarray(initial_indices, dtype=int)
    n_initial = len(initial)
    if not n_initial or batch_size < 1 or not np.isfinite(target):
        return None
    expected = float(n_initial)
    if np.max(scores[initial]) < target:
        mask = np.ones(len(scores), dtype=bool)
        mask[initial] = False
        mask[np.asarray(excluded_indices, dtype=int)] = False
        remaining = scores[mask]
        n = len(remaining)
        k = int(np.count_nonzero(remaining >= target))
        if not k:
            return None
        survival = 1.0
        for t in range(n):
            if t % batch_size == 0:
                expected += min(batch_size, n-t) * survival
            survival *= max(0, n-k-t) / (n-t)
            if survival == 0:
                break
    saved = expected - actual_measurements
    return {
        'target_score': float(target),
        'actual_measurements': int(actual_measurements),
        'random_expected_measurements': expected,
        'saved_measurements': saved,
        'saved_percent': 100 * saved / expected,
    }
