"""Configuration round-tripping and benchmark metric correctness."""

from __future__ import annotations

import numpy as np
import pytest

from drucion.config import REPO_ROOT, ExperimentConfig
from drucion.evaluation.metrics import (
    RunMetrics,
    aggregate,
    budget_to_match,
    compute_run_metrics,
    curve_band,
    evals_to_threshold,
    normalised_auc,
    paired_comparison,
)


class TestConfig:
    def test_total_budget(self):
        cfg = ExperimentConfig()
        cfg.loop.n_initial = 50
        cfg.loop.batch_size = 10
        cfg.loop.n_rounds = 15
        assert cfg.total_budget == 200

    def test_dict_round_trip(self):
        cfg = ExperimentConfig()
        cfg.loop.acquisition = "ei"
        cfg.embedding.n_components = 32
        cfg.surrogate.kind = "rf"
        restored = ExperimentConfig.from_dict(cfg.to_dict())
        assert restored.to_dict() == cfg.to_dict()
        assert restored.loop.acquisition == "ei"
        assert restored.embedding.n_components == 32
        assert restored.surrogate.kind == "rf"

    def test_default_yaml_loads_and_matches_defaults(self):
        path = REPO_ROOT / "configs" / "default.yaml"
        assert path.exists(), "configs/default.yaml is documented and must exist"
        cfg = ExperimentConfig.from_yaml(path)
        default = ExperimentConfig()
        assert cfg.assay_id == default.assay_id
        assert cfg.loop.n_initial == default.loop.n_initial
        assert cfg.loop.batch_size == default.loop.batch_size
        assert cfg.loop.n_rounds == default.loop.n_rounds
        assert cfg.loop.acquisition == default.loop.acquisition
        assert cfg.embedding.featurizer == default.embedding.featurizer
        assert cfg.embedding.n_components == default.embedding.n_components
        assert cfg.surrogate.kind == default.surrogate.kind


class TestNormalisedAuc:
    def test_flat_curve_at_pool_mean_scores_zero(self):
        n = np.array([10, 20, 30])
        best = np.array([1.0, 1.0, 1.0])
        assert normalised_auc(n, best, pool_mean=1.0, pool_max=5.0) == pytest.approx(0.0)

    def test_flat_curve_at_pool_max_scores_one(self):
        n = np.array([10, 20, 30])
        best = np.array([5.0, 5.0, 5.0])
        assert normalised_auc(n, best, pool_mean=1.0, pool_max=5.0) == pytest.approx(1.0)

    def test_earlier_discovery_scores_higher(self):
        n = np.array([0, 10, 20, 30])
        early = np.array([0.0, 4.0, 4.0, 4.0])
        late = np.array([0.0, 0.0, 0.0, 4.0])
        a = normalised_auc(n, early, 0.0, 4.0)
        b = normalised_auc(n, late, 0.0, 4.0)
        assert a > b

    def test_degenerate_pool_returns_nan(self):
        n = np.array([1, 2])
        assert np.isnan(normalised_auc(n, np.array([1.0, 1.0]), 2.0, 2.0))

    def test_single_point_returns_nan(self):
        assert np.isnan(normalised_auc(np.array([1]), np.array([1.0]), 0.0, 2.0))


class TestEvalsToThreshold:
    def test_returns_first_crossing(self):
        n = np.array([10, 20, 30, 40])
        best = np.array([0.1, 0.4, 0.9, 1.2])
        assert evals_to_threshold(n, best, 0.8) == 30

    def test_returns_none_when_never_reached(self):
        n = np.array([10, 20])
        assert evals_to_threshold(n, np.array([0.1, 0.2]), 5.0) is None

    def test_exact_equality_counts_as_reached(self):
        n = np.array([10, 20])
        assert evals_to_threshold(n, np.array([0.1, 0.5]), 0.5) == 20


def _run(method, seed, best=1.0, auc=0.5, evals=50, measured=(0, 1)):
    return RunMetrics(
        method=method,
        seed=seed,
        n_measured=100,
        best_measured=best,
        best_mutant="A1B",
        auc_normalised=auc,
        final_regret=2.0 - best,
        top1pct_recall=0.1,
        top10pct_recall=0.2,
        evals_to_threshold=evals,
        threshold=1.0,
        mean_surrogate_spearman=0.4,
    )


class TestComputeRunMetrics:
    def test_assembles_expected_fields(self):
        pool_stats = {
            "pool_max": 3.0,
            "pool_min": -1.0,
            "pool_mean": 0.0,
            "pool_std": 1.0,
            "p90": 1.5,
            "p99": 2.5,
        }
        m = compute_run_metrics(
            method="ucb",
            seed=0,
            n_measured=np.array([50, 60, 70]),
            best_so_far=np.array([1.0, 2.0, 2.6]),
            measured_indices=[1, 2, 3],
            pool_stats=pool_stats,
            top1pct={1, 9},
            top10pct={1, 2, 8, 9},
            best_mutant="K2R",
        )
        assert m.best_measured == 2.6
        assert m.final_regret == pytest.approx(0.4)
        assert m.top1pct_recall == pytest.approx(0.5)
        assert m.top10pct_recall == pytest.approx(0.5)
        assert m.evals_to_threshold == 70
        assert m.threshold == 2.5

    def test_handles_never_reaching_threshold(self):
        pool_stats = {
            "pool_max": 9.0, "pool_min": 0.0, "pool_mean": 1.0,
            "pool_std": 1.0, "p90": 7.0, "p99": 8.0,
        }
        m = compute_run_metrics(
            "random", 0, np.array([10, 20]), np.array([1.0, 1.1]),
            [0], pool_stats, set(), set(), "A1B",
        )
        assert m.evals_to_threshold is None
        assert m.top1pct_recall == 0.0


class TestAggregate:
    def test_mean_and_std_per_method(self):
        runs = [_run("ucb", 0, best=1.0), _run("ucb", 1, best=3.0)]
        agg = aggregate(runs)["ucb"]
        assert agg["n_seeds"] == 2
        assert agg["best_measured_mean"] == pytest.approx(2.0)
        assert agg["best_measured_std"] == pytest.approx(1.0)

    def test_threshold_hit_rate_excludes_censored_runs(self):
        runs = [
            _run("ucb", 0, evals=40),
            _run("ucb", 1, evals=60),
            _run("ucb", 2, evals=None),
        ]
        agg = aggregate(runs)["ucb"]
        assert agg["threshold_hit_rate"] == pytest.approx(2 / 3)
        # Censored run must not be averaged in as if it were a number.
        assert agg["evals_to_threshold_mean"] == pytest.approx(50.0)

    def test_all_censored_reports_none(self):
        runs = [_run("ucb", 0, evals=None), _run("ucb", 1, evals=None)]
        agg = aggregate(runs)["ucb"]
        assert agg["threshold_hit_rate"] == 0.0
        assert agg["evals_to_threshold_mean"] is None


class TestPairedComparison:
    def _pair(self, better: bool):
        runs = []
        for seed in range(6):
            runs.append(_run("random", seed, best=1.0 + 0.1 * seed, auc=0.5))
            delta = 0.5 if better else -0.5
            runs.append(_run("ucb", seed, best=1.0 + 0.1 * seed + delta, auc=0.6))
        return runs

    def test_detects_consistent_improvement(self):
        out = paired_comparison(self._pair(better=True))
        m = out["methods"]["ucb"]["metrics"]["best_measured"]
        assert m["wins"] == 6
        assert m["mean_difference"] == pytest.approx(0.5)
        assert m["wilcoxon_p"] is not None and m["wilcoxon_p"] < 0.05

    def test_reports_losses_honestly(self):
        out = paired_comparison(self._pair(better=False))
        m = out["methods"]["ucb"]["metrics"]["best_measured"]
        assert m["wins"] == 0
        assert m["mean_difference"] < 0

    def test_lower_is_better_metric_flips_win_direction(self):
        runs = []
        for seed in range(6):
            runs.append(_run("random", seed, evals=100))
            runs.append(_run("ucb", seed, evals=60))
        m = paired_comparison(runs)["methods"]["ucb"]["metrics"]["evals_to_threshold"]
        assert m["lower_is_better"] is True
        assert m["wins"] == 6
        assert m["mean_difference"] == pytest.approx(-40.0)

    def test_missing_baseline_returns_empty(self):
        assert paired_comparison([_run("ucb", 0)], baseline="random") == {}

    def test_too_few_seeds_reports_no_pvalue(self):
        runs = [_run("random", 0), _run("ucb", 0, best=2.0)]
        m = paired_comparison(runs)["methods"]["ucb"]["metrics"]["best_measured"]
        assert m["wilcoxon_p"] is None
        assert m["n"] == 1


class TestBudgetToMatch:
    def test_finds_earlier_budget_for_better_method(self):
        curves = {
            "random": {"n_measured": [10, 20, 30], "mean": [0.0, 0.5, 1.0], "std": [0, 0, 0]},
            "ucb": {"n_measured": [10, 20, 30], "mean": [0.0, 1.0, 1.5], "std": [0, 0, 0]},
        }
        out = budget_to_match(curves)
        assert out["target"] == 1.0
        assert out["methods"]["ucb"]["budget_to_match"] == 20
        assert out["methods"]["ucb"]["savings_fraction"] == pytest.approx(1 / 3)

    def test_reports_none_when_never_matched(self):
        curves = {
            "random": {"n_measured": [10, 20], "mean": [0.0, 5.0], "std": [0, 0]},
            "weak": {"n_measured": [10, 20], "mean": [0.0, 1.0], "std": [0, 0]},
        }
        out = budget_to_match(curves)
        assert out["methods"]["weak"]["budget_to_match"] is None
        assert out["methods"]["weak"]["savings_fraction"] is None


class TestCurveBand:
    def test_mean_and_std_across_runs(self):
        x = np.array([10, 20])
        curves = [(x, np.array([1.0, 2.0])), (x, np.array([3.0, 4.0]))]
        gx, mean, std = curve_band(curves)
        assert gx.tolist() == [10, 20]
        assert mean.tolist() == [2.0, 3.0]
        assert std.tolist() == [1.0, 1.0]

    def test_refuses_mismatched_budget_grids(self):
        a = (np.array([10, 20]), np.array([1.0, 2.0]))
        b = (np.array([10, 30]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="mismatched budget grids"):
            curve_band([a, b])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            curve_band([])
