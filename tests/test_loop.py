"""Closed-loop state transitions, reproducibility and integration."""

from __future__ import annotations


import numpy as np
import pytest

from drucion.config import ExperimentConfig
from drucion.data.proteingym import load_assay
from drucion.optimization.loop import ClosedLoopOptimizer


@pytest.fixture
def make_opt(synthetic_assay, synthetic_features):
    """Factory for optimizers over a fresh oracle each time."""

    def _make(acquisition="ucb", rounds=3, n_initial=20, batch_size=5, seed=0, **kw):
        assay = load_assay(
            synthetic_assay["assay_id"],
            assay_dir=synthetic_assay["assay_dir"],
            reference_csv=synthetic_assay["reference_csv"],
        )
        cfg = ExperimentConfig(assay_id=synthetic_assay["assay_id"])
        cfg.loop.acquisition = acquisition
        cfg.loop.n_rounds = rounds
        cfg.loop.n_initial = n_initial
        cfg.loop.batch_size = batch_size
        cfg.loop.seed = seed
        for k, v in kw.items():
            setattr(cfg.loop, k, v)
        return ClosedLoopOptimizer(assay, cfg, features=synthetic_features), cfg

    return _make


class TestStateTransitions:
    def test_starts_with_initial_design_recorded_as_round_zero(self, make_opt):
        opt, _ = make_opt()
        assert opt.round == 0
        assert len(opt.history) == 1
        assert opt.history[0].round == 0
        assert opt.history[0].selected == []
        assert opt.oracle.n_queries == 20
        assert not opt.finished

    def test_each_step_advances_exactly_one_round(self, make_opt):
        opt, _ = make_opt(rounds=3)
        for expected in (1, 2, 3):
            record = opt.step()
            assert record.round == expected
            assert opt.round == expected
            assert len(opt.history) == expected + 1

    def test_each_step_measures_exactly_batch_size_variants(self, make_opt):
        opt, _ = make_opt(batch_size=5)
        before = opt.oracle.n_queries
        record = opt.step()
        assert len(record.selected) == 5
        assert opt.oracle.n_queries == before + 5
        assert record.n_measured == opt.oracle.n_queries

    def test_finishes_after_configured_rounds(self, make_opt):
        opt, cfg = make_opt(rounds=3)
        opt.run()
        assert opt.finished
        assert opt.round == 3
        assert opt.oracle.n_queries == cfg.total_budget

    def test_stepping_after_finish_raises(self, make_opt):
        opt, _ = make_opt(rounds=2)
        opt.run()
        with pytest.raises(RuntimeError, match="finished"):
            opt.step()

    def test_run_is_idempotent_once_finished(self, make_opt):
        opt, _ = make_opt(rounds=2)
        opt.run()
        n = opt.oracle.n_queries
        opt.run()
        assert opt.oracle.n_queries == n

    def test_step_and_run_produce_identical_results(self, make_opt):
        a, _ = make_opt(rounds=3, seed=5)
        b, _ = make_opt(rounds=3, seed=5)
        a.run()
        for _ in range(3):
            b.step()
        assert a.observed == b.observed
        assert a.best_measured == pytest.approx(b.best_measured)


class TestLearningCurve:
    def test_is_monotonically_non_decreasing(self, make_opt):
        opt, _ = make_opt(rounds=5)
        opt.run()
        _, best = opt.learning_curve()
        assert np.all(np.diff(best) >= 0)

    def test_x_axis_tracks_cumulative_measurements(self, make_opt):
        opt, _ = make_opt(rounds=4, n_initial=20, batch_size=5)
        opt.run()
        n, _ = opt.learning_curve()
        assert n.tolist() == [20, 25, 30, 35, 40]

    def test_best_measured_is_an_actually_measured_value(self, make_opt):
        opt, _ = make_opt(rounds=3)
        opt.run()
        assert opt.best_measured == pytest.approx(max(opt.observed_y))
        assert opt.best_index in opt.observed
        assert opt.pool.mutants[opt.best_index] == opt.best_mutant


class TestRecordContents:
    def test_selected_variants_carry_prediction_and_measurement(self, make_opt):
        opt, _ = make_opt(acquisition="ucb", rounds=1)
        record = opt.step()
        for v in record.selected:
            assert v.predicted_fitness is not None
            assert v.predicted_uncertainty is not None
            assert v.predicted_uncertainty >= 0
            assert np.isfinite(v.measured_fitness)
            assert v.mutant == opt.pool.mutants[v.index]
            assert v.positions

    def test_random_method_records_no_surrogate_dependence(self, make_opt):
        """Random still fits a surrogate for reporting, but must not rank with it."""
        opt, _ = make_opt(acquisition="random", rounds=2)
        opt.run()
        # Selection order should not be sorted by predicted fitness.
        preds = [
            v.predicted_fitness for r in opt.history[1:] for v in r.selected
        ]
        assert len(preds) == 10
        assert preds != sorted(preds, reverse=True)

    def test_round_record_serialises(self, make_opt):
        opt, _ = make_opt(rounds=1)
        d = opt.step().to_dict()
        assert isinstance(d["selected"], list)
        assert isinstance(d["selected"][0], dict)
        assert "measured_fitness" in d["selected"][0]

    def test_state_dict_is_json_serialisable(self, make_opt):
        import json

        opt, _ = make_opt(rounds=2)
        opt.run()
        json.dumps(opt.state_dict())


class TestReproducibility:
    @pytest.mark.parametrize(
        "acquisition", ["random", "greedy", "ucb", "ei"]
    )
    def test_same_seed_gives_identical_run(self, make_opt, acquisition):
        a, _ = make_opt(acquisition=acquisition, seed=11, rounds=3)
        b, _ = make_opt(acquisition=acquisition, seed=11, rounds=3)
        a.run()
        b.run()
        assert a.observed == b.observed
        assert a.observed_y == pytest.approx(b.observed_y)
        assert a.best_mutant == b.best_mutant

    def test_different_seeds_diverge(self, make_opt):
        a, _ = make_opt(acquisition="random", seed=1, rounds=3)
        b, _ = make_opt(acquisition="random", seed=2, rounds=3)
        a.run()
        b.run()
        assert a.observed != b.observed

    def test_initial_design_is_seed_determined_and_shared(self):
        x = ClosedLoopOptimizer.draw_initial_indices(500, 50, seed=4)
        y = ClosedLoopOptimizer.draw_initial_indices(500, 50, seed=4)
        z = ClosedLoopOptimizer.draw_initial_indices(500, 50, seed=5)
        assert x == y
        assert x != z
        assert len(set(x)) == 50


class TestFairness:
    def test_methods_share_identical_initial_observations(
        self, synthetic_assay, synthetic_features
    ):
        """The contract that makes the benchmark a controlled comparison."""
        initial = ClosedLoopOptimizer.draw_initial_indices(
            240, 20, seed=3
        )
        firsts = {}
        for acq in ("random", "greedy", "ucb", "ei"):
            assay = load_assay(
                synthetic_assay["assay_id"],
                assay_dir=synthetic_assay["assay_dir"],
                reference_csv=synthetic_assay["reference_csv"],
            )
            cfg = ExperimentConfig(assay_id=synthetic_assay["assay_id"])
            cfg.loop.acquisition = acq
            cfg.loop.n_initial = 20
            cfg.loop.batch_size = 5
            cfg.loop.n_rounds = 2
            opt = ClosedLoopOptimizer(
                assay, cfg, features=synthetic_features, initial_indices=initial
            )
            firsts[acq] = (
                opt.observed[:20],
                opt.history[0].best_measured,
            )
        baseline = firsts["random"]
        for acq, got in firsts.items():
            assert got[0] == baseline[0] == initial, acq
            assert got[1] == pytest.approx(baseline[1]), acq

    def test_all_methods_consume_the_same_budget(
        self, synthetic_assay, synthetic_features
    ):
        budgets = {}
        for acq in ("random", "greedy", "ucb", "ei"):
            assay = load_assay(
                synthetic_assay["assay_id"],
                assay_dir=synthetic_assay["assay_dir"],
                reference_csv=synthetic_assay["reference_csv"],
            )
            cfg = ExperimentConfig(assay_id=synthetic_assay["assay_id"])
            cfg.loop.acquisition = acq
            cfg.loop.n_initial = 20
            cfg.loop.batch_size = 5
            cfg.loop.n_rounds = 3
            opt = ClosedLoopOptimizer(assay, cfg, features=synthetic_features)
            opt.run()
            budgets[acq] = opt.oracle.n_queries
        assert len(set(budgets.values())) == 1
        assert set(budgets.values()) == {35}


class TestIntegration:
    """Several complete optimization rounds, end to end."""

    def test_ten_round_run_on_synthetic_assay(
        self, synthetic_assay, synthetic_features
    ):
        assay = load_assay(
            synthetic_assay["assay_id"],
            assay_dir=synthetic_assay["assay_dir"],
            reference_csv=synthetic_assay["reference_csv"],
        )
        cfg = ExperimentConfig(assay_id=synthetic_assay["assay_id"])
        cfg.loop.n_initial = 30
        cfg.loop.batch_size = 8
        cfg.loop.n_rounds = 10
        cfg.loop.acquisition = "ucb"
        holdout = list(range(200, 240))
        opt = ClosedLoopOptimizer(
            assay,
            cfg,
            features=synthetic_features,
            initial_indices=list(range(30)),
            holdout_indices=holdout,
        )
        history = opt.run()

        assert len(history) == 11
        assert opt.finished
        assert opt.oracle.n_queries == 30 + 80
        assert len(set(opt.observed)) == len(opt.observed)
        assert not (set(opt.observed) & set(holdout))
        _, best = opt.learning_curve()
        assert np.all(np.diff(best) >= 0)
        # Every round must have produced a usable surrogate.
        assert all(r.surrogate_kind in ("gp", "rf") for r in history[1:])
        assert all(
            r.holdout_spearman is None or np.isfinite(r.holdout_spearman)
            for r in history[1:]
        )

    def test_active_learning_enriches_top_variants_on_learnable_landscape(
        self, synthetic_assay, synthetic_learnable_features
    ):
        """Sanity check on synthetic data with a genuinely learnable signal.

        Measures *enrichment* (how many top-decile variants the budget found)
        rather than the single best value: on a small pool the max saturates
        and carries almost no signal. This is a smoke test of the machinery,
        not a scientific claim — the real comparison is
        ``scripts/run_benchmark.py`` on ProteinGym data.
        """
        results: dict[str, list[float]] = {"random": [], "ucb": []}
        for acq in results:
            for seed in range(4):
                assay = load_assay(
                    synthetic_assay["assay_id"],
                    assay_dir=synthetic_assay["assay_dir"],
                    reference_csv=synthetic_assay["reference_csv"],
                )
                cfg = ExperimentConfig(assay_id=synthetic_assay["assay_id"])
                cfg.loop.n_initial = 20
                cfg.loop.batch_size = 5
                cfg.loop.n_rounds = 6
                cfg.loop.acquisition = acq
                cfg.loop.seed = seed
                initial = ClosedLoopOptimizer.draw_initial_indices(
                    len(assay.pool), 20, seed
                )
                opt = ClosedLoopOptimizer(
                    assay, cfg, features=synthetic_learnable_features,
                    initial_indices=initial,
                )
                opt.run()
                top10 = assay.oracle.top_quantile_indices(0.90)
                recall = len(set(opt.observed) & top10) / len(top10)
                results[acq].append(recall)
        assert np.mean(results["ucb"]) > np.mean(results["random"]), results

    def test_exhausting_the_pool_raises_rather_than_overrunning(
        self, synthetic_assay, synthetic_features
    ):
        assay = load_assay(
            synthetic_assay["assay_id"],
            assay_dir=synthetic_assay["assay_dir"],
            reference_csv=synthetic_assay["reference_csv"],
        )
        cfg = ExperimentConfig(assay_id=synthetic_assay["assay_id"])
        pool_n = len(assay.pool)
        cfg.loop.n_initial = pool_n - 40
        cfg.loop.batch_size = 30
        cfg.loop.n_rounds = 5  # requested budget exceeds the pool
        cfg.loop.acquisition = "ucb"
        opt = ClosedLoopOptimizer(assay, cfg, features=synthetic_features)
        with pytest.raises(RuntimeError, match="exhausted"):
            opt.run()
        assert opt.oracle.n_queries <= len(assay.pool)


class TestRandomBaselineIsUniform:
    """The random baseline must be plain uniform sampling, not a smarter one.

    If the batch diversity filter were applied to random selection, we would be
    comparing against a better-than-random baseline and quietly understating
    our own result's context.
    """

    def _observed_for(self, synthetic_assay, features, diversity_quantile):
        assay = load_assay(
            synthetic_assay["assay_id"],
            assay_dir=synthetic_assay["assay_dir"],
            reference_csv=synthetic_assay["reference_csv"],
        )
        cfg = ExperimentConfig(assay_id=synthetic_assay["assay_id"])
        cfg.loop.acquisition = "random"
        cfg.loop.n_initial = 20
        cfg.loop.batch_size = 10
        cfg.loop.n_rounds = 4
        cfg.loop.seed = 3
        cfg.loop.diversity_quantile = diversity_quantile
        opt = ClosedLoopOptimizer(
            assay, cfg, features=features, initial_indices=list(range(20))
        )
        opt.run()
        return opt.observed

    def test_diversity_setting_does_not_change_random_selection(
        self, synthetic_assay, synthetic_features
    ):
        off = self._observed_for(synthetic_assay, synthetic_features, 0.0)
        on = self._observed_for(synthetic_assay, synthetic_features, 0.25)
        assert off == on

    def test_non_random_methods_do_respond_to_diversity(
        self, synthetic_assay, synthetic_learnable_features
    ):
        def run(dq):
            assay = load_assay(
                synthetic_assay["assay_id"],
                assay_dir=synthetic_assay["assay_dir"],
                reference_csv=synthetic_assay["reference_csv"],
            )
            cfg = ExperimentConfig(assay_id=synthetic_assay["assay_id"])
            cfg.loop.acquisition = "greedy"
            cfg.loop.n_initial = 20
            cfg.loop.batch_size = 10
            cfg.loop.n_rounds = 3
            cfg.loop.seed = 3
            cfg.loop.diversity_quantile = dq
            opt = ClosedLoopOptimizer(
                assay,
                cfg,
                features=synthetic_learnable_features,
                initial_indices=list(range(20)),
            )
            opt.run()
            return opt.observed

        assert run(0.0) != run(0.3)
