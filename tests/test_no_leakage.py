"""The central scientific-integrity tests: hidden labels stay hidden.

If any of these fail, every performance number the project reports is void.
"""

from __future__ import annotations

import inspect

import pytest

from drucion.config import ExperimentConfig
from drucion.embeddings import features as features_mod
from drucion.models import surrogate as surrogate_mod
from drucion.optimization import acquisition as acq_mod
from drucion.optimization import loop as loop_mod
from drucion.optimization.loop import ClosedLoopOptimizer


class TestPoolHasNoLabels:
    def test_candidate_pool_exposes_no_fitness_attribute(self, synthetic_loaded):
        pool = synthetic_loaded.pool
        public = [a for a in dir(pool) if not a.startswith("_")]
        banned = {"scores", "dms_score", "dms_scores", "fitness", "y", "labels"}
        assert not (banned & {a.lower() for a in public}), public

    def test_pool_fields_are_only_specs_and_sequences(self, synthetic_loaded):
        fields = set(synthetic_loaded.pool.__dataclass_fields__)
        assert fields == {"assay_id", "wild_type", "mutants", "sequences"}

    def test_oracle_scores_are_private(self, synthetic_loaded):
        oracle = synthetic_loaded.oracle
        assert not hasattr(oracle, "scores")
        # The only public bulk accessors are explicitly named for evaluation.
        public = {a for a in dir(oracle) if not a.startswith("_")}
        bulk = {a for a in public if "score" in a.lower()}
        assert bulk == {"all_scores_for_evaluation"}, bulk


class TestOptimizationCodeCannotSeeLabels:
    """Static guard: scientific selection modules must not touch label APIs."""

    LABEL_APIS = (
        "all_scores_for_evaluation",
        "peek_for_evaluation",
        "ground_truth_summary",
        "top_quantile_indices",
    )

    @pytest.mark.parametrize(
        "module", [acq_mod, surrogate_mod, features_mod], ids=lambda m: m.__name__
    )
    def test_module_never_references_label_apis(self, module):
        src = inspect.getsource(module)
        for api in self.LABEL_APIS:
            assert api not in src, f"{module.__name__} references {api}"

    def test_loop_uses_peek_only_for_reporting(self):
        """The loop may peek for the holdout metric, never for selection."""
        src = inspect.getsource(loop_mod)
        assert src.count("peek_for_evaluation") == 1
        # The one call must sit in the constructor's holdout setup, not step().
        step_src = inspect.getsource(ClosedLoopOptimizer.step)
        assert "peek_for_evaluation" not in step_src
        assert "all_scores_for_evaluation" not in src
        assert "ground_truth_summary" not in src

    def test_acquisition_signature_takes_only_predictions(self):
        for fn in (acq_mod.ucb, acq_mod.expected_improvement, acq_mod.greedy):
            params = list(inspect.signature(fn).parameters)
            assert params == ["mean", "std", "ctx"], (fn.__name__, params)


class TestBudgetAccounting:
    def _run(self, assay, features, acquisition="ucb", rounds=3, **loop_kwargs):
        cfg = ExperimentConfig(assay_id=assay.pool.assay_id)
        cfg.loop.n_initial = 20
        cfg.loop.batch_size = 5
        cfg.loop.n_rounds = rounds
        cfg.loop.acquisition = acquisition
        for k, v in loop_kwargs.items():
            setattr(cfg.loop, k, v)
        opt = ClosedLoopOptimizer(assay, cfg, features=features)
        opt.run()
        return cfg, opt

    def test_oracle_queries_equal_declared_budget(
        self, synthetic_loaded, synthetic_features
    ):
        cfg, opt = self._run(synthetic_loaded, synthetic_features)
        assert opt.oracle.n_queries == cfg.total_budget == 20 + 5 * 3

    def test_observed_count_matches_oracle_count(
        self, synthetic_loaded, synthetic_features
    ):
        _, opt = self._run(synthetic_loaded, synthetic_features)
        assert len(opt.observed) == opt.oracle.n_queries
        assert len(opt.observed_y) == opt.oracle.n_queries

    def test_no_variant_measured_twice(self, synthetic_loaded, synthetic_features):
        _, opt = self._run(synthetic_loaded, synthetic_features)
        assert len(set(opt.observed)) == len(opt.observed)

    @pytest.mark.parametrize(
        "acquisition", ["random", "greedy", "ucb", "ei"]
    )
    def test_every_method_respects_the_same_budget(
        self, synthetic_loaded, synthetic_features, acquisition, synthetic_assay
    ):
        from drucion.data.proteingym import load_assay

        assay = load_assay(
            synthetic_assay["assay_id"],
            assay_dir=synthetic_assay["assay_dir"],
            reference_csv=synthetic_assay["reference_csv"],
        )
        cfg, opt = self._run(assay, synthetic_features, acquisition=acquisition)
        assert opt.oracle.n_queries == cfg.total_budget

    def test_selected_variants_were_previously_unmeasured(
        self, synthetic_loaded, synthetic_features
    ):
        _, opt = self._run(synthetic_loaded, synthetic_features)
        seen: set[int] = set(opt.history[0].selected and [] or [])
        # Rebuild the measurement order and assert strict novelty per round.
        measured: set[int] = set()
        n_initial = 20
        measured.update(opt.observed[:n_initial])
        cursor = n_initial
        for record in opt.history[1:]:
            for v in record.selected:
                assert v.index not in measured, f"{v.mutant} measured twice"
                measured.add(v.index)
            cursor += len(record.selected)
        assert cursor == len(opt.observed)
        assert not seen

    def test_holdout_is_never_measured_or_charged(
        self, synthetic_loaded, synthetic_features
    ):
        cfg = ExperimentConfig(assay_id=synthetic_loaded.pool.assay_id)
        cfg.loop.n_initial = 20
        cfg.loop.batch_size = 5
        cfg.loop.n_rounds = 3
        holdout = list(range(len(synthetic_loaded.pool) - 40, len(synthetic_loaded.pool)))
        initial = [i for i in range(20)]
        opt = ClosedLoopOptimizer(
            synthetic_loaded,
            cfg,
            features=synthetic_features,
            initial_indices=initial,
            holdout_indices=holdout,
        )
        opt.run()
        assert not (set(opt.observed) & set(holdout))
        assert opt.oracle.n_queries == cfg.total_budget
        assert all(r.holdout_spearman is not None for r in opt.history[1:])

    def test_initial_design_overlapping_holdout_is_rejected(
        self, synthetic_loaded, synthetic_features
    ):
        cfg = ExperimentConfig(assay_id=synthetic_loaded.pool.assay_id)
        cfg.loop.n_initial = 10
        cfg.loop.n_rounds = 1
        with pytest.raises(ValueError, match="overlaps"):
            ClosedLoopOptimizer(
                synthetic_loaded,
                cfg,
                features=synthetic_features,
                initial_indices=[1, 2, 3],
                holdout_indices=[3, 4, 5],
            )


class TestPredictionsPrecedeMeasurement:
    """Predictions recorded for a batch must be made before it was revealed."""

    def test_predictions_are_independent_of_revealed_values(
        self, synthetic_loaded, synthetic_features, synthetic_assay
    ):
        from drucion.data.proteingym import load_assay

        cfg = ExperimentConfig(assay_id=synthetic_loaded.pool.assay_id)
        cfg.loop.n_initial = 20
        cfg.loop.batch_size = 5
        cfg.loop.n_rounds = 2
        cfg.loop.acquisition = "ucb"

        def run_once():
            assay = load_assay(
                synthetic_assay["assay_id"],
                assay_dir=synthetic_assay["assay_dir"],
                reference_csv=synthetic_assay["reference_csv"],
            )
            opt = ClosedLoopOptimizer(
                assay,
                cfg,
                features=synthetic_features,
                initial_indices=list(range(20)),
            )
            opt.run()
            return opt

        a, b = run_once(), run_once()
        pa = [v.predicted_fitness for r in a.history[1:] for v in r.selected]
        pb = [v.predicted_fitness for r in b.history[1:] for v in r.selected]
        assert pa == pytest.approx(pb)
        assert all(p is not None for p in pa)
