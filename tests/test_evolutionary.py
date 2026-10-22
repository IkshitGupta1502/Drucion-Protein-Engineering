"""ESM-guided evolutionary baseline: ranking, elite focusing, and the
unvalidated-proposal contract."""

from __future__ import annotations

import numpy as np
import pytest

from drucion.data.proteingym import CandidatePool
from drucion.optimization.evolutionary import ESMEvolutionaryRanker, ProposalReport


@pytest.fixture
def pool():
    wt = "MKTAYIAKQRQ"
    specs = [(2, "R"), (2, "E"), (4, "G"), (6, "V"), (8, "L"), (10, "H")]
    mutants, seqs = [], []
    for pos, mut in specs:
        mutants.append(f"{wt[pos - 1]}{pos}{mut}")
        seqs.append(wt[: pos - 1] + mut + wt[pos:])
    return CandidatePool(
        assay_id="TINY", wild_type=wt, mutants=tuple(mutants), sequences=tuple(seqs)
    )


@pytest.fixture
def ranker(pool):
    # Deliberately increasing zero-shot scores so ordering is predictable.
    zs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    return ESMEvolutionaryRanker(pool, zs, rng=np.random.default_rng(0))


class TestScoring:
    def test_without_observations_ranks_by_zero_shot(self, ranker):
        scores = ranker.score([], np.array([]))
        assert np.argmax(scores) == 5
        assert np.array_equal(np.argsort(scores), np.arange(6))

    def test_score_length_matches_pool(self, ranker, pool):
        assert ranker.score([0, 1], np.array([1.0, 0.0])).shape == (len(pool),)

    def test_elite_positions_promoted_above_everything_else(self, ranker):
        # Observe index 0 (K2R) as the single elite -> position 2 is "hot".
        scores = ranker.score([0], np.array([5.0]))
        # Indices 0 and 1 both mutate position 2, so both must outrank the rest
        # even though their raw zero-shot scores are the lowest.
        hot = {0, 1}
        cold_max = max(scores[i] for i in range(6) if i not in hot)
        assert all(scores[i] > cold_max for i in hot)

    def test_within_elite_group_zero_shot_still_decides(self, ranker):
        scores = ranker.score([0], np.array([5.0]))
        # index 1 has higher zero-shot than index 0, both at position 2.
        assert scores[1] > scores[0]

    def test_elite_fraction_controls_how_many_count(self, pool):
        zs = np.zeros(len(pool))
        r = ESMEvolutionaryRanker(pool, zs, elite_fraction=0.5)
        observed = [0, 2, 3, 4]
        y = np.array([4.0, 3.0, 2.0, 1.0])
        # Top half of 4 observations = 2 elites -> positions 2 and 4.
        assert r.elite_positions(observed, y) == {2, 4}

    def test_elite_positions_empty_without_observations(self, ranker):
        assert ranker.elite_positions([], np.array([])) == set()

    def test_at_least_one_elite_is_always_kept(self, pool):
        r = ESMEvolutionaryRanker(pool, np.zeros(len(pool)), elite_fraction=0.01)
        assert len(r.elite_positions([0, 1], np.array([1.0, 0.0]))) == 1


class TestProposals:
    def test_no_proposals_without_elites(self, ranker):
        report = ranker.propose_novel_variants([], np.array([]), n=10)
        assert report.n_proposed == 0
        assert report.n_in_measured_pool == 0
        assert report.n_unvalidated == 0

    def test_proposals_split_into_measured_and_unvalidated(self, ranker):
        report = ranker.propose_novel_variants([0], np.array([5.0]), n=30)
        assert report.n_proposed > 0
        assert report.n_proposed == report.n_in_measured_pool + report.n_unvalidated

    def test_unvalidated_proposals_are_genuinely_absent_from_the_pool(
        self, ranker, pool
    ):
        """The honesty contract: nothing outside the assay gets a score."""
        ranker.propose_novel_variants([0], np.array([5.0]), n=40)
        for spec in ranker.last_report.unvalidated_examples:
            assert pool.index_of(spec) is None

    def test_proposals_never_substitute_the_wild_type_residue(self, ranker, pool):
        from drucion.data.proteingym import parse_mutant

        ranker.propose_novel_variants([0], np.array([5.0]), n=40)
        for spec in ranker.last_report.unvalidated_examples:
            (sub,) = parse_mutant(spec)
            assert sub.mut_aa != sub.wt_aa
            assert pool.wild_type[sub.position - 1] == sub.wt_aa

    def test_proposals_only_touch_elite_positions(self, ranker):
        from drucion.data.proteingym import parse_mutant

        ranker.propose_novel_variants([2], np.array([5.0]), n=40)  # T4G -> pos 4
        for spec in ranker.last_report.unvalidated_examples:
            (sub,) = parse_mutant(spec)
            assert sub.position == 4

    def test_is_reproducible_with_a_fixed_seed(self, pool):
        zs = np.arange(len(pool), dtype=float)
        a = ESMEvolutionaryRanker(pool, zs, rng=np.random.default_rng(7))
        b = ESMEvolutionaryRanker(pool, zs, rng=np.random.default_rng(7))
        ra = a.propose_novel_variants([0], np.array([1.0]), n=20)
        rb = b.propose_novel_variants([0], np.array([1.0]), n=20)
        assert ra.to_dict() == rb.to_dict()

    def test_report_serialises_and_truncates_examples(self):
        report = ProposalReport(
            n_proposed=10,
            n_in_measured_pool=2,
            n_unvalidated=8,
            unvalidated_examples=[f"A{i}C" for i in range(8)],
        )
        d = report.to_dict()
        assert d["n_proposed"] == 10
        assert len(d["unvalidated_examples"]) == 5


class TestLoopIntegration:
    def test_evolutionary_requires_zero_shot_scores(self, synthetic_loaded, synthetic_features):
        from drucion.config import ExperimentConfig
        from drucion.optimization.loop import ClosedLoopOptimizer

        cfg = ExperimentConfig(assay_id=synthetic_loaded.pool.assay_id)
        cfg.loop.acquisition = "evolutionary"
        cfg.loop.n_initial = 20
        cfg.loop.n_rounds = 1
        with pytest.raises(ValueError, match="zero-shot"):
            ClosedLoopOptimizer(synthetic_loaded, cfg, features=synthetic_features)

    def test_evolutionary_run_respects_budget_and_reports_proposals(
        self, synthetic_loaded, synthetic_features
    ):
        from drucion.config import ExperimentConfig
        from drucion.optimization.loop import ClosedLoopOptimizer

        cfg = ExperimentConfig(assay_id=synthetic_loaded.pool.assay_id)
        cfg.loop.acquisition = "evolutionary"
        cfg.loop.n_initial = 20
        cfg.loop.batch_size = 5
        cfg.loop.n_rounds = 3
        zs = np.random.default_rng(0).standard_normal(len(synthetic_loaded.pool))
        opt = ClosedLoopOptimizer(
            synthetic_loaded, cfg, features=synthetic_features, zero_shot_scores=zs
        )
        opt.run()
        assert opt.oracle.n_queries == cfg.total_budget
        assert len(set(opt.observed)) == len(opt.observed)
        for record in opt.history[1:]:
            assert record.proposal_report is not None
            r = record.proposal_report
            assert r["n_proposed"] == r["n_in_measured_pool"] + r["n_unvalidated"]
