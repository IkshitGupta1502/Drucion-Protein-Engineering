"""The closed loop: predict -> acquire -> measure -> retrain -> repeat.

This module is Drucion's scientific core. One :class:`ClosedLoopOptimizer`
owns a single run: a candidate pool, a sealed oracle, a feature matrix and a
surrogate. :meth:`step` performs exactly one experimental round and returns a
fully-populated :class:`RoundRecord`, which is what both the CLI benchmark and
the live dashboard render.

Budget discipline
-----------------
* Initial observations are drawn uniformly at random from the pool, seeded, and
  are **identical across methods for a given seed** — the loop is handed them
  rather than choosing them, so no method gets a better start.
* A variant already measured is never selected again; the available mask is
  updated before any acquisition score is computed.
* ``n_measured`` is read back from the oracle, not tracked separately, so the
  reported budget is the oracle's own count.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import ExperimentConfig
from ..data.proteingym import LoadedAssay
from ..embeddings.features import FeatureSet, build_features
from ..models.surrogate import build_surrogate
from .acquisition import (
    ACQUISITIONS,
    AcquisitionContext,
    pairwise_distance_quantile,
    select_batch,
)
from .evolutionary import ESMEvolutionaryRanker

logger = logging.getLogger(__name__)


def _safe_spearman(pred: np.ndarray, actual: np.ndarray) -> float | None:
    """Spearman correlation, or ``None`` when it is undefined (no variance)."""
    from scipy.stats import spearmanr

    if len(pred) < 3 or np.std(pred) < 1e-12 or np.std(actual) < 1e-12:
        return None
    rho = spearmanr(pred, actual).statistic
    return None if np.isnan(rho) else float(rho)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class VariantRecord:
    """One selected variant, with prediction and revealed measurement side by side."""

    index: int
    mutant: str
    positions: list[int]
    #: Surrogate posterior mean *before* this variant was measured. None for
    #: methods that do not use a surrogate to select.
    predicted_fitness: float | None
    #: Surrogate posterior standard deviation before measurement.
    predicted_uncertainty: float | None
    acquisition_score: float
    #: The real, previously published DMS measurement, revealed by the oracle.
    measured_fitness: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PendingBatch:
    """A proposed batch that has been chosen but not yet measured.

    This is the hand-off point between the two halves of the loop. In the
    retrospective benchmark the oracle fills it in immediately. In a real
    campaign it is exported as a worklist, the bench measures it, and the
    values come back through :meth:`ClosedLoopOptimizer.ingest`.
    """

    round: int
    indices: list[int]
    mutants: list[str]
    sequences: list[str]
    positions: list[list[int]]
    predicted_fitness: list[float | None]
    predicted_uncertainty: list[float | None]
    acquisition_score: list[float]
    #: Surrogate state captured at proposal time, so the record reflects the
    #: model that actually made the choice rather than a later refit.
    surrogate_kind: str = "none"
    surrogate_info: dict = field(default_factory=dict)
    holdout_spearman: float | None = None
    holdout_rmse: float | None = None
    proposal_report: dict | None = None
    proposed_at: float = 0.0

    def __len__(self) -> int:
        return len(self.indices)

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "variants": [
                {
                    "index": i,
                    "mutant": m,
                    "positions": p,
                    "predicted_fitness": pf,
                    "predicted_uncertainty": pu,
                    "acquisition_score": a,
                }
                for i, m, p, pf, pu, a in zip(
                    self.indices,
                    self.mutants,
                    self.positions,
                    self.predicted_fitness,
                    self.predicted_uncertainty,
                    self.acquisition_score,
                )
            ],
            "surrogate_kind": self.surrogate_kind,
            "holdout_spearman": self.holdout_spearman,
        }


@dataclass
class RoundRecord:
    """Complete snapshot of one experimental round."""

    round: int
    n_measured: int
    best_measured: float
    mean_measured: float
    selected: list[VariantRecord] = field(default_factory=list)
    surrogate_kind: str = ""
    surrogate_info: dict = field(default_factory=dict)
    #: Spearman correlation of surrogate predictions against the measurements
    #: revealed in *this round's batch*. Computed after selection was already
    #: committed, so it is out-of-sample — but the batch is a deliberately
    #: biased, range-restricted sample, so this number reads low and is not a
    #: measure of general accuracy. Use ``holdout_spearman`` for that.
    surrogate_spearman: float | None = None
    surrogate_rmse: float | None = None
    #: Spearman on a fixed random holdout that is excluded from the candidate
    #: pool entirely: never trainable, never selectable, never charged to
    #: budget. This is the honest predictive-accuracy curve.
    holdout_spearman: float | None = None
    holdout_rmse: float | None = None
    proposal_report: dict | None = None
    #: Where this round's numbers came from: "proteingym" for the
    #: retrospective oracle, "uploaded" for values returned from a bench.
    measurement_source: str = "proteingym"
    #: Proposed variants that came back without a measurement and were
    #: returned to the candidate pool.
    n_unmeasured: int = 0
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["selected"] = [v.to_dict() for v in self.selected]
        return d


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class ClosedLoopOptimizer:
    """Uncertainty-aware active learning over a measured ProteinGym pool."""

    def __init__(
        self,
        assay: LoadedAssay,
        config: ExperimentConfig,
        features: FeatureSet | None = None,
        zero_shot_scores: np.ndarray | None = None,
        initial_indices: list[int] | None = None,
        holdout_indices: list[int] | None = None,
        progress=None,
    ) -> None:
        self.assay = assay
        self.config = config
        self.pool = assay.pool
        self.oracle = assay.oracle
        self.features = features or build_features(
            assay.pool, config.embedding, progress=progress
        )
        self.X = self.features.X

        self.rng = np.random.default_rng(config.loop.seed)
        self.acquisition_name = config.loop.acquisition

        self.available = np.ones(len(self.pool), dtype=bool)
        self.observed: list[int] = []
        self.observed_y: list[float] = []
        self.history: list[RoundRecord] = []
        self.round = 0
        self.finished = False
        #: Batch chosen but not yet measured. Non-None means the loop is
        #: waiting on an experiment.
        self.pending: PendingBatch | None = None
        #: Every distinct provenance this run's measurements came from.
        self.measurement_sources: set[str] = set()
        #: Rounds added beyond the original plan, via :meth:`extend`.
        self.extended_by = 0

        self.surrogate = build_surrogate(config.surrogate, seed=config.loop.seed)
        self._uses_surrogate = self.acquisition_name in ACQUISITIONS

        self._ranker: ESMEvolutionaryRanker | None = None
        if self.acquisition_name == "evolutionary":
            if zero_shot_scores is None:
                raise ValueError(
                    "The evolutionary baseline requires precomputed ESM-2 "
                    "zero-shot scores for the pool."
                )
            self._ranker = ESMEvolutionaryRanker(
                self.pool,
                zero_shot_scores,
                rng=np.random.default_rng(config.loop.seed),
            )

        self._min_distance = pairwise_distance_quantile(
            self.X, config.loop.diversity_quantile, seed=config.loop.seed
        )

        # Fixed evaluation holdout: removed from the selectable pool so it can
        # never be trained on or measured, and therefore never costs budget.
        self.holdout: list[int] = list(holdout_indices or [])
        if self.holdout:
            self.available[self.holdout] = False
            self._holdout_y = self.oracle.peek_for_evaluation(self.holdout)
        else:
            self._holdout_y = np.empty(0)

        self._seed_initial_observations(initial_indices)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def draw_initial_indices(pool_size: int, n_initial: int, seed: int) -> list[int]:
        """Deterministic, label-free initial design shared by every method."""
        rng = np.random.default_rng(seed)
        n = min(n_initial, pool_size)
        return sorted(rng.choice(pool_size, size=n, replace=False).tolist())

    def _seed_initial_observations(self, initial_indices: list[int] | None) -> None:
        idx = initial_indices
        if idx is None:
            idx = self.draw_initial_indices(
                len(self.pool), self.config.loop.n_initial, self.config.loop.seed
            )
        overlap = set(idx) & set(self.holdout)
        if overlap:
            raise ValueError(
                f"Initial design overlaps the evaluation holdout at {sorted(overlap)}"
            )
        y = self.oracle.query(idx)
        self.observed = list(idx)
        self.observed_y = y.tolist()
        self.available[idx] = False
        self.history.append(
            RoundRecord(
                round=0,
                n_measured=len(self.observed),
                best_measured=float(np.max(y)),
                mean_measured=float(np.mean(y)),
                selected=[],
                surrogate_kind="(initial design)",
            )
        )

    # ------------------------------------------------------------------
    # One experimental round
    # ------------------------------------------------------------------

    def propose(self) -> PendingBatch:
        """Choose the next batch **without measuring it**.

        Fits the surrogate on everything measured so far, scores the
        unobserved pool, and selects a batch. Nothing is revealed and no
        budget is spent: the returned batch is a request for measurements.

        Calling this twice without an intervening :meth:`ingest` returns the
        same pending batch rather than re-proposing, so an exported worklist
        and the batch the optimizer is waiting on can never drift apart.
        """
        if self.finished:
            raise RuntimeError("Optimization already finished")
        if self.pending is not None:
            return self.pending

        t0 = time.time()
        round_no = self.round + 1

        y_obs = np.asarray(self.observed_y, dtype=float)
        X_obs = self.X[self.observed]
        best_observed = float(y_obs.max())

        mean = std = None
        surrogate_kind = "none"
        surrogate_info: dict = {}

        # 1. Retrain the surrogate on everything measured so far.
        #    Fitted for every method so surrogate accuracy is comparable, but
        #    only consulted for selection by the Bayesian/greedy strategies.
        try:
            self.surrogate.fit(X_obs, y_obs)
            mean, std = self.surrogate.predict(self.X)
            surrogate_kind = self.surrogate.kind
            surrogate_info = self.surrogate.fit_info
        except Exception as exc:  # noqa: BLE001
            logger.warning("Surrogate unavailable at round %s: %s", round_no, exc)
            if self._uses_surrogate:
                raise

        # 2. Score the unobserved pool.
        ctx = AcquisitionContext(
            best_observed=best_observed, beta=self.config.loop.ucb_beta
        )
        proposal_report = None
        if self.acquisition_name == "random":
            # Plain uniform sampling from the untested pool — the baseline a
            # lab actually runs. Deliberately NOT passed through the batch
            # diversity filter: random screening has no notion of diversity,
            # and filtering it would compare against a smarter-than-random
            # baseline.
            scores = self.rng.random(len(self.pool))
        elif self.acquisition_name == "evolutionary":
            assert self._ranker is not None
            scores = self._ranker.score(self.observed, y_obs)
            proposal_report = self._ranker.propose_novel_variants(
                self.observed, y_obs
            ).to_dict()
        else:
            scores = ACQUISITIONS[self.acquisition_name](mean, std, ctx)

        # 3. Select a batch of previously untested variants.
        chosen = select_batch(
            scores,
            self.available,
            self.config.loop.batch_size,
            features=self.X,
            min_distance=(
                0.0 if self.acquisition_name == "random" else self._min_distance
            ),
        )
        if not chosen:
            self.finished = True
            raise RuntimeError("Candidate pool exhausted")

        # 4. Held-out accuracy of the model that made this choice, measured
        #    before anything is revealed.
        holdout_rho = holdout_rmse = None
        if mean is not None and self.holdout:
            hpred = mean[self.holdout]
            holdout_rho = _safe_spearman(hpred, self._holdout_y)
            holdout_rmse = float(np.sqrt(np.mean((hpred - self._holdout_y) ** 2)))

        # Reserve the batch so a concurrent propose cannot pick it again.
        self.available[chosen] = False

        self.pending = PendingBatch(
            round=round_no,
            indices=[int(i) for i in chosen],
            mutants=[self.pool.mutants[i] for i in chosen],
            sequences=[self.pool.sequences[i] for i in chosen],
            positions=[self.pool.mutated_positions(i) for i in chosen],
            predicted_fitness=[
                float(mean[i]) if mean is not None else None for i in chosen
            ],
            predicted_uncertainty=[
                float(std[i]) if std is not None else None for i in chosen
            ],
            acquisition_score=[float(scores[i]) for i in chosen],
            surrogate_kind=surrogate_kind,
            surrogate_info=surrogate_info,
            holdout_spearman=holdout_rho,
            holdout_rmse=holdout_rmse,
            proposal_report=proposal_report,
            proposed_at=t0,
        )
        return self.pending

    def ingest(
        self,
        measurements: dict[str, float] | dict[int, float] | np.ndarray,
        source: str = "oracle",
    ) -> RoundRecord:
        """Fold measured values for the pending batch into the observed set.

        ``measurements`` maps mutant spec (or pool index) to fitness, or is an
        array aligned with the pending batch. Variants in the pending batch
        that are absent from ``measurements`` are treated as **not measured**:
        they are returned to the candidate pool so they can be proposed again,
        which is what happens when a well fails at the bench.

        ``source`` is recorded on the round so a run fed by uploaded bench
        data can never be mistaken for the retrospective benchmark.
        """
        if self.pending is None:
            raise RuntimeError("No pending batch — call propose() first")
        batch = self.pending

        # Normalise to {pool index: value}.
        by_index: dict[int, float] = {}
        if isinstance(measurements, np.ndarray):
            if measurements.shape[0] != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} values for the pending batch, "
                    f"got {measurements.shape[0]}"
                )
            by_index = {
                idx: float(v) for idx, v in zip(batch.indices, measurements)
            }
        else:
            index_of = {m: i for m, i in zip(batch.mutants, batch.indices)}
            for key, value in measurements.items():
                if isinstance(key, str):
                    if key not in index_of:
                        raise ValueError(
                            f"{key!r} is not in the pending batch for round "
                            f"{batch.round}"
                        )
                    by_index[index_of[key]] = float(value)
                else:
                    if int(key) not in set(batch.indices):
                        raise ValueError(
                            f"Index {key} is not in the pending batch for "
                            f"round {batch.round}"
                        )
                    by_index[int(key)] = float(value)

        if not by_index:
            raise ValueError("No usable measurements supplied")
        bad = [k for k, v in by_index.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"Non-finite measurement for pool index {bad[0]}")

        # Unmeasured proposals go back into the pool rather than being burned.
        unmeasured = [i for i in batch.indices if i not in by_index]
        if unmeasured:
            self.available[unmeasured] = True

        records: list[VariantRecord] = []
        for slot, idx in enumerate(batch.indices):
            if idx not in by_index:
                continue
            records.append(
                VariantRecord(
                    index=int(idx),
                    mutant=batch.mutants[slot],
                    positions=batch.positions[slot],
                    predicted_fitness=batch.predicted_fitness[slot],
                    predicted_uncertainty=batch.predicted_uncertainty[slot],
                    acquisition_score=batch.acquisition_score[slot],
                    measured_fitness=by_index[idx],
                )
            )
            self.observed.append(int(idx))
            self.observed_y.append(by_index[idx])

        revealed = np.array([r.measured_fitness for r in records], dtype=float)
        pred = np.array(
            [
                r.predicted_fitness if r.predicted_fitness is not None else np.nan
                for r in records
            ],
            dtype=float,
        )
        spearman = rmse = None
        if len(records) >= 3 and np.all(np.isfinite(pred)):
            spearman = _safe_spearman(pred, revealed)
            rmse = float(np.sqrt(np.mean((pred - revealed) ** 2)))

        self.round = batch.round
        self.pending = None
        self.measurement_sources.add(source)

        all_y = np.asarray(self.observed_y, dtype=float)
        record = RoundRecord(
            round=batch.round,
            n_measured=self.n_measured,
            best_measured=float(all_y.max()),
            mean_measured=float(all_y.mean()),
            selected=records,
            surrogate_kind=batch.surrogate_kind,
            surrogate_info=batch.surrogate_info,
            surrogate_spearman=spearman,
            surrogate_rmse=rmse,
            holdout_spearman=batch.holdout_spearman,
            holdout_rmse=batch.holdout_rmse,
            proposal_report=batch.proposal_report,
            measurement_source=source,
            n_unmeasured=len(unmeasured),
            wall_time_s=time.time() - batch.proposed_at,
        )
        self.history.append(record)
        if self.round >= self.config.loop.n_rounds:
            self.finished = True
        return record

    def step(self) -> RoundRecord:
        """One full retrospective round: propose, query the oracle, ingest.

        This is the benchmark path. The oracle is the only thing that makes it
        different from a real campaign, where :meth:`propose` and
        :meth:`ingest` are separated by an actual experiment.
        """
        batch = self.propose()
        revealed = self.oracle.query(batch.indices)
        return self.ingest(revealed, source="proteingym")

    def extend(self, additional_rounds: int = 1) -> int:
        """Allow more rounds than the run was originally planned for.

        ``n_rounds`` is a benchmark construct: every method gets the same fixed
        budget so the comparison is fair. A real campaign has no such stopping
        rule — you keep going while the protein keeps improving and the budget
        holds. This makes that explicit rather than silently moving the goal
        posts, so a run that was extended is visibly different from one that
        was not.

        Returns the new total round count.
        """
        if additional_rounds < 1:
            raise ValueError("additional_rounds must be at least 1")
        if not self.available.any():
            raise RuntimeError("Candidate pool exhausted — nothing left to test")
        self.config.loop.n_rounds += additional_rounds
        self.finished = False
        self.extended_by += additional_rounds
        return self.config.loop.n_rounds

    def run(self, progress=None) -> list[RoundRecord]:
        """Run every remaining round."""
        while not self.finished:
            rec = self.step()
            if progress:
                progress(rec)
        return self.history

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def n_measured(self) -> int:
        """Variants measured so far.

        Counted from the observed set so it is correct in both modes. In the
        retrospective benchmark it must equal the oracle's own query counter;
        that invariant is asserted rather than assumed, because the budget
        claims in the README depend on it.
        """
        n = len(self.observed)
        if self.measurement_sources <= {"proteingym"} and n != self.oracle.n_queries:
            raise RuntimeError(
                f"Budget accounting diverged: {n} observed vs "
                f"{self.oracle.n_queries} oracle queries"
            )
        return n

    @property
    def awaiting_measurements(self) -> bool:
        """True when a batch has been proposed and is waiting on results."""
        return self.pending is not None

    @property
    def best_index(self) -> int:
        return int(self.observed[int(np.argmax(self.observed_y))])

    @property
    def best_mutant(self) -> str:
        return self.pool.mutants[self.best_index]

    @property
    def best_measured(self) -> float:
        return float(np.max(self.observed_y))

    def learning_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """``(cumulative_measurements, best_measured_so_far)``."""
        n = np.array([r.n_measured for r in self.history], dtype=int)
        best = np.maximum.accumulate(
            np.array([r.best_measured for r in self.history], dtype=float)
        )
        return n, best

    def state_dict(self) -> dict:
        n, best = self.learning_curve()
        return {
            "assay_id": self.pool.assay_id,
            "acquisition": self.acquisition_name,
            "round": self.round,
            "n_rounds": self.config.loop.n_rounds,
            "finished": self.finished,
            "awaiting_measurements": self.awaiting_measurements,
            "extended_by": self.extended_by,
            "measurement_sources": sorted(self.measurement_sources),
            "n_measured": self.n_measured,
            "budget": self.config.total_budget,
            "pool_size": len(self.pool),
            "best_measured": self.best_measured,
            "best_mutant": self.best_mutant,
            "features": self.features.to_dict(),
            "surrogate_kind": self.surrogate.kind,
            "learning_curve": {
                "n_measured": n.tolist(),
                "best_measured": best.tolist(),
            },
            "history": [r.to_dict() for r in self.history],
        }
