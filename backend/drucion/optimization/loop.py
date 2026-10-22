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
                n_measured=self.oracle.n_queries,
                best_measured=float(np.max(y)),
                mean_measured=float(np.mean(y)),
                selected=[],
                surrogate_kind="(initial design)",
            )
        )

    # ------------------------------------------------------------------
    # One experimental round
    # ------------------------------------------------------------------

    def step(self) -> RoundRecord:
        """Run one round: fit, predict, acquire, measure, record."""
        if self.finished:
            raise RuntimeError("Optimization already finished")
        t0 = time.time()
        self.round += 1

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
            logger.warning("Surrogate unavailable at round %s: %s", self.round, exc)
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

        # 4. Query the experimental-data oracle. This is the only reveal path.
        revealed = self.oracle.query(chosen)

        # 5. Fold the new measurements into the observed dataset.
        records: list[VariantRecord] = []
        for i, idx in enumerate(chosen):
            records.append(
                VariantRecord(
                    index=int(idx),
                    mutant=self.pool.mutants[idx],
                    positions=self.pool.mutated_positions(idx),
                    predicted_fitness=(
                        float(mean[idx]) if mean is not None else None
                    ),
                    predicted_uncertainty=(
                        float(std[idx]) if std is not None else None
                    ),
                    acquisition_score=float(scores[idx]),
                    measured_fitness=float(revealed[i]),
                )
            )
        self.available[chosen] = False
        self.observed.extend(int(i) for i in chosen)
        self.observed_y.extend(revealed.tolist())

        # 6. Honest held-out accuracy: predictions were made before these
        #    measurements were revealed, so this round's batch is a genuine
        #    out-of-sample test set for the surrogate.
        spearman = rmse = None
        if mean is not None and len(chosen) >= 3:
            pred = mean[chosen]
            spearman = _safe_spearman(pred, revealed)
            rmse = float(np.sqrt(np.mean((pred - revealed) ** 2)))

        #    The holdout is a fixed, unbiased, never-selectable sample, so this
        #    is the number to quote for surrogate accuracy.
        holdout_rho = holdout_rmse = None
        if mean is not None and self.holdout:
            hpred = mean[self.holdout]
            holdout_rho = _safe_spearman(hpred, self._holdout_y)
            holdout_rmse = float(np.sqrt(np.mean((hpred - self._holdout_y) ** 2)))

        all_y = np.asarray(self.observed_y, dtype=float)
        record = RoundRecord(
            round=self.round,
            n_measured=self.oracle.n_queries,
            best_measured=float(all_y.max()),
            mean_measured=float(all_y.mean()),
            selected=records,
            surrogate_kind=surrogate_kind,
            surrogate_info=surrogate_info,
            surrogate_spearman=spearman,
            surrogate_rmse=rmse,
            holdout_spearman=holdout_rho,
            holdout_rmse=holdout_rmse,
            proposal_report=proposal_report,
            wall_time_s=time.time() - t0,
        )
        self.history.append(record)
        if self.round >= self.config.loop.n_rounds:
            self.finished = True
        return record

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
            "n_measured": self.oracle.n_queries,
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
