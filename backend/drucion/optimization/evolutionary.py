"""ESM-guided evolutionary candidate ranking — the non-learning baseline.

This is Drucion's reimplementation of the search style used by EvoEmu
(github.com/ibrahimaasim77/EvoEmu): ESM-2 proposes biologically plausible
substitutions, elite variants from previous rounds seed the next proposal
round, and the search walks outward from the consensus of those elites. EvoEmu
scored candidates with a structural model and never trained anything; here the
same proposal machinery is kept but candidates are ranked by ESM-2 zero-shot
plausibility, giving a baseline that uses experimental feedback **only** to
choose which positions to explore, with no surrogate fitness model.

Honesty constraint
------------------
A proposal is only experimentally queryable if that exact variant exists in the
measured ProteinGym pool. Proposals outside the pool are reported as
``unvalidated`` and are never assigned an invented fitness value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.proteingym import CandidatePool, parse_mutant


@dataclass
class ProposalReport:
    """Bookkeeping for one round of generative proposal."""

    n_proposed: int = 0
    n_in_measured_pool: int = 0
    n_unvalidated: int = 0
    unvalidated_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_proposed": self.n_proposed,
            "n_in_measured_pool": self.n_in_measured_pool,
            "n_unvalidated": self.n_unvalidated,
            "unvalidated_examples": self.unvalidated_examples[:5],
        }


class ESMEvolutionaryRanker:
    """Ranks pool candidates by ESM-2 plausibility, focused by elite feedback.

    Parameters
    ----------
    zero_shot_scores:
        Precomputed ESM-2 masked-marginal score per pool candidate. Label-free.
    elite_fraction:
        Fraction of observed variants treated as elites each round.
    """

    def __init__(
        self,
        pool: CandidatePool,
        zero_shot_scores: np.ndarray,
        elite_fraction: float = 0.2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.pool = pool
        self.zero_shot = np.asarray(zero_shot_scores, dtype=float)
        self.elite_fraction = elite_fraction
        self.rng = rng or np.random.default_rng(0)
        self._positions = [
            {s.position for s in parse_mutant(m)} for m in pool.mutants
        ]
        self.last_report = ProposalReport()

    # ------------------------------------------------------------------

    def elite_positions(
        self, observed: list[int], observed_y: np.ndarray
    ) -> set[int]:
        """Sequence positions mutated by the best-measured variants so far."""
        if not observed:
            return set()
        n_elite = max(1, int(round(self.elite_fraction * len(observed))))
        order = np.argsort(-np.asarray(observed_y))[:n_elite]
        positions: set[int] = set()
        for rank in order.tolist():
            positions |= self._positions[observed[rank]]
        return positions

    def score(self, observed: list[int], observed_y: np.ndarray) -> np.ndarray:
        """Acquisition-equivalent score for every pool candidate.

        Candidates that mutate a position favoured by the current elites are
        promoted above the rest; within each group, ESM-2 plausibility decides.
        No fitness is predicted and no surrogate is trained.
        """
        scores = self.zero_shot.copy()
        hot = self.elite_positions(observed, observed_y)
        if hot:
            # Rank-preserving offset: every elite-adjacent candidate outranks
            # every non-adjacent one, ESM score breaks ties inside each group.
            span = float(np.ptp(scores)) or 1.0
            adjacency = np.array(
                [1.0 if (p & hot) else 0.0 for p in self._positions]
            )
            scores = scores + adjacency * (span + 1.0)
        return scores

    # ------------------------------------------------------------------

    def propose_novel_variants(
        self, observed: list[int], observed_y: np.ndarray, n: int = 50
    ) -> ProposalReport:
        """Generate fresh single-substitution variants around the elites.

        This is the genuinely *generative* half of the evolutionary engine. It
        records how many proposals happen to exist in the measured pool, and
        flags the rest as unvalidated. The closed-loop comparison only ever
        queries the in-pool subset.
        """
        from ..data.proteingym import AMINO_ACIDS

        report = ProposalReport()
        hot = sorted(self.elite_positions(observed, observed_y))
        if not hot:
            self.last_report = report
            return report
        wt = self.pool.wild_type
        seen: set[str] = set()
        for _ in range(n):
            pos = int(self.rng.choice(hot))
            wt_aa = wt[pos - 1]
            choices = [a for a in AMINO_ACIDS if a != wt_aa]
            mut_aa = str(self.rng.choice(choices))
            spec = f"{wt_aa}{pos}{mut_aa}"
            if spec in seen:
                continue
            seen.add(spec)
            report.n_proposed += 1
            if self.pool.index_of(spec) is not None:
                report.n_in_measured_pool += 1
            else:
                report.n_unvalidated += 1
                report.unvalidated_examples.append(spec)
        self.last_report = report
        return report
