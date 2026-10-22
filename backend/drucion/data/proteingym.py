"""ProteinGym DMS substitution assay loader, validator and experimental oracle.

Scientific framing
------------------
Drucion runs a **retrospective simulation of experimental selection**. Every
fitness value it ever reveals is a real, previously published deep-mutational-
scanning (DMS) measurement from ProteinGym. No wet-lab experiment is performed
here, and no fitness value is ever invented.

The optimizer is given a :class:`CandidatePool` that contains only *mutation
specifications and sequences*. Measured ``DMS_score`` values live exclusively
inside :class:`ExperimentalOracle`, behind a counter, and become visible one
variant at a time as the optimizer spends its budget. That separation is what
makes the benchmark honest, and it is enforced by tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import PROTEINGYM_DIR, REFERENCE_CSV

#: The 20 canonical amino acids. ProteinGym substitution assays use only these.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = frozenset(AMINO_ACIDS)
_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


class AssayValidationError(ValueError):
    """Raised when an assay file violates the expected ProteinGym contract."""


# ---------------------------------------------------------------------------
# Mutation parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Substitution:
    """One single-residue substitution in ProteinGym's 1-indexed convention."""

    wt_aa: str
    position: int  # 1-indexed into the wild-type sequence
    mut_aa: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.wt_aa}{self.position}{self.mut_aa}"


def parse_mutant(mutant: str) -> list[Substitution]:
    """Parse a ProteinGym ``mutant`` string (e.g. ``"H24C"`` or ``"H24C:A42G"``).

    Raises :class:`AssayValidationError` on anything that is not a list of
    colon-separated single-residue substitutions over the canonical alphabet.
    """
    subs: list[Substitution] = []
    for token in mutant.split(":"):
        m = _MUTATION_RE.match(token.strip())
        if m is None:
            raise AssayValidationError(f"Unparseable mutation token: {token!r}")
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if wt not in _AA_SET or mut not in _AA_SET:
            raise AssayValidationError(
                f"Non-canonical amino acid in mutation {token!r}"
            )
        if pos < 1:
            raise AssayValidationError(f"Non-positive position in {token!r}")
        subs.append(Substitution(wt, pos, mut))
    if not subs:
        raise AssayValidationError(f"Empty mutant specification: {mutant!r}")
    return subs


def apply_mutations(wild_type: str, mutant: str) -> str:
    """Apply a ``mutant`` specification to ``wild_type`` and return the sequence."""
    seq = list(wild_type)
    for sub in parse_mutant(mutant):
        idx = sub.position - 1
        if idx >= len(seq):
            raise AssayValidationError(
                f"Mutation {sub} is outside the wild-type sequence "
                f"(length {len(wild_type)})"
            )
        if seq[idx] != sub.wt_aa:
            raise AssayValidationError(
                f"Mutation {sub} expects {sub.wt_aa} at position {sub.position} "
                f"but wild type has {seq[idx]}"
            )
        seq[idx] = sub.mut_aa
    return "".join(seq)


# ---------------------------------------------------------------------------
# Assay metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssayMetadata:
    """Provenance and biology for one DMS assay, surfaced in the UI."""

    assay_id: str
    uniprot_id: str
    molecule_name: str
    source_organism: str
    selection_assay: str
    selection_type: str
    first_author: str
    title: str
    year: str
    seq_len: int
    n_mutants: int
    n_single_mutants: int
    n_multiple_mutants: int
    wild_type: str
    region_mutated: str = ""
    source: str = "ProteinGym v1.3 (DMS substitutions)"
    license: str = "MIT (ProteinGym); see individual assay publications"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


@lru_cache(maxsize=1)
def _reference_table(path: str | None = None) -> pd.DataFrame:
    ref = Path(path) if path else REFERENCE_CSV
    if not ref.exists():
        raise FileNotFoundError(
            f"ProteinGym reference file not found at {ref}. "
            "Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(ref)


def list_assays(
    reference_csv: str | None = None,
    assay_dir: Path | None = None,
    only_local: bool = True,
) -> list[dict]:
    """Return lightweight descriptors for assays available on disk."""
    ref = _reference_table(reference_csv)
    directory = assay_dir or PROTEINGYM_DIR
    out: list[dict] = []
    for _, row in ref.iterrows():
        assay_id = row["DMS_id"]
        csv_path = directory / f"{assay_id}.csv"
        if only_local and not csv_path.exists():
            continue
        out.append(
            {
                "assay_id": assay_id,
                "molecule_name": str(row.get("molecule_name", "")),
                "source_organism": str(row.get("source_organism", "")),
                "selection_type": str(row.get("selection_type", "")),
                "selection_assay": str(row.get("selection_assay", "")),
                "seq_len": int(row["seq_len"]),
                "n_mutants": int(row["DMS_total_number_mutants"]),
                "n_single_mutants": int(row["DMS_number_single_mutants"]),
                "year": str(row.get("year", "")),
                "available": csv_path.exists(),
            }
        )
    return out


def load_assay_metadata(
    assay_id: str, reference_csv: str | None = None
) -> AssayMetadata:
    ref = _reference_table(reference_csv)
    rows = ref[ref["DMS_id"] == assay_id]
    if rows.empty:
        raise KeyError(f"Unknown ProteinGym assay id: {assay_id!r}")
    row = rows.iloc[0]
    return AssayMetadata(
        assay_id=assay_id,
        uniprot_id=str(row.get("UniProt_ID", "")),
        molecule_name=str(row.get("molecule_name", "")),
        source_organism=str(row.get("source_organism", "")),
        selection_assay=str(row.get("selection_assay", "")),
        selection_type=str(row.get("selection_type", "")),
        first_author=str(row.get("first_author", "")),
        title=str(row.get("title", "")),
        year=str(row.get("year", "")),
        seq_len=int(row["seq_len"]),
        n_mutants=int(row["DMS_total_number_mutants"]),
        n_single_mutants=int(row["DMS_number_single_mutants"]),
        n_multiple_mutants=int(row["DMS_number_multiple_mutants"]),
        wild_type=str(row["target_seq"]),
        region_mutated=str(row.get("region_mutated", "")),
    )


# ---------------------------------------------------------------------------
# Candidate pool (label-free) and oracle (label-holding)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidatePool:
    """The label-free view of an assay handed to the optimization agent.

    Contains mutation specifications and full sequences only. It deliberately
    has no attribute holding measured fitness.
    """

    assay_id: str
    wild_type: str
    mutants: tuple[str, ...]
    sequences: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.mutants)

    @property
    def size(self) -> int:
        return len(self.mutants)

    def index_of(self, mutant: str) -> int | None:
        """Index of ``mutant`` in the measured pool, or ``None`` if unmeasured."""
        return self._mutant_index.get(mutant)

    @property
    def _mutant_index(self) -> dict[str, int]:
        # Built lazily once; frozen dataclass so stash on the instance dict.
        cached = self.__dict__.get("_mutant_index_cache")
        if cached is None:
            cached = {m: i for i, m in enumerate(self.mutants)}
            object.__setattr__(self, "_mutant_index_cache", cached)
        return cached

    def mutated_positions(self, idx: int) -> list[int]:
        return [s.position for s in parse_mutant(self.mutants[idx])]


class ExperimentalOracle:
    """Holds the real measured DMS scores and reveals them only when queried.

    Every reveal is a lookup into previously published assay data. The oracle
    tracks a strict budget: :meth:`query` is the one and only route from a
    candidate index to a fitness number.
    """

    def __init__(self, assay_id: str, scores: Sequence[float]) -> None:
        self.assay_id = assay_id
        # Single underscore + no public accessor for the whole array: callers
        # must go through query(). Tests assert there is no bulk public getter.
        self._scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(self._scores)):
            raise AssayValidationError("Oracle received non-finite DMS scores")
        self._queried: set[int] = set()
        self.n_queries = 0

    def __len__(self) -> int:
        return int(self._scores.size)

    # -- the one legitimate access path -----------------------------------

    def query(self, indices: Iterable[int]) -> np.ndarray:
        """Reveal measured fitness for ``indices`` and charge the budget.

        Re-querying an already-revealed index is allowed (it is free, since the
        measurement is already known) but is *not* charged twice; the optimizer
        is responsible for not wasting budget on duplicates, and
        :meth:`n_queries` reflects the number of distinct variants measured.
        """
        idx = np.asarray(list(indices), dtype=int)
        if idx.size == 0:
            return np.empty(0, dtype=float)
        if idx.min() < 0 or idx.max() >= self._scores.size:
            raise IndexError("Oracle query outside candidate pool")
        for i in idx.tolist():
            if i not in self._queried:
                self._queried.add(i)
                self.n_queries += 1
        return self._scores[idx].copy()

    # -- aggregate statistics used only for reporting / plot reference lines

    @property
    def revealed_indices(self) -> list[int]:
        return sorted(self._queried)

    def ground_truth_summary(self) -> dict[str, float]:
        """Pool-level statistics used *only* for evaluation and plot axes.

        This is never exposed to acquisition or surrogate code; the benchmark
        reports regret against the pool maximum, which requires knowing it
        after the fact.
        """
        s = self._scores
        return {
            "pool_max": float(s.max()),
            "pool_min": float(s.min()),
            "pool_mean": float(s.mean()),
            "pool_std": float(s.std()),
            "p90": float(np.quantile(s, 0.90)),
            "p99": float(np.quantile(s, 0.99)),
        }

    def top_quantile_indices(self, q: float = 0.99) -> set[int]:
        """Indices whose measured score is at or above quantile ``q``."""
        thresh = float(np.quantile(self._scores, q))
        return set(np.flatnonzero(self._scores >= thresh).tolist())

    def peek_for_evaluation(self, indices: Iterable[int]) -> np.ndarray:
        """Read measurements **without** charging the budget — reporting only.

        This exists so the surrogate can be scored against a fixed holdout that
        is excluded from the candidate pool, which is the only statistically
        clean way to track predictive accuracy during a run. It must never be
        called from acquisition or training code; the loop calls it only after
        a round's selection has already been committed, and
        ``tests/test_no_leakage.py`` pins that behaviour.
        """
        idx = np.asarray(list(indices), dtype=int)
        if idx.size == 0:
            return np.empty(0, dtype=float)
        return self._scores[idx].copy()

    def all_scores_for_evaluation(self) -> np.ndarray:
        """Copy of every measured score, for post-hoc evaluation only.

        Named explicitly so that any use inside the optimization path is
        obvious in review and caught by tests.
        """
        return self._scores.copy()


@dataclass(frozen=True)
class LoadedAssay:
    """Everything a run needs: metadata, label-free pool, and sealed oracle."""

    metadata: AssayMetadata
    pool: CandidatePool
    oracle: ExperimentalOracle
    validation: dict


def load_assay(
    assay_id: str,
    assay_dir: Path | None = None,
    reference_csv: str | None = None,
    pool_size: int | None = None,
    singles_only: bool = True,
    seed: int = 0,
) -> LoadedAssay:
    """Load, validate and seal one ProteinGym substitution assay.

    Parameters
    ----------
    pool_size:
        Optionally subsample the candidate pool to this many variants
        (deterministic given ``seed``). The subsample is drawn **without
        looking at fitness**, so it does not bias the comparison.
    singles_only:
        Keep only single-residue substitutions. Keeps the mutation-feature
        baseline well defined and the biology interpretable.
    """
    directory = assay_dir or PROTEINGYM_DIR
    csv_path = Path(directory) / f"{assay_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Assay file not found: {csv_path}")

    meta = load_assay_metadata(assay_id, reference_csv)
    df = pd.read_csv(csv_path)

    required = {"mutant", "mutated_sequence", "DMS_score"}
    missing = required - set(df.columns)
    if missing:
        raise AssayValidationError(f"{assay_id}: missing columns {sorted(missing)}")

    report: dict = {"rows_in_file": int(len(df))}

    # 1. Drop rows with missing scores or sequences.
    before = len(df)
    df = df.dropna(subset=["mutant", "mutated_sequence", "DMS_score"])
    report["dropped_missing"] = before - len(df)

    # 2. Keep single mutants if requested.
    if singles_only:
        before = len(df)
        df = df[~df["mutant"].str.contains(":", regex=False)]
        report["dropped_multiple_mutants"] = before - len(df)
    else:
        report["dropped_multiple_mutants"] = 0

    # 3. Validate alphabet and sequence length against the wild type.
    wt = meta.wild_type
    report["wild_type_length"] = len(wt)
    bad_alphabet, bad_length, inconsistent = [], [], []
    for mutant, seq in zip(df["mutant"], df["mutated_sequence"]):
        if len(seq) != len(wt):
            bad_length.append(mutant)
            continue
        if not _AA_SET.issuperset(seq):
            bad_alphabet.append(mutant)
            continue
        try:
            if apply_mutations(wt, mutant) != seq:
                inconsistent.append(mutant)
        except AssayValidationError:
            inconsistent.append(mutant)
    drop = set(bad_alphabet) | set(bad_length) | set(inconsistent)
    report["dropped_bad_alphabet"] = len(bad_alphabet)
    report["dropped_length_mismatch"] = len(bad_length)
    report["dropped_mutation_sequence_inconsistent"] = len(inconsistent)
    if drop:
        df = df[~df["mutant"].isin(drop)]

    # 4. Deduplicate identical sequences (keep first occurrence).
    before = len(df)
    df = df.drop_duplicates(subset=["mutated_sequence"], keep="first")
    report["dropped_duplicate_sequences"] = before - len(df)

    df = df.sort_values("mutant", kind="mergesort").reset_index(drop=True)
    if df.empty:
        raise AssayValidationError(f"{assay_id}: no valid variants survived validation")

    # 5. Optional label-free subsampling.
    if pool_size is not None and pool_size < len(df):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(df), size=pool_size, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
        report["subsampled_to"] = int(pool_size)

    report["pool_size"] = int(len(df))

    pool = CandidatePool(
        assay_id=assay_id,
        wild_type=wt,
        mutants=tuple(df["mutant"].tolist()),
        sequences=tuple(df["mutated_sequence"].tolist()),
    )
    oracle = ExperimentalOracle(assay_id, df["DMS_score"].to_numpy(dtype=float))
    return LoadedAssay(metadata=meta, pool=pool, oracle=oracle, validation=report)
