"""Worklist export and bench-result intake — the prospective half of the loop.

The retrospective benchmark closes its loop against ProteinGym. A real
campaign closes it against a bench: Drucion proposes a batch, you export it as
a worklist, somebody runs the experiment, and the measured values come back
through :func:`parse_measurements`.

Provenance
----------
Values that arrive this way are **not** ProteinGym measurements and are never
treated as such. They are tagged ``source="uploaded"`` on the round that
consumes them, surfaced in the UI, and excluded from the retrospective
benchmark. Drucion takes an uploaded number at face value — it has no way to
audit it — so the honest thing is to label where it came from and keep it out
of any claim that rests on published data.

The CSV round trip is deliberately boring: a header comment block a human can
read, one row per variant, and a single blank column to fill in.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..data.proteingym import AssayMetadata, parse_mutant
from ..optimization.loop import PendingBatch

#: Columns written on export. ``measured_fitness`` is the one to fill in.
EXPORT_COLUMNS = [
    "mutant",
    "position",
    "wild_type_aa",
    "variant_aa",
    "predicted_fitness",
    "predicted_uncertainty",
    "acquisition_rank",
    "measured_fitness",
    "notes",
]

#: The only columns intake actually requires.
REQUIRED_COLUMNS = {"mutant", "measured_fitness"}


class WorklistError(ValueError):
    """The uploaded sheet could not be read as a worklist."""


def batch_token(experiment_id: str, batch: PendingBatch) -> str:
    """Short stable token identifying this exact proposed batch.

    Written into the exported sheet and checked on intake, so results from
    round 3 cannot be silently folded into round 5 after a re-propose.
    """
    h = hashlib.sha256()
    h.update(experiment_id.encode())
    h.update(str(batch.round).encode())
    for m in batch.mutants:
        h.update(m.encode())
        h.update(b",")
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_worklist(
    experiment_id: str,
    batch: PendingBatch,
    metadata: AssayMetadata,
    prefilled: Sequence[float] | None = None,
    prefill_note: str = "",
) -> str:
    """Render the pending batch as a CSV worklist.

    ``prefilled`` optionally fills ``measured_fitness`` — used to hand back
    real published values so the intake path can be exercised end to end
    without anybody inventing numbers. When used, the provenance of those
    values is stated in the header block.
    """
    buf = io.StringIO()
    token = batch_token(experiment_id, batch)

    buf.write(f"# Drucion worklist — {metadata.molecule_name or metadata.assay_id}\n")
    buf.write(f"# assay: {metadata.assay_id}\n")
    buf.write(f"# experiment: {experiment_id}\n")
    buf.write(f"# round: {batch.round}\n")
    buf.write(f"# batch_token: {token}\n")
    buf.write(f"# variants: {len(batch)}\n")
    buf.write("#\n")
    buf.write("# Fill in the measured_fitness column and upload this file back.\n")
    buf.write("# Higher values mean better fitness. Leave a row blank if that\n")
    buf.write("# variant was not measured; it returns to the candidate pool.\n")
    buf.write("# Do not edit the mutant column or the batch_token above.\n")
    if prefilled is not None:
        buf.write("#\n")
        buf.write(f"# PRE-FILLED: {prefill_note}\n")
    buf.write("#\n")

    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for rank, slot in enumerate(range(len(batch)), start=1):
        subs = parse_mutant(batch.mutants[slot])
        writer.writerow(
            {
                "mutant": batch.mutants[slot],
                "position": ";".join(str(s.position) for s in subs),
                "wild_type_aa": "".join(s.wt_aa for s in subs),
                "variant_aa": "".join(s.mut_aa for s in subs),
                "predicted_fitness": _fmt(batch.predicted_fitness[slot]),
                "predicted_uncertainty": _fmt(batch.predicted_uncertainty[slot]),
                "acquisition_rank": rank,
                "measured_fitness": (
                    # Exact round-trip: a prefilled value is read back in, so
                    # formatting it for looks would silently alter the data.
                    _exact(prefilled[slot]) if prefilled is not None else ""
                ),
                "notes": "",
            }
        )
    return buf.getvalue()


def _fmt(value: float | None) -> str:
    """Readable form for display-only columns (never read back in)."""
    return "" if value is None else f"{float(value):.6g}"


def _exact(value: float | None) -> str:
    """Shortest representation that round-trips to the same float."""
    return "" if value is None else repr(float(value))


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------


@dataclass
class IntakeResult:
    """What a parsed worklist actually contained."""

    measurements: dict[str, float] = field(default_factory=dict)
    #: Rows left blank — not measured, returned to the pool.
    skipped: list[str] = field(default_factory=list)
    #: (row label, reason) for anything that could not be used.
    rejected: list[tuple[str, str]] = field(default_factory=list)
    #: Variants proposed this round that the sheet did not mention at all.
    missing: list[str] = field(default_factory=list)
    token_in_file: str | None = None

    @property
    def n_accepted(self) -> int:
        return len(self.measurements)

    def to_dict(self) -> dict:
        return {
            "n_accepted": self.n_accepted,
            "n_skipped": len(self.skipped),
            "n_rejected": len(self.rejected),
            "n_missing": len(self.missing),
            "accepted": self.measurements,
            "skipped": self.skipped,
            "rejected": [{"row": r, "reason": why} for r, why in self.rejected],
            "missing": self.missing,
            "batch_token": self.token_in_file,
        }


def parse_measurements(
    text: str,
    expected: Iterable[str] | None = None,
    expected_token: str | None = None,
    strict_token: bool = True,
) -> IntakeResult:
    """Parse an uploaded worklist into measurements, reporting every problem.

    Nothing is silently dropped: a row that cannot be used appears in
    ``rejected`` with the reason, a blank row in ``skipped``, and a proposed
    variant the sheet omits entirely in ``missing``.

    Parameters
    ----------
    expected:
        The mutants currently proposed. Rows naming anything else are rejected
        rather than accepted, because a value for a variant the optimizer did
        not ask for has nowhere to go in this round.
    expected_token / strict_token:
        Guards against uploading a stale sheet from an earlier round.
    """
    expected_set = set(expected) if expected is not None else None
    result = IntakeResult()

    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("# batch_token:"):
            result.token_in_file = stripped.split(":", 1)[1].strip()
            break

    if (
        strict_token
        and expected_token is not None
        and result.token_in_file is not None
        and result.token_in_file != expected_token
    ):
        raise WorklistError(
            f"This sheet is for batch {result.token_in_file}, but the "
            f"experiment is waiting on batch {expected_token}. Export a fresh "
            "worklist for the current round."
        )

    body = "\n".join(l for l in lines if not l.lstrip().startswith("#"))
    if not body.strip():
        raise WorklistError("The file contains no rows.")

    reader = csv.DictReader(io.StringIO(body))
    if reader.fieldnames is None:
        raise WorklistError("The file has no header row.")
    headers = {(h or "").strip().lower() for h in reader.fieldnames}
    missing_cols = REQUIRED_COLUMNS - headers
    if missing_cols:
        raise WorklistError(
            f"Missing required column(s): {', '.join(sorted(missing_cols))}. "
            f"Found: {', '.join(sorted(h for h in headers if h))}."
        )

    seen: set[str] = set()
    for row_no, row in enumerate(reader, start=2):
        norm = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        mutant = norm.get("mutant", "")
        label = mutant or f"row {row_no}"

        if not mutant:
            result.rejected.append((label, "no mutant specified"))
            continue
        try:
            parse_mutant(mutant)
        except ValueError as exc:
            result.rejected.append((label, f"unparseable mutation: {exc}"))
            continue
        if mutant in seen:
            result.rejected.append((label, "duplicate row for this variant"))
            continue
        seen.add(mutant)
        if expected_set is not None and mutant not in expected_set:
            result.rejected.append(
                (label, "not part of the batch the optimizer is waiting on")
            )
            continue

        raw = norm.get("measured_fitness", "")
        if raw == "":
            result.skipped.append(mutant)
            continue
        try:
            value = float(raw)
        except ValueError:
            result.rejected.append((label, f"measured_fitness {raw!r} is not a number"))
            continue
        if not np.isfinite(value):
            result.rejected.append((label, "measured_fitness is not finite"))
            continue
        result.measurements[mutant] = value

    if expected_set is not None:
        result.missing = sorted(expected_set - seen)
    return result
