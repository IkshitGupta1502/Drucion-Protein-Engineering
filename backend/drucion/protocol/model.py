"""The protocol data model, and the provenance rule it enforces.

Drucion's governing principle is that it never presents an invented number as
a measurement. In a benchmark that principle is enforced by
:class:`~drucion.data.proteingym.ExperimentalOracle`. In a bench protocol the
same principle matters more, because the reader acts on the number with real
materials: a wrong buffer pH costs a week and a batch of purified protein,
and it looks exactly as plausible on the page as a right one.

So provenance is part of the type, not a convention:

``sourced``
    Taken from a specific record or publication — a UniProt annotation, the
    originating deep-mutational-scanning paper's own conditions. Carries a
    citation.
``standard``
    Protein-independent laboratory technique. True regardless of which protein
    is being handled, so it needs no per-protein source.
``estimated``
    Inferred, including by a language model. **Must** carry both the reasoning
    and an experiment that would confirm it, and is rendered as a range rather
    than a point value wherever the underlying claim is a range.

:class:`Fact` refuses to be constructed if an estimate lacks its verification
step, so a protocol physically cannot contain an unchecked invented number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["sourced", "standard", "estimated"]

TIER_LABELS: dict[str, str] = {
    "sourced": "Sourced",
    "standard": "Standard technique",
    "estimated": "ESTIMATE — verify",
}

#: Rendered beside each tier so a reader knows what they are looking at
#: without having to learn the vocabulary first.
TIER_NOTES: dict[str, str] = {
    "sourced": "from a cited record or publication",
    "standard": "general technique, not specific to this protein",
    "estimated": "inferred, not measured — confirm before relying on it",
}


class ProtocolError(ValueError):
    """Raised when a protocol would violate the provenance rule."""


@dataclass
class Fact:
    """One statement in a protocol, carrying where it came from.

    Parameters
    ----------
    label:
        What the value describes ("Optimum pH", "Incubation temperature").
    value:
        The statement itself. For an estimate this should be a range, not a
        point value, whenever the underlying claim is uncertain.
    tier:
        Provenance. See the module docstring.
    source:
        For ``sourced``, the citation or record it came from. For
        ``estimated``, the reasoning behind the inference.
    source_url:
        Optional link backing ``source``.
    verify:
        How to confirm the value experimentally. Required for ``estimated``.
    """

    label: str
    value: str
    tier: Tier = "standard"
    source: str = ""
    source_url: str = ""
    verify: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIER_LABELS:
            raise ProtocolError(
                f"Unknown provenance tier {self.tier!r} for {self.label!r}; "
                f"expected one of {sorted(TIER_LABELS)}"
            )
        if self.tier == "estimated":
            # The whole point of the tier. An estimate the reader cannot check
            # is indistinguishable from an invented measurement.
            if not self.verify.strip():
                raise ProtocolError(
                    f"Estimated fact {self.label!r} has no verification step. "
                    "Every estimate must say how to confirm it experimentally."
                )
            if not self.source.strip():
                raise ProtocolError(
                    f"Estimated fact {self.label!r} has no reasoning. "
                    "Every estimate must say what it was inferred from."
                )
        if self.tier == "sourced" and not self.source.strip():
            raise ProtocolError(
                f"Sourced fact {self.label!r} has no citation. "
                "A fact with no source is an estimate; label it as one."
            )

    @property
    def is_estimate(self) -> bool:
        return self.tier == "estimated"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "tier": self.tier,
            "source": self.source,
            "source_url": self.source_url,
            "verify": self.verify,
        }


@dataclass
class Section:
    """A titled block of a protocol: some prose, some facts, some steps."""

    title: str
    body: str = ""
    facts: list[Fact] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    @property
    def estimates(self) -> list[Fact]:
        return [f for f in self.facts if f.is_estimate]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "body": self.body,
            "facts": [f.to_dict() for f in self.facts],
            "steps": list(self.steps),
        }


@dataclass
class Protocol:
    """A bench protocol for one assay, with every number's provenance attached."""

    assay_id: str
    title: str
    sections: list[Section] = field(default_factory=list)
    #: Variants the optimizer proposed, if the protocol was generated for a
    #: specific round. Empty for a general protocol.
    worklist: list[str] = field(default_factory=list)
    generated_by: str = ""
    notes: str = ""

    @property
    def facts(self) -> list[Fact]:
        return [f for s in self.sections for f in s.facts]

    @property
    def estimates(self) -> list[Fact]:
        """Every unverified inference in the protocol, for the summary banner."""
        return [f for f in self.facts if f.is_estimate]

    def tier_counts(self) -> dict[str, int]:
        counts = {t: 0 for t in TIER_LABELS}
        for f in self.facts:
            counts[f.tier] += 1
        return counts

    def section(self, title: str) -> Section | None:
        for s in self.sections:
            if s.title == title:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "assay_id": self.assay_id,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "worklist": list(self.worklist),
            "generated_by": self.generated_by,
            "notes": self.notes,
            "tier_counts": self.tier_counts(),
        }
