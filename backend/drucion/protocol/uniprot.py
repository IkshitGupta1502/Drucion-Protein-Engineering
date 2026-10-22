"""UniProt lookup for experimentally determined protein conditions.

Why this exists
---------------
A bench protocol states numbers someone will act on: a buffer pH, an
incubation temperature, a cofactor that must be present. Drucion's governing
principle is that it never invents a measured value, and a protocol is where
inventing one stops being a bad prediction and starts being a wasted week of
bench time and a batch of purified protein.

So the protocol generator looks for real annotations first. UniProt records
carry pH and temperature optima, cofactors, catalytic activity, subcellular
location and PDB cross-references, most of them with evidence codes pointing
at a specific publication. Anything found here is a **sourced** fact and is
cited. Anything not found here is either protein-independent boilerplate or an
explicitly flagged estimate — see :mod:`drucion.protocol.generate`.

ProteinGym identifies proteins by UniProt *entry name* (``TPMT_HUMAN``), not
accession (``P51580``), so lookups go through the search endpoint. Many
ProteinGym targets are sparse TrEMBL entries with no annotations at all
(``A4GRB6_PSEAI`` is one); that is a normal outcome, not an error, and the
generator must degrade gracefully rather than fabricate.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CACHE_DIR

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://www.uniprot.org/uniprotkb/{accession}/entry"


class UniProtError(RuntimeError):
    """Raised when UniProt cannot be reached. A missing entry is not an error."""


@dataclass
class Evidence:
    """A pointer to where an annotation came from."""

    code: str = ""  # ECO evidence code
    source: str = ""  # e.g. "PubMed"
    identifier: str = ""  # e.g. "1848640"

    @property
    def citation(self) -> str:
        if self.source and self.identifier:
            if self.source == "PubMed":
                return f"PMID {self.identifier}"
            return f"{self.source} {self.identifier}"
        return self.code or ""

    @property
    def url(self) -> str:
        if self.source == "PubMed" and self.identifier:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.identifier}/"
        return ""


@dataclass
class Annotation:
    """One annotation and the evidence for *that* annotation.

    Evidence is per-annotation, not per-record. UniProt attributes the pH
    optimum and the kinetic constants to different publications, and a
    protocol that pooled them would cite the wrong paper beside a number —
    precisely the failure this package exists to prevent.
    """

    text: str
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def citation_suffix(self) -> str:
        cites = [e.citation for e in self.evidence if e.citation]
        return " (" + ", ".join(dict.fromkeys(cites[:3])) + ")" if cites else ""

    def to_dict(self) -> dict:
        return {"text": self.text, "evidence": [e.__dict__ for e in self.evidence]}

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(text=d["text"], evidence=[Evidence(**e) for e in d.get("evidence", [])])


@dataclass
class UniProtRecord:
    """The subset of a UniProt entry a bench protocol actually needs."""

    entry_name: str
    accession: str = ""
    protein_name: str = ""
    organism: str = ""
    sequence_length: int = 0
    #: Free-text optima, as UniProt states them ("Optimum pH is 7.5.").
    ph_dependence: list[Annotation] = field(default_factory=list)
    temperature_dependence: list[Annotation] = field(default_factory=list)
    kinetics: list[Annotation] = field(default_factory=list)
    cofactors: list[Annotation] = field(default_factory=list)
    catalytic_activity: list[str] = field(default_factory=list)
    subcellular_location: list[str] = field(default_factory=list)
    function: list[str] = field(default_factory=list)
    #: PDB ids, so a protocol can point at a real structure.
    pdb_ids: list[str] = field(default_factory=list)
    ec_number: str = ""
    #: Evidence attached to the biophysical annotations, where UniProt gives it.
    evidence: list[Evidence] = field(default_factory=list)
    #: False when the entry exists but carries no usable annotation.
    found: bool = False

    @property
    def url(self) -> str:
        return UNIPROT_ENTRY_URL.format(accession=self.accession) if self.accession else ""

    @property
    def has_conditions(self) -> bool:
        """True when UniProt supplied any experimentally grounded condition."""
        return bool(self.ph_dependence or self.temperature_dependence or self.kinetics)

    def to_dict(self) -> dict:
        d = {
            "entry_name": self.entry_name,
            "accession": self.accession,
            "protein_name": self.protein_name,
            "organism": self.organism,
            "sequence_length": self.sequence_length,
            "ph_dependence": [a.to_dict() for a in self.ph_dependence],
            "temperature_dependence": [a.to_dict() for a in self.temperature_dependence],
            "kinetics": [a.to_dict() for a in self.kinetics],
            "cofactors": [a.to_dict() for a in self.cofactors],
            "catalytic_activity": self.catalytic_activity,
            "subcellular_location": self.subcellular_location,
            "function": self.function,
            "pdb_ids": self.pdb_ids,
            "ec_number": self.ec_number,
            "evidence": [e.__dict__ for e in self.evidence],
            "found": self.found,
            "url": self.url,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "UniProtRecord":
        d = dict(d)
        d.pop("url", None)
        evidence = [Evidence(**e) for e in d.pop("evidence", [])]
        annotations = {
            f: [Annotation.from_dict(a) for a in d.pop(f, [])]
            for f in ("ph_dependence", "temperature_dependence", "kinetics", "cofactors")
        }
        return cls(evidence=evidence, **annotations, **d)


def _texts(block: dict, key: str) -> list[Annotation]:
    """Pull free-text values out of a dependence block, each with its evidence."""
    out = []
    for item in block.get(key, {}).get("texts", []):
        value = item.get("value", "").strip()
        if value:
            out.append(Annotation(text=value, evidence=_evidence_from(item.get("evidences", []))))
    return out


def _evidence_from(items: list[dict]) -> list[Evidence]:
    out = []
    for e in items or []:
        out.append(
            Evidence(
                code=e.get("evidenceCode", ""),
                source=e.get("source", ""),
                identifier=str(e.get("id", "")),
            )
        )
    return out


def _parse_entry(entry: dict, entry_name: str) -> UniProtRecord:
    desc = entry.get("proteinDescription", {})
    rec_name = desc.get("recommendedName", {})
    protein_name = rec_name.get("fullName", {}).get("value", "")
    if not protein_name:
        submitted = desc.get("submissionNames", [])
        if submitted:
            protein_name = submitted[0].get("fullName", {}).get("value", "")

    record = UniProtRecord(
        entry_name=entry_name,
        accession=entry.get("primaryAccession", ""),
        protein_name=protein_name,
        organism=entry.get("organism", {}).get("scientificName", ""),
        sequence_length=int(entry.get("sequence", {}).get("length", 0) or 0),
        ec_number=rec_name.get("ecNumbers", [{}])[0].get("value", "")
        if rec_name.get("ecNumbers")
        else "",
        found=True,
    )

    for comment in entry.get("comments", []):
        kind = comment.get("commentType", "")
        if kind == "BIOPHYSICOCHEMICAL PROPERTIES":
            record.ph_dependence += _texts(comment, "phDependence")
            record.temperature_dependence += _texts(comment, "temperatureDependence")
            kinetics = comment.get("kineticParameters", {})
            for velocity in kinetics.get("maximumVelocities", []):
                enzyme = velocity.get("enzyme", "")
                ev = _evidence_from(velocity.get("evidences", []))
                record.kinetics.append(
                    Annotation(
                        text=f"Vmax {velocity.get('velocity')} "
                        f"{velocity.get('unit', '')}"
                        + (f" — {enzyme}" if enzyme else ""),
                        evidence=ev,
                    )
                )
                record.evidence += ev
            for km in kinetics.get("michaelisConstants", []):
                substrate = km.get("substrate", "")
                # UniProt names this field "constant", not "value"; reading the
                # wrong key renders every Km as "None".
                value = km.get("constant", km.get("value"))
                if value is None:
                    continue
                ev = _evidence_from(km.get("evidences", []))
                record.kinetics.append(
                    Annotation(
                        text=f"Km {value} {km.get('unit', '')}"
                        + (f" for {substrate}" if substrate else ""),
                        evidence=ev,
                    )
                )
                record.evidence += ev
            for ann in record.ph_dependence + record.temperature_dependence:
                record.evidence += ann.evidence
        elif kind == "COFACTOR":
            for cof in comment.get("cofactors", []):
                name = cof.get("name", "")
                if name:
                    ev = _evidence_from(cof.get("evidences", []))
                    record.cofactors.append(Annotation(text=name, evidence=ev))
                    record.evidence += ev
        elif kind == "CATALYTIC ACTIVITY":
            name = comment.get("reaction", {}).get("name", "")
            if name:
                record.catalytic_activity.append(name)
        elif kind == "SUBCELLULAR LOCATION":
            for loc in comment.get("subcellularLocations", []):
                value = loc.get("location", {}).get("value", "")
                if value:
                    record.subcellular_location.append(value)
        elif kind == "FUNCTION":
            for text in comment.get("texts", []):
                value = text.get("value", "")
                if value:
                    record.function.append(value)

    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "PDB":
            record.pdb_ids.append(xref.get("id", ""))

    # De-duplicate evidence while keeping order.
    seen: set[tuple[str, str, str]] = set()
    unique: list[Evidence] = []
    for e in record.evidence:
        key = (e.code, e.source, e.identifier)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    record.evidence = unique
    record.pdb_ids = record.pdb_ids[:8]
    return record


def _cache_path(entry_name: str) -> Path:
    safe = entry_name.replace("/", "_")
    return CACHE_DIR / "uniprot" / f"{safe}.json"


def fetch_uniprot(
    entry_name: str,
    use_cache: bool = True,
    timeout: float = 30.0,
) -> UniProtRecord:
    """Look up one UniProt entry by entry name (``TPMT_HUMAN``) or accession.

    Returns a record with ``found=False`` when the entry does not exist, which
    is a normal outcome for ProteinGym's sparser targets. Raises
    :class:`UniProtError` only when UniProt itself could not be reached, so a
    network failure is never silently rendered as "no annotations".
    """
    path = _cache_path(entry_name)
    if use_cache and path.exists():
        return UniProtRecord.from_dict(json.loads(path.read_text()))

    query = urllib.parse.quote(f"id:{entry_name}")
    url = f"{_SEARCH_URL}?query={query}&format=json&size=1"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise UniProtError(f"UniProt lookup failed for {entry_name}: {exc}") from exc

    results = payload.get("results", [])
    if not results:
        record = UniProtRecord(entry_name=entry_name, found=False)
    else:
        record = _parse_entry(results[0], entry_name)

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), indent=2))
    return record
