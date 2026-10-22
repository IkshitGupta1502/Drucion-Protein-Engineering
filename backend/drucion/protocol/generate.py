"""Assemble a bench protocol for one assay, source-first.

Order of precedence, and it matters
-----------------------------------
1. **The assay's own published conditions.** ProteinGym's ``selection_assay``
   field often states the real numbers used — antibiotic concentrations,
   incubation temperatures. Those are the conditions the fitness values in the
   oracle were actually measured under, so they are the right ones to
   reproduce, and they are sourced to the originating paper.
2. **UniProt annotations.** pH and temperature optima, cofactors, catalytic
   activity, localisation, PDB entries, usually with a PubMed citation.
3. **Standard technique.** Protein-independent laboratory practice, true
   regardless of the target.
4. **A language model, for gaps only.** Reached last, asked only for what
   steps 1-3 did not supply, and everything it returns is tier ``estimated``
   with a mandatory verification step.

The model is never asked to restate something a record already answered. That
is deliberate: an LLM asked "what is this protein's optimum pH" will answer
whether or not it knows, and the answer is indistinguishable on the page from
UniProt's. Asking it only for genuine gaps keeps the two apart.

Biosafety
---------
Containment level is an institutional determination that depends on strain,
host and local regulation, so this module never asserts one. It states what is
typically involved, names the hazards specific to the target, and directs the
reader to their biosafety officer. A protocol that quietly omitted the
question would be worse than one that asks it.
"""

from __future__ import annotations

import json
import logging
import re

from ..data.proteingym import AssayMetadata
from .model import Fact, Protocol, Section
from .uniprot import UniProtError, UniProtRecord, fetch_uniprot

logger = logging.getLogger(__name__)

#: Hazard notes keyed on what the assay measures. Deliberately about the class
#: of work rather than a containment level, which is not ours to assign.
_HAZARD_NOTES: dict[str, str] = {
    "Antibiotics resistance": (
        "This assay propagates an antibiotic-resistance gene and uses "
        "clinically relevant antibiotics. Resistance-gene work is regulated at "
        "most institutions and the waste stream is controlled."
    ),
    "Growth": (
        "This assay propagates live bacterial culture under antibiotic "
        "selection. Culture and selection waste are both controlled streams."
    ),
    "Protein stability": (
        "This assay uses mammalian cell culture and flow cytometry. Cell-line "
        "provenance and sorter decontamination are the usual concerns."
    ),
}

_ASSAY_FAMILY: dict[str, str] = {
    "Antibiotics resistance": "antibiotic_mic",
    "Growth": "growth_selection",
    "Protein stability": "abundance_facs",
}


def _parse_assay_conditions(selection_assay: str) -> list[Fact]:
    """Pull concrete conditions out of ProteinGym's free-text assay field.

    The field frequently carries the real numbers, e.g.::

        drug resistance (128/16/2.0 ug/mL ampicillin, ... @ 25C, 37C)

    Those are sourced facts: they describe how the measurements in the oracle
    were produced. Parsing is deliberately conservative — anything not clearly
    a concentration or a temperature is left in the verbatim description rather
    than guessed at.
    """
    facts: list[Fact] = []
    text = selection_assay.strip()
    if not text:
        return facts

    temps = re.findall(r"(\d+(?:\.\d+)?)\s*(?:deg\s*)?C\b", text)
    if temps:
        unique = sorted({t for t in temps}, key=float)
        facts.append(
            Fact(
                label="Incubation temperature",
                value=", ".join(f"{t} °C" for t in unique),
                tier="sourced",
                source="Assay conditions as recorded in ProteinGym",
            )
        )

    # Concentration runs like "128/16/2.0 ug/mL ampicillin".
    for match in re.finditer(
        r"([\d.]+(?:/[\d.]+)*)\s*(ug/mL|µg/mL|mg/mL|mM|nM|uM)\s+([A-Za-z][A-Za-z\- ]*)",
        text,
    ):
        amounts, unit, agent = match.groups()
        agent = agent.strip().rstrip(",.").strip()
        if not agent:
            continue
        facts.append(
            Fact(
                label=f"{agent.capitalize()} concentration",
                value=f"{amounts.replace('/', ', ')} {unit}",
                tier="sourced",
                source="Assay conditions as recorded in ProteinGym",
            )
        )
    return facts


def _uniprot_facts(record: UniProtRecord) -> list[Fact]:
    """Turn UniProt annotations into sourced facts with citations."""
    facts: list[Fact] = []
    base = "UniProt " + (record.accession or record.entry_name)

    def _cite(annotation) -> str:
        """Cite the paper backing *this* annotation, not the record's pool.

        UniProt attributes the pH optimum and the kinetic constants to
        different publications. Pooling them would print the wrong PMID beside
        a number, which is a provenance failure dressed up as a citation.
        """
        return base + annotation.citation_suffix

    for ann in record.ph_dependence:
        facts.append(
            Fact(
                label="Optimum pH",
                value=ann.text.rstrip("."),
                tier="sourced",
                source=_cite(ann),
                source_url=record.url,
            )
        )
    for ann in record.temperature_dependence:
        facts.append(
            Fact(
                label="Optimum temperature",
                value=ann.text.rstrip("."),
                tier="sourced",
                source=_cite(ann),
                source_url=record.url,
            )
        )
    for ann in record.kinetics[:4]:
        facts.append(
            Fact(
                label="Kinetics",
                value=ann.text,
                tier="sourced",
                source=_cite(ann),
                source_url=record.url,
            )
        )
    for ann in record.cofactors:
        facts.append(
            Fact(
                label="Required cofactor",
                value=ann.text,
                tier="sourced",
                source=_cite(ann),
                source_url=record.url,
            )
        )
    if record.subcellular_location:
        facts.append(
            Fact(
                label="Subcellular location",
                value=", ".join(dict.fromkeys(record.subcellular_location)),
                tier="sourced",
                # Location carries no per-annotation evidence in UniProt, so
                # the record itself is the citation.
                source=base,
                source_url=record.url,
            )
        )
    return facts


_STANDARD_STEPS: dict[str, list[str]] = {
    "antibiotic_mic": [
        "Transform each variant plasmid into the expression strain and plate "
        "on non-selective medium; pick single colonies.",
        "Grow overnight cultures, then dilute to a standardised inoculum "
        "(OD600 is the usual handle) so every variant starts equal.",
        "Prepare a two-fold antibiotic dilution series spanning the "
        "concentrations above, in a 96-well plate.",
        "Inoculate, seal, and incubate at the temperature given above.",
        "Read growth after the fixed endpoint and record the lowest "
        "concentration with no visible growth as that variant's MIC.",
        "Record every reading in the logbook, including wells that failed.",
    ],
    "growth_selection": [
        "Transform the variant library or individual variants into the "
        "selection strain.",
        "Grow to exponential phase in non-selective medium.",
        "Split into the antibiotic concentrations given above, keeping one "
        "non-selective arm as the reference.",
        "Track optical density over the growth window and fit the doubling "
        "time for each variant.",
        "Normalise each variant's rate to the wild-type control run on the "
        "same plate.",
        "Record raw curves and fitted rates in the logbook.",
    ],
    "abundance_facs": [
        "Produce a stable line per variant with the fusion construct and a "
        "separately expressed normalisation marker.",
        "Culture to the density your sorter protocol expects; include "
        "wild-type and a known-destabilised variant on every run.",
        "Sort on the ratio of fusion signal to normalisation marker, not on "
        "raw fluorescence, so expression differences divide out.",
        "Collect the sorted bins and quantify variant frequency in each.",
        "Convert bin frequencies to an abundance score per variant.",
        "Record gating, bin boundaries and run date in the logbook — these "
        "drift between sessions and explain more disagreements than anything "
        "else.",
    ],
}

_STANDARD_CONTROLS: list[str] = [
    "Wild-type, on every plate or run. Drucion's scores are relative to it, "
    "so a run without it cannot be placed on the same scale as the others.",
    "A known-null or known-destabilised variant, to confirm the assay can "
    "still resolve a bad variant on this particular day.",
    "A no-template or untransformed well, to catch contamination.",
    "Technical replicates of at least two variants per plate, to estimate "
    "within-run noise before trusting between-variant differences.",
]


def _gap_prompt(
    metadata: AssayMetadata, record: UniProtRecord, known: list[str]
) -> str:
    """Ask only for what the records did not supply."""
    known_text = "\n".join(f"- {k}" for k in known) or "- (nothing)"
    return f"""\
You are helping prepare a bench protocol for a protein assay. Several
conditions are already known from published records and are listed below. Do
NOT restate them.

Protein: {metadata.molecule_name or record.protein_name or 'unknown'}
Organism: {metadata.source_organism or record.organism or 'unknown'}
Assay: {metadata.selection_assay or 'unknown'}
Assay type: {metadata.selection_type or 'unknown'}
Sequence length: {metadata.seq_len}

Already known from records:
{known_text}

For anything still missing that a bench scientist would need, give your best
inference. These are explicitly understood to be estimates, not measurements,
and each will be shown to the reader labelled as an estimate with your
verification step beside it.

Return ONLY a JSON object:

{{
  "estimates": [
    {{"label": "<what this is, e.g. 'Thermal stability'>",
      "value": "<a RANGE, not a point value, with units>",
      "reasoning": "<why you infer this, one sentence>",
      "verify": "<a specific experiment that would confirm it>"}}
  ],
  "handling_notes": ["<practical warnings specific to this protein>"],
  "pitfalls": ["<what most often goes wrong in this kind of assay>"]
}}

Rules:
- At most 6 estimates. Omit anything you cannot reason about — an empty list
  is a correct answer.
- "value" must be a range wherever the underlying quantity is uncertain.
- "verify" must name a real experiment, never "check the literature".
- Do not state a biosafety containment level; that is set by the institution.
"""


def _llm_gap_facts(
    metadata: AssayMetadata,
    record: UniProtRecord,
    known: list[str],
    model: str,
    api_key: str | None,
) -> tuple[list[Fact], list[str], list[str]]:
    """Ask the model for gaps. Returns (facts, handling notes, pitfalls)."""
    from ..embeddings.agent_prior import AgentPriorError, _extract_json, call_gemini

    prompt = _gap_prompt(metadata, record, known)
    try:
        text = call_gemini(prompt, model=model, api_key=api_key)
        obj = _extract_json(text)
    except AgentPriorError as exc:
        logger.warning("protocol: gap lookup unavailable (%s)", exc)
        return [], [], []

    facts: list[Fact] = []
    for item in obj.get("estimates", [])[:6]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = str(item.get("value", "")).strip()
        reasoning = str(item.get("reasoning", "")).strip()
        verify = str(item.get("verify", "")).strip()
        if not (label and value and reasoning and verify):
            # An estimate missing its reasoning or its check is dropped rather
            # than shown — Fact would refuse it anyway.
            logger.info("protocol: dropped incomplete estimate %r", label)
            continue
        facts.append(
            Fact(
                label=label,
                value=value,
                tier="estimated",
                source=reasoning,
                verify=verify,
            )
        )
    notes = [str(n).strip() for n in obj.get("handling_notes", [])[:6] if str(n).strip()]
    pitfalls = [str(p).strip() for p in obj.get("pitfalls", [])[:6] if str(p).strip()]
    return facts, notes, pitfalls


def generate_protocol(
    metadata: AssayMetadata,
    worklist: list[str] | None = None,
    use_llm: bool = True,
    model: str = "gemini-3.8-flash",
    api_key: str | None = None,
    use_cache: bool = True,
) -> Protocol:
    """Build a protocol for ``metadata``, optionally for a specific batch.

    ``worklist`` is the set of variants the optimizer proposed this round; when
    given, the protocol includes them as the sample list. ``use_llm=False``
    produces a protocol containing only sourced and standard material, which is
    the right choice when no estimate is wanted at all.
    """
    sections: list[Section] = []
    protein_name = metadata.molecule_name or metadata.assay_id

    # -- records first -----------------------------------------------------
    try:
        record = fetch_uniprot(metadata.uniprot_id, use_cache=use_cache)
    except UniProtError as exc:
        logger.warning("protocol: %s", exc)
        record = UniProtRecord(entry_name=metadata.uniprot_id, found=False)

    condition_facts = _parse_assay_conditions(metadata.selection_assay)
    condition_facts += _uniprot_facts(record)

    # -- overview ----------------------------------------------------------
    overview = Section(
        title="Target",
        body=(
            f"{protein_name} from {metadata.source_organism or 'unknown organism'}, "
            f"{metadata.seq_len} residues, region mutated "
            f"{metadata.region_mutated or 'not stated'}. "
            f"Fitness is measured by: {metadata.selection_assay or 'not stated'}."
        ),
    )
    if record.found and record.function:
        overview.body += f"\n\nFunction: {record.function[0]}"
    if record.pdb_ids:
        overview.body += (
            f"\n\nStructures available: {', '.join(record.pdb_ids[:5])} (PDB)."
        )
    overview.facts.append(
        Fact(
            label="Originating measurements",
            value=f"{metadata.first_author} {metadata.year}, {metadata.title}",
            tier="sourced",
            source="ProteinGym reference table",
        )
    )
    if record.found and record.url:
        overview.facts.append(
            Fact(
                label="UniProt entry",
                value=record.accession or record.entry_name,
                tier="sourced",
                source="UniProt",
                source_url=record.url,
            )
        )
    sections.append(overview)

    # -- biosafety ---------------------------------------------------------
    hazard = _HAZARD_NOTES.get(
        metadata.selection_type,
        "Confirm the hazards of this construct and host with your biosafety officer.",
    )
    sections.append(
        Section(
            title="Biosafety and handling",
            body=(
                f"{hazard}\n\n"
                "**Containment level is not stated here.** It depends on the "
                "strain, the host, the construct and your local regulation, and "
                "it is your institution's determination to make. Confirm it with "
                "your biosafety officer before starting, and follow your local "
                "waste procedure."
            ),
        )
    )

    # -- conditions --------------------------------------------------------
    conditions = Section(
        title="Conditions",
        body=(
            "Reproduce the originating assay's conditions where they are stated: "
            "the fitness values Drucion was trained on were measured under them, "
            "so results obtained under different conditions are not on the same "
            "scale."
        ),
        facts=condition_facts,
    )
    if not condition_facts:
        conditions.body += (
            "\n\nNo conditions were recoverable from the assay record or from "
            "UniProt for this target."
        )
    sections.append(conditions)

    # -- procedure ---------------------------------------------------------
    family = _ASSAY_FAMILY.get(metadata.selection_type, "")
    steps = _STANDARD_STEPS.get(family, [])
    procedure = Section(
        title="Procedure",
        body=(
            "General technique for this assay class. It is not specific to this "
            "protein; adapt it to your strain, vector and equipment."
            if steps
            else "No standard procedure is encoded for this assay type "
            f"({metadata.selection_type or 'unknown'}). Adapt the originating "
            "paper's methods section."
        ),
        steps=steps,
    )
    sections.append(procedure)

    sections.append(
        Section(
            title="Controls",
            body=(
                "Drucion compares variants to each other, so a run without "
                "controls cannot be merged with earlier rounds."
            ),
            steps=list(_STANDARD_CONTROLS),
        )
    )

    # -- gaps, last --------------------------------------------------------
    generated_by = "records only"
    if use_llm:
        known = [f"{f.label}: {f.value}" for f in condition_facts]
        gap_facts, handling, pitfalls = _llm_gap_facts(
            metadata, record, known, model=model, api_key=api_key
        )
        if gap_facts or handling or pitfalls:
            generated_by = f"records + {model} (gaps only)"
            estimates = Section(
                title="Estimates — verify before relying on these",
                body=(
                    "None of the following was found in a record. Each is an "
                    "inference, shown with the reasoning behind it and an "
                    "experiment that would confirm it. **Do not treat these as "
                    "measurements.**"
                ),
                facts=gap_facts,
            )
            if handling:
                estimates.steps = [f"Handling: {n}" for n in handling]
            if pitfalls:
                estimates.steps += [f"Common pitfall: {p}" for p in pitfalls]
            sections.append(estimates)

    # -- the batch ---------------------------------------------------------
    if worklist:
        sections.append(
            Section(
                title="Samples this round",
                body=(
                    f"{len(worklist)} variants proposed by Drucion. Measure all "
                    "of them, including any that look unpromising: the loop "
                    "learns from low values too, and silently dropping them "
                    "biases the next round."
                ),
                steps=list(worklist),
            )
        )

    sections.append(
        Section(
            title="Recording results",
            body=(
                "Enter every measurement, including failures and outliers, "
                "against its variant identifier. A missing value and a measured "
                "low value are different inputs to the next round, and only one "
                "of them is information."
            ),
        )
    )

    return Protocol(
        assay_id=metadata.assay_id,
        title=f"Bench protocol — {protein_name}",
        sections=sections,
        worklist=list(worklist or []),
        generated_by=generated_by,
    )
