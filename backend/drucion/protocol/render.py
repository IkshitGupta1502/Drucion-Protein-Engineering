"""Render a :class:`~drucion.protocol.model.Protocol` as Markdown.

The rendering is where the provenance rule becomes visible. Estimates are
never printed as bare values: each carries its reasoning and the experiment
that would confirm it, and the document opens with a count of how many
unverified inferences it contains, so a reader knows before the first section
how much of what follows is grounded.
"""

from __future__ import annotations

from .model import TIER_LABELS, TIER_NOTES, Fact, Protocol


def _fact_line(fact: Fact) -> str:
    source = fact.source
    if fact.source_url:
        source = f"[{source}]({fact.source_url})"
    if fact.tier == "estimated":
        return (
            f"- **{fact.label}:** {fact.value}  \n"
            f"  *Estimate — not measured.* Reasoning: {fact.source}  \n"
            f"  **Confirm by:** {fact.verify}"
        )
    if fact.tier == "sourced":
        return f"- **{fact.label}:** {fact.value}  \n  Source: {source}"
    return f"- **{fact.label}:** {fact.value}"


def render_markdown(protocol: Protocol) -> str:
    """Return the protocol as a Markdown document."""
    counts = protocol.tier_counts()
    out: list[str] = [f"# {protocol.title}", ""]
    out.append(f"Assay: `{protocol.assay_id}`")
    if protocol.generated_by:
        out.append(f"Compiled from: {protocol.generated_by}")
    out.append("")

    # The banner: what is grounded and what is not, before anything else.
    out.append("## Before you start")
    out.append("")
    # Steps are standard technique too, and the Procedure section is entirely
    # steps — counting only facts here would report "0 standard" under a page
    # of it.
    n_standard = counts["standard"] + sum(
        len(s.steps)
        for s in protocol.sections
        if s.title in ("Procedure", "Controls")
    )
    out.append(
        f"This protocol contains **{counts['sourced']} sourced** values "
        f"(from a cited record or publication), **{n_standard}** "
        f"standard-technique steps and statements, and "
        f"**{counts['estimated']} estimates**."
    )
    out.append("")
    if protocol.estimates:
        out.append(
            f"The {counts['estimated']} estimates are inferences, not "
            "measurements. Each is marked, carries its reasoning, and names an "
            "experiment that would confirm it. Verify them before committing "
            "material:"
        )
        out.append("")
        for fact in protocol.estimates:
            out.append(f"- {fact.label} — confirm by: {fact.verify}")
        out.append("")
    else:
        out.append(
            "No estimates: every value below comes from a record or is general "
            "technique."
        )
        out.append("")

    out.append("Provenance labels used below:")
    out.append("")
    for tier, label in TIER_LABELS.items():
        out.append(f"- **{label}** — {TIER_NOTES[tier]}")
    out.append("")

    for section in protocol.sections:
        out.append(f"## {section.title}")
        out.append("")
        if section.body:
            out.append(section.body)
            out.append("")
        if section.facts:
            for fact in section.facts:
                out.append(_fact_line(fact))
            out.append("")
        if section.steps:
            numbered = section.title in ("Procedure",)
            for i, step in enumerate(section.steps, start=1):
                out.append(f"{i}. {step}" if numbered else f"- {step}")
            out.append("")

    if protocol.notes:
        out.append("## Notes")
        out.append("")
        out.append(protocol.notes)
        out.append("")

    return "\n".join(out).rstrip() + "\n"
