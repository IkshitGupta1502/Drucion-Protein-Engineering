"""Tests for bench protocol generation.

The rule under test throughout: a protocol may not present an invented number
as a measurement. Everything else here supports that one guarantee.

No test makes a network call. UniProt and the language model are both stubbed.
"""

from __future__ import annotations

import json

import pytest

from drucion.protocol import generate, render, uniprot
from drucion.protocol.model import (
    Fact,
    Protocol,
    ProtocolError,
    Section,
)
from drucion.protocol.render import render_markdown
from drucion.protocol.uniprot import Evidence, UniProtError, UniProtRecord


# ---------------------------------------------------------------------------
# The provenance rule
# ---------------------------------------------------------------------------


def test_estimate_without_verification_is_refused():
    """The core guarantee: an unverifiable estimate cannot be constructed."""
    with pytest.raises(ProtocolError, match="no verification step"):
        Fact(
            label="Thermal stability",
            value="60-70 C",
            tier="estimated",
            source="metallo-beta-lactamases typically unfold in this range",
        )


def test_estimate_without_reasoning_is_refused():
    with pytest.raises(ProtocolError, match="no reasoning"):
        Fact(
            label="Thermal stability",
            value="60-70 C",
            tier="estimated",
            verify="thermal shift assay",
        )


def test_sourced_fact_without_citation_is_refused():
    """A fact with no source is an estimate; it may not pose as sourced."""
    with pytest.raises(ProtocolError, match="no citation"):
        Fact(label="Optimum pH", value="7.5", tier="sourced")


def test_unknown_tier_is_refused():
    with pytest.raises(ProtocolError, match="Unknown provenance tier"):
        Fact(label="x", value="y", tier="probably-fine")


def test_standard_fact_needs_nothing_extra():
    f = Fact(label="Buffer", value="50 mM Tris-HCl")
    assert f.tier == "standard"
    assert not f.is_estimate


def test_wellformed_estimate_is_accepted():
    f = Fact(
        label="Thermal stability",
        value="60-70 C",
        tier="estimated",
        source="metallo-beta-lactamases typically unfold in this range",
        verify="differential scanning fluorimetry on purified protein",
    )
    assert f.is_estimate


# ---------------------------------------------------------------------------
# Parsing the assay's own published conditions
# ---------------------------------------------------------------------------


def test_assay_conditions_parsed_as_sourced():
    """ProteinGym's free-text assay field carries the real numbers."""
    text = (
        "drug resistance (128/16/2.0 ug/mL ampicillin, 4.0/0.5 ug/mL "
        "cefotaxime, 0.031 ug/mL meropenem @ 25C, 37C)"
    )
    facts = generate._parse_assay_conditions(text)
    labels = {f.label: f.value for f in facts}
    assert "Incubation temperature" in labels
    assert "25" in labels["Incubation temperature"]
    assert "37" in labels["Incubation temperature"]
    assert labels["Ampicillin concentration"] == "128, 16, 2.0 ug/mL"
    assert labels["Meropenem concentration"] == "0.031 ug/mL"
    assert all(f.tier == "sourced" for f in facts)
    assert all(f.source for f in facts)


def test_empty_assay_text_yields_nothing():
    assert generate._parse_assay_conditions("") == []
    assert generate._parse_assay_conditions("Protein abundance (FACS)") == []


# ---------------------------------------------------------------------------
# UniProt parsing
# ---------------------------------------------------------------------------


def _uniprot_payload() -> dict:
    return {
        "primaryAccession": "P51580",
        "proteinDescription": {
            "recommendedName": {
                "fullName": {"value": "Thiopurine S-methyltransferase"},
                "ecNumbers": [{"value": "2.1.1.67"}],
            }
        },
        "organism": {"scientificName": "Homo sapiens"},
        "sequence": {"length": 245},
        "comments": [
            {
                "commentType": "BIOPHYSICOCHEMICAL PROPERTIES",
                "phDependence": {
                    "texts": [
                        {
                            "value": "Optimum pH is 7.5.",
                            "evidences": [
                                {
                                    "evidenceCode": "ECO:0000269",
                                    "source": "PubMed",
                                    "id": "1848640",
                                }
                            ],
                        }
                    ]
                },
            },
            {
                "commentType": "SUBCELLULAR LOCATION",
                "subcellularLocations": [{"location": {"value": "Cytoplasm"}}],
            },
            {"commentType": "FUNCTION", "texts": [{"value": "Catalyzes methylation."}]},
        ],
        "uniProtKBCrossReferences": [{"database": "PDB", "id": "2BZG"}],
    }


def test_uniprot_parse_extracts_conditions_and_evidence():
    rec = uniprot._parse_entry(_uniprot_payload(), "TPMT_HUMAN")
    assert rec.accession == "P51580"
    assert [a.text for a in rec.ph_dependence] == ["Optimum pH is 7.5."]
    assert rec.has_conditions
    assert rec.evidence[0].citation == "PMID 1848640"
    assert "pubmed" in rec.evidence[0].url
    assert rec.pdb_ids == ["2BZG"]
    assert rec.ec_number == "2.1.1.67"


def test_kinetics_uses_the_constant_field():
    """UniProt names the Km field 'constant'; the wrong key renders 'None'."""
    payload = _uniprot_payload()
    payload["comments"].append(
        {
            "commentType": "BIOPHYSICOCHEMICAL PROPERTIES",
            "kineticParameters": {
                "michaelisConstants": [
                    {
                        "constant": 18.5,
                        "unit": "uM",
                        "substrate": "S-adenosyl-L-methionine",
                        "evidences": [
                            {
                                "evidenceCode": "ECO:0000269",
                                "source": "PubMed",
                                "id": "18484748",
                            }
                        ],
                    }
                ]
            },
        }
    )
    rec = uniprot._parse_entry(payload, "TPMT_HUMAN")
    assert any("18.5 uM" in a.text for a in rec.kinetics)
    assert not any("None" in a.text for a in rec.kinetics)


def test_each_fact_cites_its_own_annotations_paper():
    """Evidence is per-annotation: pooling it misattributes the citation.

    UniProt credits the pH optimum and the kinetic constants to different
    publications. A protocol that printed both PMIDs beside each number would
    be citing a paper that does not support it.
    """
    payload = _uniprot_payload()
    payload["comments"].append(
        {
            "commentType": "BIOPHYSICOCHEMICAL PROPERTIES",
            "kineticParameters": {
                "michaelisConstants": [
                    {
                        "constant": 18.5,
                        "unit": "uM",
                        "substrate": "SAM",
                        "evidences": [
                            {
                                "evidenceCode": "ECO:0000269",
                                "source": "PubMed",
                                "id": "18484748",
                            }
                        ],
                    }
                ]
            },
        }
    )
    rec = uniprot._parse_entry(payload, "TPMT_HUMAN")
    facts = generate._uniprot_facts(rec)
    ph = next(f for f in facts if f.label == "Optimum pH")
    km = next(f for f in facts if f.label == "Kinetics")
    assert "PMID 1848640" in ph.source
    assert "PMID 18484748" not in ph.source
    assert "PMID 18484748" in km.source
    assert "PMID 1848640" not in km.source


def test_uniprot_facts_carry_citation():
    rec = uniprot._parse_entry(_uniprot_payload(), "TPMT_HUMAN")
    facts = generate._uniprot_facts(rec)
    ph = next(f for f in facts if f.label == "Optimum pH")
    assert ph.tier == "sourced"
    assert "P51580" in ph.source
    assert "PMID 1848640" in ph.source


def test_sparse_entry_produces_no_invented_facts():
    """A TrEMBL entry with no annotations must yield nothing, not a guess."""
    rec = uniprot._parse_entry({"primaryAccession": "A4GRB6"}, "A4GRB6_PSEAI")
    assert rec.found is True
    assert not rec.has_conditions
    assert generate._uniprot_facts(rec) == []


def test_missing_entry_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(uniprot, "CACHE_DIR", tmp_path)

    class _Resp:
        def read(self):
            return json.dumps({"results": []}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(uniprot.urllib.request, "urlopen", lambda *a, **k: _Resp())
    rec = uniprot.fetch_uniprot("NOPE_NOPE")
    assert rec.found is False


def test_network_failure_raises_rather_than_claiming_no_data(monkeypatch, tmp_path):
    """A failed lookup must not be indistinguishable from 'no annotations'."""
    monkeypatch.setattr(uniprot, "CACHE_DIR", tmp_path)

    def boom(*a, **k):
        raise uniprot.urllib.error.URLError("offline")

    monkeypatch.setattr(uniprot.urllib.request, "urlopen", boom)
    with pytest.raises(UniProtError):
        uniprot.fetch_uniprot("TPMT_HUMAN")


def test_uniprot_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(uniprot, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    class _Resp:
        def read(self):
            calls["n"] += 1
            return json.dumps({"results": [_uniprot_payload()]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(uniprot.urllib.request, "urlopen", lambda *a, **k: _Resp())
    first = uniprot.fetch_uniprot("TPMT_HUMAN")
    second = uniprot.fetch_uniprot("TPMT_HUMAN")
    assert calls["n"] == 1
    assert second.ph_dependence == first.ph_dependence


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_uniprot(monkeypatch):
    rec = uniprot._parse_entry(_uniprot_payload(), "TPMT_HUMAN")
    monkeypatch.setattr(generate, "fetch_uniprot", lambda *a, **k: rec)
    return rec


def test_generate_without_llm_has_no_estimates(synthetic_loaded, stub_uniprot):
    p = generate.generate_protocol(synthetic_loaded.metadata, use_llm=False)
    assert p.estimates == []
    assert p.tier_counts()["estimated"] == 0
    assert p.generated_by == "records only"


def test_generate_includes_biosafety_section(synthetic_loaded, stub_uniprot):
    p = generate.generate_protocol(synthetic_loaded.metadata, use_llm=False)
    bio = p.section("Biosafety and handling")
    assert bio is not None
    # Containment is the institution's call; the protocol must not assign one.
    assert "biosafety officer" in bio.body.lower()
    assert "bsl-" not in bio.body.lower()


def test_generate_includes_worklist(synthetic_loaded, stub_uniprot):
    variants = list(synthetic_loaded.pool.mutants[:5])
    p = generate.generate_protocol(
        synthetic_loaded.metadata, worklist=variants, use_llm=False
    )
    samples = p.section("Samples this round")
    assert samples is not None
    assert samples.steps == variants


def test_uniprot_failure_degrades_without_inventing(synthetic_loaded, monkeypatch):
    def boom(*a, **k):
        raise UniProtError("offline")

    monkeypatch.setattr(generate, "fetch_uniprot", boom)
    p = generate.generate_protocol(synthetic_loaded.metadata, use_llm=False)
    assert p.estimates == []
    assert p.section("Conditions") is not None


def test_llm_gaps_become_flagged_estimates(synthetic_loaded, stub_uniprot, monkeypatch):
    payload = json.dumps(
        {
            "estimates": [
                {
                    "label": "Thermal stability",
                    "value": "55-65 C",
                    "reasoning": "typical for this fold",
                    "verify": "differential scanning fluorimetry",
                }
            ],
            "handling_notes": ["Keep on ice"],
            "pitfalls": ["Aggregation above 1 mg/mL"],
        }
    )
    monkeypatch.setattr(generate, "call_gemini", lambda *a, **k: payload, raising=False)
    monkeypatch.setattr(
        "drucion.embeddings.agent_prior.call_gemini", lambda *a, **k: payload
    )
    p = generate.generate_protocol(synthetic_loaded.metadata, use_llm=True)
    assert len(p.estimates) == 1
    est = p.estimates[0]
    assert est.tier == "estimated"
    assert est.verify == "differential scanning fluorimetry"
    assert "gaps only" in p.generated_by


def test_incomplete_llm_estimate_is_dropped(synthetic_loaded, stub_uniprot, monkeypatch):
    """An estimate with no verification step is discarded, never shown."""
    payload = json.dumps(
        {
            "estimates": [
                {"label": "Thermal stability", "value": "55-65 C", "reasoning": "hunch"},
                {
                    "label": "Optimum ionic strength",
                    "value": "100-200 mM",
                    "reasoning": "typical",
                    "verify": "activity assay across a salt series",
                },
            ]
        }
    )
    monkeypatch.setattr(
        "drucion.embeddings.agent_prior.call_gemini", lambda *a, **k: payload
    )
    p = generate.generate_protocol(synthetic_loaded.metadata, use_llm=True)
    labels = [e.label for e in p.estimates]
    assert labels == ["Optimum ionic strength"]


def test_llm_failure_leaves_protocol_usable(synthetic_loaded, stub_uniprot, monkeypatch):
    from drucion.embeddings.agent_prior import AgentPriorError

    def boom(*a, **k):
        raise AgentPriorError("503")

    monkeypatch.setattr("drucion.embeddings.agent_prior.call_gemini", boom)
    p = generate.generate_protocol(synthetic_loaded.metadata, use_llm=True)
    assert p.estimates == []
    assert p.generated_by == "records only"
    assert p.section("Procedure") is not None


def test_gap_prompt_does_not_restate_known_facts(synthetic_loaded, stub_uniprot):
    known = ["Optimum pH: Optimum pH is 7.5", "Incubation temperature: 37 C"]
    prompt = generate._gap_prompt(synthetic_loaded.metadata, stub_uniprot, known)
    assert "NOT restate them" in prompt
    assert "Optimum pH is 7.5" in prompt
    assert "containment level" in prompt.lower()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _protocol_with_estimate() -> Protocol:
    return Protocol(
        assay_id="X",
        title="Bench protocol",
        sections=[
            Section(
                title="Conditions",
                facts=[
                    Fact(
                        label="Optimum pH",
                        value="7.5",
                        tier="sourced",
                        source="UniProt P51580",
                        source_url="https://example.org",
                    ),
                    Fact(
                        label="Thermal stability",
                        value="55-65 C",
                        tier="estimated",
                        source="typical for this fold",
                        verify="differential scanning fluorimetry",
                    ),
                ],
            ),
            Section(title="Procedure", steps=["Do the thing", "Do the next thing"]),
        ],
    )


def test_render_flags_estimates_up_front():
    md = render_markdown(_protocol_with_estimate())
    before = md.split("## Conditions")[0]
    assert "1 estimates" in before
    assert "differential scanning fluorimetry" in before


def test_render_never_prints_a_bare_estimate():
    md = render_markdown(_protocol_with_estimate())
    line = next(l for l in md.splitlines() if "55-65 C" in l)
    block = md[md.index(line) : md.index(line) + 400]
    assert "Estimate" in block
    assert "not measured" in block
    assert "Confirm by:" in block


def test_render_cites_sourced_values():
    md = render_markdown(_protocol_with_estimate())
    assert "[UniProt P51580](https://example.org)" in md


def test_render_counts_standard_steps_not_just_facts():
    """The Procedure section is all steps; reporting 0 standard would mislead."""
    md = render_markdown(_protocol_with_estimate())
    assert "**2** standard-technique steps" in md


def test_render_says_so_when_nothing_is_estimated():
    p = Protocol(
        assay_id="X",
        title="T",
        sections=[Section(title="Conditions", facts=[Fact(label="Buffer", value="Tris")])],
    )
    md = render_markdown(p)
    assert "No estimates" in md
