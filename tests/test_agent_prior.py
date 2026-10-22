"""Tests for the LLM-proposed structural prior.

No test here makes a network call. The Gemini transport is exercised against a
stubbed response, and everything else — prompt construction, validation,
scoring, the ablation modes and the leakage boundary — is pure local logic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from drucion.config import EmbeddingConfig
from drucion.embeddings import agent_prior as ap
from drucion.embeddings.agent_prior import (
    AgentPrior,
    AgentPriorError,
    CriticalPosition,
    SubstitutionRule,
    build_prompt,
    parse_prior,
    score_variants,
)
from drucion.embeddings.features import build_features

from conftest import SYNTHETIC_WT


def _response(wild_type: str) -> str:
    """A well-formed model response whose positions match ``wild_type``."""
    return json.dumps(
        {
            "critical_positions": [
                {
                    "position": 3,
                    "wt_aa": wild_type[2],
                    "tolerance": "none",
                    "reason": "catalytic",
                },
                {
                    "position": 10,
                    "wt_aa": wild_type[9],
                    "tolerance": "partial",
                    "reason": "near active site",
                },
                {
                    "position": 20,
                    "wt_aa": wild_type[19],
                    "tolerance": "high",
                    "reason": "surface",
                },
            ],
            "substitution_rules": [
                {
                    "from_class": "hydrophobic",
                    "to_class": "proline",
                    "effect": "strongly_deleterious",
                    "reason": "breaks backbone",
                }
            ],
            "notes": "Reasoned from motif structure.",
        }
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_prompt_contains_sequence_and_metadata(synthetic_loaded):
    prompt = build_prompt(synthetic_loaded.metadata)
    assert synthetic_loaded.metadata.wild_type in prompt
    assert "Synthetic test protein" in prompt
    assert "critical_positions" in prompt


def test_blind_prompt_withholds_identity(synthetic_loaded):
    """The name-stripped ablation must actually strip the name."""
    meta = synthetic_loaded.metadata
    prompt = build_prompt(meta, blind=True)
    assert meta.wild_type in prompt  # sequence is still given
    assert meta.molecule_name not in prompt
    assert meta.source_organism not in prompt
    assert meta.first_author not in prompt
    assert meta.uniprot_id not in prompt


def test_prompt_never_contains_a_measurement(synthetic_loaded):
    """The prompt is label-free: no DMS score may appear in it, in any form."""
    prompt = build_prompt(synthetic_loaded.metadata)
    scores = synthetic_loaded.oracle.all_scores_for_evaluation()
    for value in scores[:50]:
        assert f"{value:.6f}" not in prompt
        assert repr(float(value)) not in prompt


def test_long_sequence_is_truncated(synthetic_loaded):
    prompt = build_prompt(synthetic_loaded.metadata, max_seq_chars=10)
    assert "truncated" in prompt


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def test_parse_valid_response():
    prior = parse_prior(
        _response(SYNTHETIC_WT),
        assay_id="SYNTH",
        wild_type=SYNTHETIC_WT,
        model="test",
        blind=False,
    )
    assert len(prior.critical_positions) == 3
    assert len(prior.substitution_rules) == 1
    assert prior.validation["positions_kept"] == 3
    assert prior.validation["wt_match_rate"] == 1.0


def test_parse_tolerates_code_fence():
    fenced = "```json\n" + _response(SYNTHETIC_WT) + "\n```"
    prior = parse_prior(
        fenced, assay_id="S", wild_type=SYNTHETIC_WT, model="test", blind=False
    )
    assert len(prior.critical_positions) == 3


def test_wrong_wild_type_residue_is_dropped():
    """The numbering guard: a claim whose expected residue is absent is dropped.

    This is what catches a prior written in Ambler or mature-protein numbering
    instead of ProteinGym's 1-indexing into target_seq.
    """
    wrong = "X" if SYNTHETIC_WT[2] != "X" else "Y"
    text = json.dumps(
        {
            "critical_positions": [
                {"position": 3, "wt_aa": wrong, "tolerance": "none"},
                {"position": 10, "wt_aa": SYNTHETIC_WT[9], "tolerance": "none"},
            ],
            "substitution_rules": [],
        }
    )
    prior = parse_prior(
        text, assay_id="S", wild_type=SYNTHETIC_WT, model="test", blind=False
    )
    assert prior.validation["dropped_wt_mismatch"] == 1
    assert [c.position for c in prior.critical_positions] == [10]


def test_offset_numbering_is_caught_wholesale():
    """A prior offset by a constant should lose most of its claims."""
    offset = 7
    positions = [10, 15, 20, 25]
    text = json.dumps(
        {
            "critical_positions": [
                # Residue identity taken from the *unshifted* position, which
                # is what an agent using a different convention would produce.
                {
                    "position": p,
                    "wt_aa": SYNTHETIC_WT[p - offset - 1],
                    "tolerance": "none",
                }
                for p in positions
            ],
            "substitution_rules": [
                {
                    "from_class": "any",
                    "to_class": "proline",
                    "effect": "deleterious",
                }
            ],
        }
    )
    prior = parse_prior(
        text, assay_id="S", wild_type=SYNTHETIC_WT, model="test", blind=False
    )
    assert prior.validation["wt_match_rate"] < 0.5


def test_out_of_range_position_is_dropped():
    text = json.dumps(
        {
            "critical_positions": [
                {"position": 99999, "wt_aa": "M", "tolerance": "none"},
                {"position": 1, "wt_aa": SYNTHETIC_WT[0], "tolerance": "none"},
            ],
            "substitution_rules": [],
        }
    )
    prior = parse_prior(
        text, assay_id="S", wild_type=SYNTHETIC_WT, model="test", blind=False
    )
    assert prior.validation["dropped_out_of_range"] == 1
    assert len(prior.critical_positions) == 1


def test_bad_tolerance_and_rule_labels_dropped():
    text = json.dumps(
        {
            "critical_positions": [
                {"position": 1, "wt_aa": SYNTHETIC_WT[0], "tolerance": "mostly"},
                {"position": 2, "wt_aa": SYNTHETIC_WT[1], "tolerance": "none"},
            ],
            "substitution_rules": [
                {"from_class": "wizard", "to_class": "polar", "effect": "neutral"},
                {"from_class": "polar", "to_class": "polar", "effect": "chaotic"},
                {"from_class": "polar", "to_class": "charged", "effect": "neutral"},
            ],
        }
    )
    prior = parse_prior(
        text, assay_id="S", wild_type=SYNTHETIC_WT, model="test", blind=False
    )
    assert prior.validation["dropped_bad_tolerance"] == 1
    assert prior.validation["dropped_bad_rule"] == 2
    assert prior.validation["rules_kept"] == 1


def test_duplicate_positions_deduplicated():
    text = json.dumps(
        {
            "critical_positions": [
                {"position": 5, "wt_aa": SYNTHETIC_WT[4], "tolerance": "none"},
                {"position": 5, "wt_aa": SYNTHETIC_WT[4], "tolerance": "high"},
            ],
            "substitution_rules": [],
        }
    )
    prior = parse_prior(
        text, assay_id="S", wild_type=SYNTHETIC_WT, model="test", blind=False
    )
    assert prior.validation["dropped_duplicate"] == 1
    assert len(prior.critical_positions) == 1


def test_empty_prior_raises():
    text = json.dumps({"critical_positions": [], "substitution_rules": []})
    with pytest.raises(AgentPriorError, match="no usable claims"):
        parse_prior(
            text, assay_id="S", wild_type=SYNTHETIC_WT, model="test", blind=False
        )


def test_non_json_response_raises():
    with pytest.raises(AgentPriorError):
        parse_prior(
            "I'm afraid I can't help with that.",
            assay_id="S",
            wild_type=SYNTHETIC_WT,
            model="test",
            blind=False,
        )


def test_missing_api_key_raises(monkeypatch, tmp_path):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ap, "ENV_FILE", tmp_path / "absent.env")
    with pytest.raises(AgentPriorError, match="No Gemini API key"):
        ap._resolve_api_key()


# ---------------------------------------------------------------------------
# Key resolution: explicit argument, environment, then .env file
# ---------------------------------------------------------------------------


def test_key_read_from_env_file(monkeypatch, tmp_path):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nGEMINI_API_KEY=from-file\n")
    monkeypatch.setattr(ap, "ENV_FILE", env)
    assert ap._resolve_api_key() == "from-file"


def test_environment_beats_env_file(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-file\n")
    monkeypatch.setattr(ap, "ENV_FILE", env)
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert ap._resolve_api_key() == "from-env"


def test_explicit_argument_beats_everything(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-file\n")
    monkeypatch.setattr(ap, "ENV_FILE", env)
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert ap._resolve_api_key("explicit") == "explicit"


@pytest.mark.parametrize(
    "line",
    [
        'GEMINI_API_KEY="quoted"',
        "GEMINI_API_KEY='quoted'",
        "export GEMINI_API_KEY=quoted",
        "  GEMINI_API_KEY = quoted  ",
    ],
)
def test_env_file_tolerates_common_paste_shapes(monkeypatch, tmp_path, line):
    """People paste shell-style lines; accept them rather than failing oddly."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(line + "\n")
    monkeypatch.setattr(ap, "ENV_FILE", env)
    assert ap._resolve_api_key() == "quoted"


def test_env_file_placeholder_is_not_silently_accepted(monkeypatch, tmp_path):
    """The .env.example placeholder must not be mistaken for a real key.

    It is accepted as a string here — the API will reject it — but the point
    of the test is that an *empty* value never counts as a key.
    """
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=\n")
    monkeypatch.setattr(ap, "ENV_FILE", env)
    with pytest.raises(AgentPriorError, match="No Gemini API key"):
        ap._resolve_api_key()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _prior_for(pool) -> AgentPrior:
    return parse_prior(
        _response(pool.wild_type),
        assay_id="S",
        wild_type=pool.wild_type,
        model="test",
        blind=False,
    )


def test_score_shape_and_finiteness(synthetic_loaded):
    pool = synthetic_loaded.pool
    scores = score_variants(_prior_for(pool), pool)
    assert scores.shape == (len(pool),)
    assert np.all(np.isfinite(scores))


def test_unflagged_positions_score_zero(synthetic_loaded):
    """A variant the prior says nothing about must sit at the neutral middle."""
    pool = synthetic_loaded.pool
    prior = AgentPrior(
        assay_id="S",
        model="test",
        blind=False,
        critical_positions=[
            CriticalPosition(position=3, wt_aa=pool.wild_type[2], tolerance="none")
        ],
    )
    scores = score_variants(prior, pool)
    for i, mutant in enumerate(pool.mutants):
        positions = pool.mutated_positions(i)
        if 3 not in positions:
            assert scores[i] == 0.0


def test_intolerant_positions_score_below_tolerant(synthetic_loaded):
    pool = synthetic_loaded.pool
    prior = AgentPrior(
        assay_id="S",
        model="test",
        blind=False,
        critical_positions=[
            CriticalPosition(position=3, wt_aa=pool.wild_type[2], tolerance="none"),
            CriticalPosition(position=10, wt_aa=pool.wild_type[9], tolerance="high"),
        ],
    )
    scores = score_variants(prior, pool)
    at_3 = [s for i, s in enumerate(scores) if 3 in pool.mutated_positions(i)]
    at_10 = [s for i, s in enumerate(scores) if 10 in pool.mutated_positions(i)]
    assert at_3 and at_10
    assert max(at_3) < min(at_10)


def test_substitution_rule_applies(synthetic_loaded):
    """A matching rule must move the score; a non-matching one must not."""
    pool = synthetic_loaded.pool
    prior = AgentPrior(
        assay_id="S",
        model="test",
        blind=False,
        substitution_rules=[
            SubstitutionRule(
                from_class="any", to_class="proline", effect="strongly_deleterious"
            )
        ],
    )
    scores = score_variants(prior, pool)
    to_pro = [
        s for i, s in enumerate(scores) if pool.mutants[i].endswith("P")
    ]
    others = [
        s for i, s in enumerate(scores) if not pool.mutants[i].endswith("P")
    ]
    assert to_pro and others
    assert max(to_pro) < min(others)


# ---------------------------------------------------------------------------
# Ablation modes
# ---------------------------------------------------------------------------


def test_shuffled_preserves_distribution_but_not_order(synthetic_loaded):
    """The key control: same marginal distribution, no per-variant information."""
    pool = synthetic_loaded.pool
    base = score_variants(_prior_for(pool), pool)
    shuffled = ap._shuffled_scores(base, seed=0)
    assert np.allclose(np.sort(base), np.sort(shuffled))
    assert not np.allclose(base, shuffled)


def test_shuffled_is_deterministic_per_seed(synthetic_loaded):
    pool = synthetic_loaded.pool
    base = score_variants(_prior_for(pool), pool)
    a = ap._shuffled_scores(base, seed=3)
    b = ap._shuffled_scores(base, seed=3)
    c = ap._shuffled_scores(base, seed=4)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_random_positions_uses_same_count(synthetic_loaded):
    pool = synthetic_loaded.pool
    prior = _prior_for(pool)
    scores = ap._random_position_scores(prior, pool, seed=0)
    assert scores.shape == (len(pool),)
    assert np.all(np.isfinite(scores))
    # Only positions the pool actually mutates may be flagged, so every
    # non-zero score must come from a real mutated position.
    assert np.count_nonzero(scores) > 0


def test_unknown_mode_raises(synthetic_loaded, monkeypatch):
    monkeypatch.setattr(
        ap, "get_prior", lambda *a, **k: _prior_for(synthetic_loaded.pool)
    )
    with pytest.raises(ValueError, match="Unknown agent prior mode"):
        ap.agent_prior_scores(
            synthetic_loaded.metadata, synthetic_loaded.pool, mode="nonsense"
        )


# ---------------------------------------------------------------------------
# Transport, caching and the feature hook
# ---------------------------------------------------------------------------


def test_call_gemini_parses_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["key"] = req.get_header("X-goog-api-key")

        class _Resp:
            def read(self):
                return json.dumps(
                    {
                        "candidates": [
                            {"content": {"parts": [{"text": '{"ok": true}'}]}}
                        ]
                    }
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
    out = ap.call_gemini("hello", model="gemini-test", api_key="k")
    assert out == '{"ok": true}'
    assert "gemini-test" in captured["url"]
    assert captured["key"] == "k"
    assert captured["body"]["generationConfig"]["temperature"] == 0.0


def test_malformed_transport_response_raises(monkeypatch):
    class _Resp:
        def read(self):
            return json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ap.urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(AgentPriorError, match="Unexpected Gemini response"):
        ap.call_gemini("hello", api_key="k")


def test_cache_round_trip_avoids_second_call(synthetic_loaded, tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_call(prompt, **kwargs):
        calls["n"] += 1
        return _response(synthetic_loaded.metadata.wild_type)

    monkeypatch.setattr(ap, "call_gemini", fake_call)

    first = ap.get_prior(synthetic_loaded.metadata, api_key="k")
    second = ap.get_prior(synthetic_loaded.metadata, api_key="k")
    assert calls["n"] == 1
    assert first.to_dict() == second.to_dict()


def test_cache_key_separates_blind_from_named(synthetic_loaded, tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_call(prompt, **kwargs):
        calls["n"] += 1
        return _response(synthetic_loaded.metadata.wild_type)

    monkeypatch.setattr(ap, "call_gemini", fake_call)
    ap.get_prior(synthetic_loaded.metadata, blind=False, api_key="k")
    ap.get_prior(synthetic_loaded.metadata, blind=True, api_key="k")
    assert calls["n"] == 2


def test_feature_hook_appends_one_column(synthetic_loaded, monkeypatch):
    """The prior must add exactly one feature column, like the zero-shot score."""
    monkeypatch.setattr(
        ap, "get_prior", lambda *a, **k: _prior_for(synthetic_loaded.pool)
    )
    base_cfg = EmbeddingConfig(
        featurizer="onehot", n_components=8, include_zero_shot=False
    )
    base = build_features(synthetic_loaded.pool, base_cfg)

    cfg = EmbeddingConfig(
        featurizer="onehot",
        n_components=8,
        include_zero_shot=False,
        include_agent_prior=True,
    )
    with_prior = build_features(
        synthetic_loaded.pool, cfg, metadata=synthetic_loaded.metadata
    )

    assert with_prior.X.shape[1] == base.X.shape[1] + 1
    assert with_prior.includes_agent_prior is True
    assert with_prior.agent_prior is not None
    assert np.allclose(with_prior.X[:, : base.X.shape[1]], base.X)
    assert with_prior.to_dict()["agent_prior"]["mode"] == "agent"


def test_feature_hook_requires_metadata(synthetic_loaded):
    cfg = EmbeddingConfig(
        featurizer="onehot",
        n_components=8,
        include_zero_shot=False,
        include_agent_prior=True,
    )
    with pytest.raises(ValueError, match="requires `metadata`"):
        build_features(synthetic_loaded.pool, cfg)


def test_prior_column_is_standardised(synthetic_loaded, monkeypatch):
    monkeypatch.setattr(
        ap, "get_prior", lambda *a, **k: _prior_for(synthetic_loaded.pool)
    )
    cfg = EmbeddingConfig(
        featurizer="onehot",
        n_components=8,
        include_zero_shot=False,
        include_agent_prior=True,
    )
    fs = build_features(
        synthetic_loaded.pool, cfg, metadata=synthetic_loaded.metadata
    )
    column = fs.X[:, -1]
    assert abs(float(column.mean())) < 1e-9
    assert np.all(np.isfinite(column))


def test_default_config_leaves_prior_off():
    """Existing runs must be unaffected until the prior is explicitly enabled."""
    cfg = EmbeddingConfig()
    assert cfg.include_agent_prior is False


def test_prior_never_queries_the_oracle(synthetic_loaded, monkeypatch):
    """The leakage boundary: building the prior must not spend any budget."""
    monkeypatch.setattr(
        ap, "get_prior", lambda *a, **k: _prior_for(synthetic_loaded.pool)
    )
    oracle = synthetic_loaded.oracle

    def forbidden(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("agent prior queried the oracle")

    monkeypatch.setattr(oracle, "query", forbidden)
    monkeypatch.setattr(oracle, "peek_for_evaluation", forbidden)
    monkeypatch.setattr(oracle, "all_scores_for_evaluation", forbidden)

    cfg = EmbeddingConfig(
        featurizer="onehot",
        n_components=8,
        include_zero_shot=False,
        include_agent_prior=True,
    )
    build_features(synthetic_loaded.pool, cfg, metadata=synthetic_loaded.metadata)
    assert oracle.n_queries == 0
