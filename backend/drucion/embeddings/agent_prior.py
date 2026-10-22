"""LLM-proposed, label-free structural prior over a candidate pool.

Scientific framing
------------------
This module adds a second *hint* alongside the ESM-2 masked-marginal zero-shot
score. Both are label-free per-variant scalars derived from pretrained
knowledge, both are appended as one feature column, and both are therefore in
the same category of modelling input: they are computed before any measurement
is revealed and never touch :class:`~drucion.data.proteingym.ExperimentalOracle`.

The agent is asked for *structure*, not for predictions of fitness it could
only know by having memorised the assay:

* which residue positions are functionally critical, and
* which substitution classes are tolerated or deleterious.

Those claims are then turned into a per-variant score by local arithmetic in
:func:`score_variants`. That split matters for three reasons:

1. **Cost.** One LLM call per assay, not one per variant. A 5,004-variant pool
   costs a single request.
2. **Auditability.** The prior is a small JSON document a reviewer can read and
   disagree with, and it is cached verbatim next to the results.
3. **Falsifiability.** Every position the agent names carries the wild-type
   residue it expects to find there. :func:`_validate_positions` checks that
   against the real wild-type sequence and discards mismatches, which catches
   the single most likely failure mode (see "Numbering" below).

Numbering
---------
ProteinGym positions are 1-indexed into ``target_seq``. The literature often
is not: beta-lactamases are conventionally numbered by the Ambler scheme, and
many sequences are published in mature-protein numbering after signal-peptide
cleavage. An agent recalling "position 238" may therefore be off by a constant
offset, which would silently corrupt the whole prior. Requiring the expected
wild-type residue turns that silent corruption into a dropped claim, and the
drop rate is reported in :attr:`AgentPrior.validation`.

Ablation modes
--------------
``mode`` selects what is actually scored, so the controls run through the same
code path as the real thing:

``agent``
    The prior as returned by the model.
``shuffled``
    The agent's per-variant scores, randomly permuted across variants. Same
    marginal distribution, no information. If this helps, the gain was a
    scaling artefact rather than knowledge.
``random_positions``
    The same *number* of critical positions, chosen uniformly at random, with
    the agent's claims discarded. Tests whether any sparse positional prior
    helps, or specifically the agent's.

``blind`` additionally withholds the assay's identifying metadata (molecule
name, organism, author, year) from the prompt, leaving only the sequence and
the assay type. Because ProteinGym assays are published, a prior that only
works non-blind is recall, not reasoning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import CACHE_DIR, ENV_FILE
from ..data.proteingym import AMINO_ACIDS, AssayMetadata, CandidatePool, parse_mutant

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_API_KEY_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

#: Tolerance labels the agent may use, and the multiplier each implies.
#: "none" means no substitution is tolerated at that position, so a variant
#: there is scored as strongly deleterious.
TOLERANCE_WEIGHTS = {"none": -1.0, "partial": -0.5, "high": 0.0}

#: Effect labels for substitution rules, and the increment each contributes.
EFFECT_WEIGHTS = {
    "strongly_deleterious": -1.0,
    "deleterious": -0.5,
    "neutral": 0.0,
    "beneficial": 0.5,
}

#: Residue classes the agent may reference in a substitution rule.
RESIDUE_CLASSES: dict[str, frozenset[str]] = {
    "hydrophobic": frozenset("AVLIMFWCY"),
    "polar": frozenset("STNQ"),
    "charged": frozenset("DEKRH"),
    "positive": frozenset("KRH"),
    "negative": frozenset("DE"),
    "aromatic": frozenset("FWYH"),
    "small": frozenset("AGSCP"),
    "proline": frozenset("P"),
    "glycine": frozenset("G"),
    "cysteine": frozenset("C"),
    "any": frozenset(AMINO_ACIDS),
}


class AgentPriorError(RuntimeError):
    """Raised when a prior cannot be obtained or does not satisfy the schema."""


# ---------------------------------------------------------------------------
# Prior representation
# ---------------------------------------------------------------------------


@dataclass
class CriticalPosition:
    """One position the agent claims is functionally important."""

    position: int  # 1-indexed into the wild-type sequence
    wt_aa: str  # residue the agent expects to find there
    tolerance: str  # key of TOLERANCE_WEIGHTS
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "wt_aa": self.wt_aa,
            "tolerance": self.tolerance,
            "reason": self.reason,
        }


@dataclass
class SubstitutionRule:
    """A claim about a class of substitutions, independent of position."""

    from_class: str
    to_class: str
    effect: str  # key of EFFECT_WEIGHTS
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "from_class": self.from_class,
            "to_class": self.to_class,
            "effect": self.effect,
            "reason": self.reason,
        }


@dataclass
class AgentPrior:
    """A validated, label-free structural prior for one assay."""

    assay_id: str
    model: str
    blind: bool
    critical_positions: list[CriticalPosition] = field(default_factory=list)
    substitution_rules: list[SubstitutionRule] = field(default_factory=list)
    notes: str = ""
    #: Counts describing what survived validation, reported alongside results.
    validation: dict = field(default_factory=dict)
    #: Raw model response text, kept verbatim so a reviewer can audit it.
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "assay_id": self.assay_id,
            "model": self.model,
            "blind": self.blind,
            "critical_positions": [c.to_dict() for c in self.critical_positions],
            "substitution_rules": [r.to_dict() for r in self.substitution_rules],
            "notes": self.notes,
            "validation": self.validation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentPrior":
        return cls(
            assay_id=d["assay_id"],
            model=d.get("model", DEFAULT_MODEL),
            blind=bool(d.get("blind", False)),
            critical_positions=[
                CriticalPosition(**c) for c in d.get("critical_positions", [])
            ],
            substitution_rules=[
                SubstitutionRule(**r) for r in d.get("substitution_rules", [])
            ],
            notes=d.get("notes", ""),
            validation=d.get("validation", {}),
            raw_response=d.get("raw_response", ""),
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SCHEMA_BLOCK = """\
Return ONLY a JSON object, no prose and no code fence, of exactly this shape:

{
  "critical_positions": [
    {"position": <int, 1-indexed into the sequence given above>,
     "wt_aa": "<the single-letter residue you expect at that position>",
     "tolerance": "none" | "partial" | "high",
     "reason": "<short mechanistic reason>"}
  ],
  "substitution_rules": [
    {"from_class": <class>, "to_class": <class>,
     "effect": "strongly_deleterious" | "deleterious" | "neutral" | "beneficial",
     "reason": "<short reason>"}
  ],
  "notes": "<one or two sentences on your overall reasoning>"
}

<class> must be one of: hydrophobic, polar, charged, positive, negative,
aromatic, small, proline, glycine, cysteine, any.

Rules:
- "position" must be an index into the sequence as given, counting the first
  residue as 1. Do not use any published numbering convention.
- "wt_aa" must be the residue actually at that index in the sequence given.
  It is checked, and any position whose residue does not match is discarded.
- "tolerance" describes how much substitution the position tolerates:
  "none" = essentially any change breaks function, "partial" = some changes
  tolerated, "high" = largely tolerant.
- List at most 40 critical positions and at most 12 substitution rules.
- Do not attempt to recall measured values from any specific published
  experiment. Reason from sequence, structure and mechanism only.
"""


def build_prompt(
    metadata: AssayMetadata,
    blind: bool = False,
    max_seq_chars: int = 2000,
) -> str:
    """Compose the prompt for one assay.

    Only label-free material is included: the wild-type sequence, the mutated
    region, and (unless ``blind``) descriptive metadata. No measured score and
    no pool statistic derived from scores is ever placed in the prompt.
    """
    seq = metadata.wild_type
    truncated = len(seq) > max_seq_chars
    shown = seq[:max_seq_chars]

    lines = [
        "You are assisting a protein engineering campaign. You will be given a",
        "wild-type protein sequence and the kind of experimental assay used to",
        "score single amino-acid substitutions in it. Your job is to say which",
        "positions matter and which kinds of substitution are tolerated, so that",
        "a statistical model can be given a sensible prior before any",
        "measurements are taken.",
        "",
    ]

    if blind:
        lines += [
            f"Assay type: {metadata.selection_type or 'unknown'}",
            f"Assay readout: {metadata.selection_assay or 'unknown'}",
            "(The protein's identity is deliberately withheld.)",
        ]
    else:
        lines += [
            f"Protein: {metadata.molecule_name or 'unknown'}",
            f"Organism: {metadata.source_organism or 'unknown'}",
            f"UniProt: {metadata.uniprot_id or 'unknown'}",
            f"Assay type: {metadata.selection_type or 'unknown'}",
            f"Assay readout: {metadata.selection_assay or 'unknown'}",
        ]

    if metadata.region_mutated:
        lines.append(f"Region mutated: {metadata.region_mutated}")
    lines += [
        f"Sequence length: {len(seq)}",
        "",
        "Wild-type sequence (1-indexed, first residue is position 1):",
        shown,
    ]
    if truncated:
        lines.append(
            f"[sequence truncated at {max_seq_chars} residues; "
            f"full length is {len(seq)}]"
        )
    lines += ["", _SCHEMA_BLOCK]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini transport
# ---------------------------------------------------------------------------


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=value`` file, ignoring blanks and ``#`` comments.

    Deliberately tiny: no interpolation, no export keyword, no multi-line
    values. Surrounding quotes are stripped so a pasted ``KEY="abc"`` works.
    """
    values: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value
    return values


def _resolve_api_key(api_key: str | None = None) -> str:
    """Find the Gemini key: explicit argument, then environment, then .env.

    The ``.env`` fallback exists so the key can live in one file at the
    repository root rather than having to be exported in every shell. It is
    gitignored. The environment takes precedence so a one-off override does
    not require editing the file.
    """
    if api_key:
        return api_key
    for name in _API_KEY_ENV:
        value = os.environ.get(name)
        if value:
            return value
    env_file = ENV_FILE
    if env_file.exists():
        values = _read_env_file(env_file)
        for name in _API_KEY_ENV:
            if values.get(name):
                return values[name]
    raise AgentPriorError(
        "No Gemini API key found. Either put a line reading "
        f"GEMINI_API_KEY=your-key in {env_file}, export GEMINI_API_KEY in your "
        "shell, or pass api_key=... . A free key is available at "
        "https://aistudio.google.com/apikey"
    )


#: HTTP statuses worth retrying: server overload and rate limiting.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def call_gemini(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    max_attempts: int = 5,
    backoff: float = 4.0,
) -> str:
    """POST ``prompt`` to the Gemini REST API and return the response text.

    Uses ``urllib`` rather than an SDK so the project gains no new dependency
    for one request per assay. ``temperature=0`` keeps the prior reproducible,
    which matters because it is a single draw per assay rather than something
    averaged over seeds.

    Transient failures (429 and 5xx) are retried with exponential backoff.
    Without this a momentary overload leaves the benchmark with controls but no
    treatment row, which is a silently misleading result rather than a loud
    failure.
    """
    key = _resolve_api_key(api_key)
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
    ).encode()
    req = urllib.request.Request(
        _ENDPOINT.format(model=model),
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code not in _RETRY_STATUSES or attempt == max_attempts - 1:
                raise AgentPriorError(
                    f"Gemini request failed ({exc.code}): {detail}"
                ) from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_attempts - 1:
                raise AgentPriorError(f"Gemini request failed: {exc}") from exc
            last_error = exc
        delay = backoff * (2**attempt)
        logger.warning(
            "agent prior: %s, retrying in %.0fs (attempt %d/%d)",
            last_error,
            delay,
            attempt + 2,
            max_attempts,
        )
        time.sleep(delay)
    else:  # pragma: no cover - loop always breaks or raises
        raise AgentPriorError(f"Gemini request failed: {last_error}")

    try:
        parts = payload["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as exc:
        raise AgentPriorError(
            f"Unexpected Gemini response shape: {json.dumps(payload)[:500]}"
        ) from exc


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating a code fence."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise AgentPriorError(f"No JSON object in response: {text[:300]!r}")
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AgentPriorError(
                f"Malformed JSON in response: {text[:300]!r}"
            ) from exc
    if not isinstance(obj, dict):
        raise AgentPriorError(f"Expected a JSON object, got {type(obj).__name__}")
    return obj


def _validate_positions(
    raw: Sequence[Any], wild_type: str
) -> tuple[list[CriticalPosition], dict]:
    """Keep only claims whose expected wild-type residue is actually there.

    This is the guard against numbering-convention drift described in the
    module docstring: a prior offset by a signal peptide fails here loudly
    instead of quietly mislabelling every position.
    """
    kept: list[CriticalPosition] = []
    report = {
        "positions_proposed": len(raw),
        "dropped_out_of_range": 0,
        "dropped_wt_mismatch": 0,
        "dropped_bad_tolerance": 0,
        "dropped_malformed": 0,
        "dropped_duplicate": 0,
    }
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            report["dropped_malformed"] += 1
            continue
        try:
            pos = int(item["position"])
            wt_aa = str(item["wt_aa"]).strip().upper()
            tolerance = str(item.get("tolerance", "partial")).strip().lower()
        except (KeyError, TypeError, ValueError):
            report["dropped_malformed"] += 1
            continue
        if tolerance not in TOLERANCE_WEIGHTS:
            report["dropped_bad_tolerance"] += 1
            continue
        if not 1 <= pos <= len(wild_type):
            report["dropped_out_of_range"] += 1
            continue
        if len(wt_aa) != 1 or wild_type[pos - 1] != wt_aa:
            report["dropped_wt_mismatch"] += 1
            continue
        if pos in seen:
            report["dropped_duplicate"] += 1
            continue
        seen.add(pos)
        kept.append(
            CriticalPosition(
                position=pos,
                wt_aa=wt_aa,
                tolerance=tolerance,
                reason=str(item.get("reason", ""))[:300],
            )
        )
    report["positions_kept"] = len(kept)
    if report["positions_proposed"]:
        report["wt_match_rate"] = round(
            len(kept) / report["positions_proposed"], 4
        )
    return kept, report


def _validate_rules(raw: Sequence[Any]) -> tuple[list[SubstitutionRule], dict]:
    kept: list[SubstitutionRule] = []
    report = {"rules_proposed": len(raw), "dropped_bad_rule": 0}
    for item in raw:
        if not isinstance(item, dict):
            report["dropped_bad_rule"] += 1
            continue
        from_class = str(item.get("from_class", "")).strip().lower()
        to_class = str(item.get("to_class", "")).strip().lower()
        effect = str(item.get("effect", "")).strip().lower().replace(" ", "_")
        if (
            from_class not in RESIDUE_CLASSES
            or to_class not in RESIDUE_CLASSES
            or effect not in EFFECT_WEIGHTS
        ):
            report["dropped_bad_rule"] += 1
            continue
        kept.append(
            SubstitutionRule(
                from_class=from_class,
                to_class=to_class,
                effect=effect,
                reason=str(item.get("reason", ""))[:300],
            )
        )
    report["rules_kept"] = len(kept)
    return kept, report


def parse_prior(
    text: str,
    assay_id: str,
    wild_type: str,
    model: str,
    blind: bool,
    max_positions: int = 40,
    max_rules: int = 12,
) -> AgentPrior:
    """Parse and validate a raw model response into an :class:`AgentPrior`."""
    obj = _extract_json(text)
    positions, pos_report = _validate_positions(
        list(obj.get("critical_positions", []))[:max_positions], wild_type
    )
    rules, rule_report = _validate_rules(
        list(obj.get("substitution_rules", []))[:max_rules]
    )
    if not positions and not rules:
        raise AgentPriorError(
            "Prior contained no usable claims after validation: "
            f"{json.dumps(pos_report | rule_report)}"
        )
    return AgentPrior(
        assay_id=assay_id,
        model=model,
        blind=blind,
        critical_positions=positions,
        substitution_rules=rules,
        notes=str(obj.get("notes", ""))[:1000],
        validation=pos_report | rule_report,
        raw_response=text,
    )


# ---------------------------------------------------------------------------
# Scoring: prior -> one scalar per variant
# ---------------------------------------------------------------------------


def score_variants(prior: AgentPrior, pool: CandidatePool) -> np.ndarray:
    """Turn a prior into one scalar per variant, higher meaning more tolerable.

    Local arithmetic only — no model call. A variant's score is the sum of the
    tolerance weight of each position it mutates and the effect weight of every
    substitution rule that matches the change. Variants at positions the agent
    did not flag score 0, which is the neutral middle of the scale rather than
    an implicit claim either way.
    """
    pos_weight = {
        c.position: TOLERANCE_WEIGHTS[c.tolerance] for c in prior.critical_positions
    }
    scores = np.zeros(len(pool), dtype=np.float64)
    for i, mutant in enumerate(pool.mutants):
        total = 0.0
        for sub in parse_mutant(mutant):
            total += pos_weight.get(sub.position, 0.0)
            for rule in prior.substitution_rules:
                if (
                    sub.wt_aa in RESIDUE_CLASSES[rule.from_class]
                    and sub.mut_aa in RESIDUE_CLASSES[rule.to_class]
                ):
                    total += EFFECT_WEIGHTS[rule.effect]
        scores[i] = total
    return scores


def _shuffled_scores(scores: np.ndarray, seed: int) -> np.ndarray:
    """Permute scores across variants: same distribution, no information."""
    rng = np.random.default_rng(seed)
    return scores[rng.permutation(scores.size)]


def _random_position_scores(
    prior: AgentPrior, pool: CandidatePool, seed: int
) -> np.ndarray:
    """Score against the same number of randomly chosen critical positions.

    Keeps the tolerance labels the agent used (so the score distribution is
    comparable) but reassigns them to positions drawn uniformly from those the
    pool actually mutates, and drops the substitution rules.
    """
    mutated = sorted({s.position for m in pool.mutants for s in parse_mutant(m)})
    n = min(len(prior.critical_positions), len(mutated))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(mutated), size=n, replace=False)
    tolerances = [c.tolerance for c in prior.critical_positions[:n]]
    pos_weight = {
        mutated[int(idx)]: TOLERANCE_WEIGHTS[tol]
        for idx, tol in zip(chosen, tolerances)
    }
    scores = np.zeros(len(pool), dtype=np.float64)
    for i, mutant in enumerate(pool.mutants):
        scores[i] = sum(
            pos_weight.get(s.position, 0.0) for s in parse_mutant(mutant)
        )
    return scores


# ---------------------------------------------------------------------------
# Caching and the public entry point
# ---------------------------------------------------------------------------


def _cache_path(assay_id: str, model: str, blind: bool, prompt: str) -> Path:
    """Cache key includes the prompt hash so a prompt edit invalidates it."""
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    tag = "blind" if blind else "named"
    safe_model = model.replace("/", "_")
    return CACHE_DIR / "agent_prior" / f"{assay_id}__{safe_model}__{tag}__{digest}.json"


def get_prior(
    metadata: AssayMetadata,
    model: str = DEFAULT_MODEL,
    blind: bool = False,
    api_key: str | None = None,
    use_cache: bool = True,
    temperature: float = 0.0,
) -> AgentPrior:
    """Fetch (or load from cache) the validated prior for one assay.

    One network call per (assay, model, blind, prompt) combination. The raw
    response is cached verbatim so results are reproducible without a key and
    so the prior can be audited after the fact.
    """
    prompt = build_prompt(metadata, blind=blind)
    path = _cache_path(metadata.assay_id, model, blind, prompt)
    if use_cache and path.exists():
        logger.info("agent prior: cache hit %s", path.name)
        return AgentPrior.from_dict(json.loads(path.read_text()))

    logger.info(
        "agent prior: calling %s for %s (blind=%s)", model, metadata.assay_id, blind
    )
    text = call_gemini(prompt, model=model, api_key=api_key, temperature=temperature)
    prior = parse_prior(
        text,
        assay_id=metadata.assay_id,
        wild_type=metadata.wild_type,
        model=model,
        blind=blind,
    )
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = prior.to_dict() | {"raw_response": prior.raw_response}
        path.write_text(json.dumps(payload, indent=2))
        logger.info("agent prior: cached to %s", path)
    return prior


def agent_prior_scores(
    metadata: AssayMetadata,
    pool: CandidatePool,
    model: str = DEFAULT_MODEL,
    blind: bool = False,
    mode: str = "agent",
    seed: int = 0,
    api_key: str | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, AgentPrior]:
    """Per-variant prior scores for ``pool``, in the requested ablation mode.

    Returns the score vector and the prior it came from, so callers can report
    the validation counts alongside whatever the scores were used for.
    """
    prior = get_prior(
        metadata,
        model=model,
        blind=blind,
        api_key=api_key,
        use_cache=use_cache,
    )
    base = score_variants(prior, pool)
    if mode == "agent":
        scores = base
    elif mode == "shuffled":
        scores = _shuffled_scores(base, seed)
    elif mode == "random_positions":
        scores = _random_position_scores(prior, pool, seed)
    else:
        raise ValueError(
            f"Unknown agent prior mode: {mode!r} "
            "(expected 'agent', 'shuffled' or 'random_positions')"
        )
    return scores, prior
