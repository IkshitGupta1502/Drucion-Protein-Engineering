# Handoff: LLM structural prior

**Branch:** `feat/llm-structural-prior`
**Status:** implemented, tested, benchmarked on one assay. The headline finding
is a **negative result with a positive sub-result**, and the honest version of
it is below.
**Date:** 2026-10-21

---

## One-paragraph summary

We added a label-free feature column derived from an LLM (Gemini) that is shown
a wild-type sequence and an assay type and asked which positions are
functionally critical. On TPMT this improves the surrogate's held-out Spearman
from 0.355 to 0.412 — a larger gain than the existing ESM-2 zero-shot feature
provides — and survives the controls that would otherwise explain it away. It
does **not** improve the thing the project exists to improve: top-1% variants
found stays at ~4.3% against random's 4.1%. The prior prevents the exploitative
selectors from degrading their own models, which is real but is a weaker claim
than enrichment. Everything here is one assay, one prior draw.

---

## What was built

| File | What it does |
|---|---|
| `backend/drucion/embeddings/agent_prior.py` | New. Prompting, transport, validation, scoring, ablation modes, caching. |
| `backend/drucion/embeddings/features.py` | Appends the prior as one feature column, mirroring the zero-shot column. Adds `metadata` parameter. |
| `backend/drucion/config.py` | Four `EmbeddingConfig` fields; `ENV_FILE` constant. |
| `backend/drucion/evaluation/plots.py` | `plot_representation_check`, `plot_agent_prior_comparison`. |
| `backend/drucion/experiments/benchmark.py` | Passes `metadata` through to `build_features`. |
| `scripts/representation_check.py` | `--agent-prior`, `--agent-model`; saves to its own JSON. |
| `scripts/run_benchmark.py` | `--agent-prior`, `--agent-prior-named`; **bug fix**, see below. |
| `tests/test_agent_prior.py` | New. 39 tests, no network calls. |
| `.env.example`, `.gitignore` | Credential handling. |

Full suite: **297 passed**.

### Design decisions worth knowing

**The agent is asked for structure, not for fitness.** It returns critical
positions and substitution rules; converting those to a per-variant score is
local arithmetic in `score_variants`. Consequences: one API call per assay
rather than per variant (a 5,004-variant pool costs one request); the prior is
a small auditable JSON document; and the model is never asked to predict a
number it could only know by having memorised the assay.

**The wild-type residue check is load-bearing.** Every position the model names
must carry the residue it expects to find there, validated against the real
sequence. ProteinGym is 1-indexed into `target_seq`; the literature frequently
is not (Ambler numbering for beta-lactamases, mature-protein numbering after
signal-peptide cleavage). Without this check an offset prior would silently
mislabel every position. Observed match rates: 34/35 named, 14/14 blind.
`validation.wt_match_rate` is persisted with every cached prior — **if it drops
below ~0.8 on a new assay, treat the prior as suspect.**

**Blind is the default.** `agent_prior_blind=True` withholds molecule name,
organism, UniProt ID and author. It costs 0.012 ρ against the named prior and
removes the memorisation confound entirely. This default was set from one
assay's result; it is defensible but it is n=1.

**Caching is keyed on a prompt hash.** Editing the prompt invalidates the cache
automatically. Cached priors live in `data/cache/agent_prior/` and include the
raw response verbatim, so results reproduce without an API key and any prior
can be audited after the fact.

---

## Results

### Representation (`results/representation_check_agent.json`)

TPMT, fit on 200 label-free variants, evaluated on the disjoint remainder, 3 seeds.

| Representation | Held-out ρ |
|---|---|
| `esm2` | 0.300 ± 0.025 |
| `esm2 + zero_shot` (baseline) | 0.355 ± 0.041 |
| `+ LLM prior, blind` | **0.412 ± 0.016** |
| `+ LLM prior, named` | 0.424 ± 0.013 |
| control: shuffled scores | 0.311 ± 0.055 |
| control: random positions | 0.353 ± 0.026 |

Both controls land at or below baseline, which rules out the two obvious
artefacts (an extra column inflating ρ; any sparse positional prior helping).
Variance also tightens markedly, 0.041 → 0.016.

### Benchmark (`results/TPMT_HUMAN_Matreyek_2018_agentprior_benchmark.json`)

10 seeds, 50 initial + 15 rounds of 10, blind prior enabled.

| Method | Top-1%, base | Top-1%, prior | Final ρ, base | Final ρ, prior |
|---|---|---|---|---|
| random | 4.1% | 4.1% | 0.350 | 0.393 |
| evolutionary | 4.6% | 4.6% | 0.357 | 0.388 |
| greedy | 3.0% | 4.9% | 0.253 | 0.337 |
| ei | 4.1% | 4.6% | 0.341 | 0.384 |
| ucb | 2.7% | 4.3% | 0.277 | 0.359 |

Normalised AUC essentially unchanged (0.505 random vs 0.492 UCB).

### How to read it

The prediction going in — derived from the README's cross-assay ρ-to-enrichment
relationship — was that lifting TPMT's ρ to ~0.41 should raise enrichment from
1.1× toward ~2.5×. **It did not.** That prediction was stated in advance and it
failed, and the README now says so.

Two candidate explanations, which these runs cannot separate:

1. ρ ≈ 0.39 in-loop is still below the threshold at which guided selection pays
   on this assay (VIM-2, where it works, sits at 0.561).
2. Whole-pool rank correlation is the wrong statistic for predicting top-tail
   enrichment. The prior mostly encodes which variants are *bad*; ruling those
   out need not sharpen discrimination among the good ones.

Distinguishing (1) from (2) is the single most valuable next experiment. See below.

---

## Next steps, in the order I would do them

1. **Replicate the representation check on VIM-2 and TEM-1.** Cheap (priors are
   one call each, cached thereafter), and it answers whether +0.057 generalises
   or whether TPMT was unusually receptive because ESM-2 is weakest there.
   ```bash
   python scripts/representation_check.py --assay A4GRB6_PSEAI_Chen_2020 \
       --featurizers esm2 --agent-prior --save
   ```
   Note: `--save` writes one file per invocation and the filename does not
   include the assay, so **move or rename the output between assays** or the
   second run overwrites the first.

2. **Benchmark the prior on VIM-2.** This is the experiment that separates
   explanation (1) from (2) above. VIM-2's surrogate is already above the
   threshold where guided selection pays, so a ρ bump there has somewhere to
   go. If enrichment moves on VIM-2 but not TPMT, explanation (1) holds and the
   prior is useful where the method already works. If it moves on neither, the
   prior improves the model in a way that does not reach the top tail, and
   explanation (2) holds. ~10 minutes of wall time.

3. **Try the stronger attachment points.** The prior is currently one feature
   column among 26, which is the weakest of the three ways to use it. Stronger:
   - **GP prior mean.** Replace `SingleTaskGP`'s constant mean with the prior's
     predicted fitness so the GP learns the residual. With 50 observations in
     24 dimensions the prior mean dominates predictions in unexplored regions —
     exactly where acquisition is choosing. Interacts with the `Standardize`
     outcome transform; needs care.
   - **Per-position kernel lengthscales.** Most principled, hardest. Requires
     keeping site-resolved features unreduced for flagged positions, since the
     mapping from "position 263 matters" to "PCA component 7 matters" is not
     clean.

4. **Sample several priors.** The prior is a single draw at temperature 0, so
   there is no spread on it — seeds vary the loop, not the prior. Drawing 5 at
   temperature ~0.7 and treating them as an ensemble would put error bars on
   the prior itself.

5. **Consider dropping it if 1 and 2 come back flat.** It adds an API
   dependency and a credential to a project whose current strength is that it
   runs offline from published data. If the gain is TPMT-only, the honest move
   is to keep it documented as a negative result and leave it off by default —
   which is what the current configuration does.

---

## Things that will bite you

**`run_benchmark.py` had a plot-overwrite bug.** `--tag` was passed to the JSON
and CSV writers but not to `plot_benchmark`, so a tagged run silently
overwrote the untagged figures. This destroyed the published TPMT
`_learning_curves.png` and `_metrics.png` during this work; both were
regenerated from the intact baseline JSON and the bug is fixed at
`scripts/run_benchmark.py:88`. If you find other figures that look wrong,
regenerate them from the corresponding `*_benchmark.json` — the JSON was never
at risk.

**Python 3.14 segfaults this project.** `pip install` on this machine resolved
Python 3.14 + torch 2.14, where gpytorch's `torch.jit` usage crashes the
interpreter — it kills `test_api`, `test_evolutionary`, `test_loop`,
`test_no_leakage` and any real benchmark run, sometimes silently with no
traceback. Work was done in a `uv`-created Python 3.12 venv at `.venv/`.
Use `.venv/bin/python`, not `python3`.

**The Gemini model name moves.** `gemini-2.5-flash` returned 404 ("no longer
available to new users") and the current default is `gemini-3.8-flash`. If
calls start 404ing, list the models available to your key:
```python
urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models',
                       headers={'x-goog-api-key': key})
```
Transient 429/5xx are retried with exponential backoff (5 attempts). A failed
prior is recorded as an error row and the run continues — which is right for a
batch, but **check that the treatment row actually exists** before reading a
result, since the controls can succeed while the treatment silently does not.
That happened during this work.

**Credentials.** The key goes in `.env` at the repo root (gitignored), not
`.env.example` (committed). Precedence: explicit argument, then environment,
then `.env`.

---

## Verification

```bash
.venv/bin/python -m pytest -q                      # 297 passed
.venv/bin/python -m pytest tests/test_agent_prior.py -q   # 39 passed, no network
```

`tests/test_agent_prior.py` covers prompt construction (including that the
blind prompt actually strips identity, and that no measured score ever appears
in a prompt), JSON parsing, the numbering guard (including a whole-prior offset
being caught), all three ablation modes, transport against a stubbed response,
cache round-tripping, key resolution precedence, and that building the prior
never touches the oracle.
