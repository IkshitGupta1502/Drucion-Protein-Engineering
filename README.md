# Drucion

**Closed-loop, uncertainty-aware active learning for protein engineering.**

Drucion learns which protein variant to measure next. It embeds variants with
ESM-2, fits a Gaussian-process surrogate on the measurements it has already
paid for, estimates its own uncertainty, picks the most informative batch of
untested variants, reveals their real measured fitness, retrains, and repeats.

Every fitness value it ever sees is a **real, previously published
deep-mutational-scanning measurement** from
[ProteinGym](https://github.com/OATML-Markslab/ProteinGym). This is a
retrospective simulation of experimental selection — no wet-lab experiment is
performed here, and no fitness value is ever invented.

---

## Headline result

On **VIM-2 β-lactamase** (`A4GRB6_PSEAI_Chen_2020`, 5,004 measured single
variants, carbapenem resistance), with 50 initial measurements, batches of 10
and 15 rounds (200 measurements total, 4% of the pool), averaged over 10 seeds:

| Method | Best measured | Normalised AUC | Top-1% variants found | Measurements to reach 99th pct |
|---|---|---|---|---|
| Random selection | 2.264 ± 0.434 | 0.737 ± 0.086 | 4.7% | 92 |
| ESM-guided evolutionary | 1.820 ± 0.493 | 0.722 ± 0.103 | 8.2% | 89 (80% of seeds) |
| Greedy surrogate | 2.110 ± 0.552 | 0.769 ± 0.101 | 9.2% | 74 (90% of seeds) |
| Bayesian AL (EI) | 2.206 ± 0.414 | 0.774 ± 0.084 | 10.0% | 78 |
| **Bayesian AL (UCB)** | **2.519 ± 0.384** | **0.821 ± 0.075** | **15.5%** | **68** |

Because every method starts each seed from *identical* initial measurements,
seeds are paired and a paired test is the right one. UCB vs. random, paired
Wilcoxon over 10 seeds:

| Metric | Seeds won | Mean difference | p |
|---|---|---|---|
| Top-1% variants found | **10 / 10** | **+10.8 points** (3.3× enrichment) | **0.002** |
| Normalised AUC | 7 / 10 | +0.084 | **0.027** |
| Measurements to 99th pct | 6 / 10 (4 tied) | **−24 measurements** | **0.031** |
| Best measured fitness | 7 / 10 | +0.254 | 0.32 (not significant) |

**Experimental efficiency:** random selection spends all 200 measurements to
reach a best fitness of 2.264. UCB reaches that same level after **140
measurements — 30% fewer experiments.**

The honest caveat is in that table: the *single best value found* is a max
statistic with large variance, and its difference is **not** statistically
significant at 10 seeds. The claims that do hold up are about enrichment and
efficiency — finding many more top variants, sooner. For protein engineering
that is the more useful quantity anyway: you want a shortlist of good leads,
not one lucky hit.

![Learning curves](results/A4GRB6_PSEAI_Chen_2020_learning_curves.png)

### Exploration keeps the model honest

A second result falls out of the same runs. Measuring each method's surrogate
against the held-out set at the end of the budget:

| Method | Final held-out ρ | Normalised AUC |
|---|---|---|
| Random selection | 0.561 | 0.737 |
| **Bayesian AL (UCB)** | 0.546 | **0.821** |
| Bayesian AL (EI) | 0.521 | 0.774 |
| ESM-guided evolutionary | 0.518 | 0.722 |
| Greedy surrogate | **0.464** | 0.769 |

Uniformly random measurement is the best way to train a general-purpose model —
its training set is unbiased — so random sets the ceiling at ρ = 0.561. Greedy
exploitation sits at the other extreme: it only ever measures what the model
already likes, so its training set becomes progressively less informative and
its model decays to 0.464.

UCB gives up almost nothing against the random ceiling (0.546 vs 0.561) while
delivering by far the best optimization outcome. That is the uncertainty term
paying for itself — it buys optimization performance at close to zero cost in
model quality, which is exactly what greedy cannot do.

### Where it does *not* work, and why

On **TPMT** (`TPMT_HUMAN_Matreyek_2018`, a human pharmacogene abundance assay)
the same system shows **no detectable benefit** over random selection. UCB
scores 1.527 ± 0.126 against random's 1.591 ± 0.194, and the ESM-guided
evolutionary baseline — which trains nothing — wins outright at 1.655 ± 0.201,
matching random's final result with 15% fewer measurements.

Paired against random over the same 10 seeds, none of UCB's differences are
significant in either direction (best measured p = 0.73, AUC p = 0.16, top-1%
p = 0.13). The accurate statement is not "active learning is worse here" but
"active learning buys nothing measurable here."

The mechanism is visible in the surrogate's own held-out accuracy:

| Assay | Surrogate ρ (start → end of run) | Does active learning help? |
|---|---|---|
| VIM-2 β-lactamase | 0.40 → **0.56** | Yes, substantially |
| TPMT | 0.23 → 0.35 | No detectable effect |

**Active-learning benefit tracks surrogate quality.** When the surrogate can
genuinely rank unseen variants, steering the budget with it pays off. When it
cannot (ρ ≈ 0.3), there is nothing to steer with and the loop reduces to an
expensive random search. The same TPMT runs also show greedy exploitation
degrading its own model hardest (ρ 0.253 vs random's 0.350) — which is why the
uncertainty estimate earns its place: UCB falls back toward exploration when
the model is uninformative, and greedy cannot.

---

## Quickstart

Requires Python 3.11+ and Node 18+. CPU-only works; CUDA and Apple MPS are used
automatically when present.

```bash
git clone https://github.com/axhero7/drucion.git && cd drucion
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py
```

`download_data.py` fetches ProteinGym v1.3 (43 MB compressed, ~1 GB expanded,
217 assays).

Build the dashboard and start the server:

```bash
cd frontend && npm install && npm run build && cd ..
.venv/bin/python -m uvicorn drucion.api.app:app --port 8000
```

Open <http://localhost:8000>.

For frontend development with hot reload, keep the API running on port 8000 and
run `npm run dev` in `frontend/`. Vite serves the UI on
<http://localhost:5173> and calls the API directly at `localhost:8000`; that
origin is already allow-listed in the API's CORS configuration.

### Reproduce the benchmark

```bash
.venv/bin/python scripts/precompute_embeddings.py A4GRB6_PSEAI_Chen_2020
.venv/bin/python scripts/run_benchmark.py --assay A4GRB6_PSEAI_Chen_2020 --seeds 10
```

Takes about 11 minutes on an M2 laptop (CPU/MPS) and writes CSV, JSON and
publication-quality plots to `results/`.

Every number quoted in this README can be regenerated from those artifacts:

```bash
.venv/bin/python scripts/report.py --assay A4GRB6_PSEAI_Chen_2020
```

### Tests

```bash
.venv/bin/python -m pytest
```

258 tests, ~4 min. Tests marked `real_data` skip automatically if ProteinGym has
not been downloaded.

---

## How it works

```
ProteinGym assay ──► CandidatePool  (mutations + sequences, NO labels)
                 └─► ExperimentalOracle  (real DMS scores, sealed behind a budget)

              ┌──────────────────────────────────────────────┐
              │  1. fit surrogate on revealed measurements   │
              │  2. predict mean + uncertainty for the pool  │
              │  3. score candidates with an acquisition fn  │
              │  4. select a diverse batch of untested ones  │
              │  5. query the oracle (spends budget)         │
              │  6. add measurements, retrain, repeat        │
              └──────────────────────────────────────────────┘
```

**Representations** (`backend/drucion/embeddings/`). Mean-pooling a 266-residue
protein barely moves when one residue changes, so the default featurizer is
*site-resolved*: for each variant it takes ESM-2's final hidden state at the
mutated residue and concatenates it with the difference from the wild-type
hidden state at that same position, `[h_mut(site), h_mut(site) − h_wt(site)]`.
That is PCA-reduced to 24 components, and the ESM-2 masked-marginal zero-shot
score is appended as one more label-free feature.

Held-out Spearman ρ on TPMT, surrogate fit on 200 random variants and tested on
the disjoint remainder (`scripts/representation_check.py --save`, saved to
`results/representation_check.json`):

| Representation | without zero-shot | with zero-shot |
|---|---|---|
| **ESM-2 site-resolved** (default) | 0.300 | **0.355** |
| ESM-2 mean-pooled | 0.261 | 0.318 |
| Mutation one-hot (control) | **−0.003** | 0.121 |

Two things to read off that table. Site-resolved features beat mean pooling,
and the zero-shot score adds a further ~0.05 on top of either. And the one-hot
control — which carries no pretrained knowledge at all — sits at ρ ≈ 0.00,
which is how we know the signal comes from ESM-2 rather than from the pipeline
or from some artefact of the evaluation.

**Surrogate** (`backend/drucion/models/surrogate.py`). BoTorch `SingleTaskGP` in
double precision with input normalisation and outcome standardisation, giving
a posterior mean and standard deviation. A random-forest ensemble (per-tree
spread as uncertainty) is available as a documented fallback and is selected
automatically if GP fitting raises; the active surrogate is always reported, so
a run never silently claims to be a GP when it is not. Honest note: the RF
ensemble is slightly *more* accurate on point predictions (ρ 0.41 vs 0.36 on
TPMT at n=200), but the GP's calibrated uncertainty is what UCB and EI need.

**Acquisition** (`backend/drucion/optimization/acquisition.py`). UCB
(`μ + βσ`), analytic Expected Improvement, greedy exploitation, and random, all
scored over the *finite* pool of measured candidates — ProteinGym can only
reveal fitness for variants that were actually assayed, so this is ranking, not
continuous optimisation in embedding space. Batch selection is a diversity-
filtered top-k: a candidate within a small feature distance of an already-chosen
pick is skipped for the next best, which stops one promising cluster from
consuming a whole batch. The diversity filter is deliberately **not** applied
to the random baseline — random screening has no notion of diversity, and
filtering it would mean comparing against a smarter-than-random baseline.
`tests/test_loop.py::TestRandomBaselineIsUniform` pins that.

**Evolutionary baseline** (`backend/drucion/optimization/evolutionary.py`).
ESM-2 ranks candidates by zero-shot plausibility, and elite variants from
previous rounds focus the search onto the positions they mutate. It uses
experimental feedback only to choose *where* to look, with no surrogate — which
is what makes it the right control for "is the trainable model doing anything?"

---

## Scientific integrity

The benchmark is only meaningful if the hidden labels stay hidden. That is
enforced in code and pinned by tests (`tests/test_no_leakage.py`, 19 tests):

- `CandidatePool` is a frozen dataclass with exactly four fields — assay id,
  wild type, mutation specs, sequences. It has no attribute that holds fitness.
- `ExperimentalOracle` keeps the scores private. `query()` is the only path
  from a candidate index to a number, and it charges the budget.
- A static test asserts that the acquisition, surrogate and featurizer modules
  never reference any label-reading API by name.
- Surrogate accuracy is measured on a fixed 500-variant holdout that is removed
  from the candidate pool entirely: never trained on, never selectable, never
  charged to budget. Reading it goes through `peek_for_evaluation`, which is
  called once, after selection is committed, and is pinned to that single call
  site by a test.
- Budget accounting is read back from the oracle's own counter, not tracked
  separately, and every method is asserted to consume exactly
  `n_initial + batch_size × n_rounds` measurements.

**Fairness.** For each seed, every method gets the same candidate pool, the same
precomputed features, the same initial measurements, the same budget, and a
fresh oracle so counters never bleed between runs. Only the selection rule
differs.

**PCA caveat.** PCA is fit on the features of the whole candidate pool. That is
unsupervised and label-free, so it leaks no fitness information, and it is
applied identically to every method. It is nonetheless transductive, and is
documented here rather than buried.

**Representation choice.** The site-resolved featurizer and the zero-shot
feature were selected once, on TPMT, using `scripts/representation_check.py`,
and then applied unchanged to the other assays. VIM-2 — the headline result —
is therefore a clean held-out application of that choice.

**Generated variants.** The evolutionary engine can propose novel substitutions
that were never assayed. Those are counted and surfaced in the UI as
*unvalidated*, and are never assigned an invented score or included in any
ground-truth comparison.

---

## Dashboard

Four views, all driven by the live API:

- **Setup** — pick any of the 217 ProteinGym assays, see its provenance,
  publication, wild-type sequence, full validation report (how many rows were
  dropped and why), and achievable fitness range; configure method, budget,
  batch size, representation, surrogate and seed.
- **Live optimization** — step one round at a time or run to completion, with
  Server-Sent Events streaming progress. Shows each selected variant's
  acquisition score, predicted fitness, predicted uncertainty and then the
  revealed measurement side by side, plus the surrogate's held-out accuracy
  climbing as measurements accumulate.
- **Benchmark** — best measured fitness vs. cumulative evaluations with ±1 s.d.
  bands, the efficiency comparison, the full metrics table, the seed-paired
  significance table, and the surrogate-accuracy curves. Reads saved results
  instantly or launches a new benchmark.
- **Protein explorer** — wild type vs. best discovered sequence with
  substitutions highlighted, predicted-vs-measured scatter coloured by round,
  and which residue positions the search concentrated on.

Model predictions (purple) and real measurements (green) are visually distinct
everywhere they appear.

---

## Repository layout

```
backend/drucion/
  config.py              # dataclass configuration, device resolution
  data/proteingym.py     # loader, validator, CandidatePool, ExperimentalOracle
  embeddings/
    esm.py               # ESM-2 embeddings (mean + site-resolved), zero-shot scores, caching
    features.py          # featurizers, PCA reduction
  models/surrogate.py    # GP (BoTorch) + RF ensemble fallback
  optimization/
    acquisition.py       # UCB, EI, greedy, batch selection with diversity
    evolutionary.py      # ESM-guided evolutionary baseline
    loop.py              # ClosedLoopOptimizer — the closed loop
  evaluation/
    metrics.py           # AUC, regret, recall, paired tests, efficiency
    plots.py             # publication-quality figures
  experiments/
    benchmark.py         # controlled multi-method, multi-seed comparison
    store.py             # SQLite persistence
  api/                   # FastAPI app, Pydantic schemas, session manager, SSE
frontend/src/            # React + TypeScript + Plotly dashboard
scripts/                 # download_data, precompute_embeddings, run_benchmark,
                         #   representation_check, report
tests/                   # 258 tests
results/                 # benchmark CSV/JSON/PNG (committed)
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/assays` | All locally available ProteinGym assays |
| `GET /api/assays/{id}?full=true` | Metadata, provenance, validation report, pool stats |
| `POST /api/experiments` | Create an experiment (features computed in background) |
| `GET /api/experiments/{id}` | Current state |
| `POST /api/experiments/{id}/step` | Run exactly one round |
| `POST /api/experiments/{id}/run` | Run remaining rounds in a background thread |
| `POST /api/experiments/{id}/pause` | Stop after the current round |
| `POST /api/experiments/{id}/reset` | Fresh run, same configuration |
| `GET /api/experiments/{id}/stream` | Server-Sent Events |
| `GET /api/experiments/{id}/results` | Full results, sequence diff, round history |
| `POST /api/benchmarks` | Launch a multi-method, multi-seed benchmark |
| `GET /api/benchmarks/{id}` | Benchmark progress and results |
| `GET /api/results/precomputed` | Saved benchmark JSON from `results/` |

Interactive docs at `/docs`.

---

## Provenance, licensing and attribution

**ProteinGym** (Notin et al., NeurIPS 2023 Datasets & Benchmarks) — MIT licence,
<https://github.com/OATML-Markslab/ProteinGym>. Assay data downloaded from
`marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/`. Each assay carries its own
underlying publication, shown in the dashboard; the primary assays here are
Chen et al. (VIM-2 β-lactamase, 2020), Matreyek et al. (TPMT, 2018) and Firnberg
et al. (TEM-1 β-lactamase, 2014).

**ESM-2** (Lin et al., *Science* 2023) — `facebook/esm2_t6_8M_UR50D`, MIT
licence. Larger checkpoints work by changing `EmbeddingConfig.model_name`.

**BoTorch / GPyTorch** (Balandat et al., NeurIPS 2020) — MIT licence.

**EvoEmu** (<https://github.com/ibrahimaasim77/EvoEmu>) — the inspiration for
this project's evolutionary search and the starting point for its framing.
EvoEmu publishes no licence file, so **no EvoEmu code is copied here**. Its
ESM-2 masked-substitution proposal logic and elite-consensus evolutionary search
were re-implemented independently in
`backend/drucion/embeddings/esm.py` and
`backend/drucion/optimization/evolutionary.py`, with attribution in the module
docstrings. Drucion's contribution relative to EvoEmu is the trainable
surrogate and the experimental feedback loop: EvoEmu performs inference-driven
search with no learning, whereas Drucion trains a model on revealed
measurements and uses its uncertainty to decide what to measure next.

BioEmu is deliberately **not** in the critical path — structure-based analysis
would be an optional extension, not a dependency.

---

## Limitations

- **Retrospective, not prospective.** Drucion selects from variants that were
  already assayed. It does not predict the fitness of genuinely novel sequences,
  and the candidate pool is therefore pre-filtered in ways a real campaign's
  would not be.
- **Single substitutions only** by default. Epistasis between mutations is not
  exercised, though multi-mutant assays load fine (`singles_only=False`).
- **The small ESM-2.** The 8M-parameter checkpoint was chosen so the whole
  demo runs on a laptop. Larger checkpoints would likely raise surrogate
  accuracy and, per the mechanism above, the size of the active-learning win.
- **Ten seeds.** Enough for the paired tests reported, not enough to call small
  differences in the max statistic.
- **Assay-dependent benefit.** Saved benchmarks cover two assays. The claim is not
  "active learning always wins" — it demonstrably does not on TPMT.
- **No structure.** 3D visualisation and structure-aware features are unbuilt.

## What is not finished

- 3Dmol.js structure view in the Protein Explorer (the view currently shows
  sequence-level and position-level analysis only).
- Larger ESM-2 checkpoints are configurable but have not been benchmarked.
- The benchmark runs methods sequentially; parallelising across seeds would cut
  wall time roughly fivefold.
