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

📊 **[Pitch deck and demo video](https://docs.google.com/presentation/d/1Et8YwEuiCaiBXIXighHtmpEg4-1I3bkKS3Wipd1Ur5U/edit?usp=sharing)**
 · [Demo walkthrough](HACKATHON_DEMO.md) · [Engineering handoff](HANDOFF.md)

---

## Headline result

On **VIM-2 β-lactamase** (`A4GRB6_PSEAI_Chen_2020`, 5,004 measured single
variants), with 200 measurements total — 4% of the pool — averaged over 10
seeds:

| Method | Best measured | Normalised AUC | Top-1% found | Measurements to 99th pct |
|---|---|---|---|---|
| Random selection | 2.264 ± 0.434 | 0.737 ± 0.086 | 4.7% | 92 |
| ESM-guided evolutionary | 1.820 ± 0.493 | 0.722 ± 0.103 | 8.2% | 89 |
| Greedy surrogate | 2.110 ± 0.552 | 0.769 ± 0.101 | 9.2% | 74 |
| Bayesian AL (EI) | 2.206 ± 0.414 | 0.774 ± 0.084 | 10.0% | 78 |
| **Bayesian AL (UCB)** | **2.519 ± 0.384** | **0.821 ± 0.075** | **15.5%** | **68** |

Every method starts each seed from *identical* initial measurements, so seeds
are paired. UCB vs. random, paired Wilcoxon over 10 seeds: top-1% recall wins
**10/10 seeds, +10.8 points, p = 0.002**; AUC 7/10, p = 0.027; measurements to
the 99th percentile −24, p = 0.031.

**Efficiency:** random spends all 200 measurements to reach a best fitness of
2.264. UCB reaches that level after **140 — 30% fewer experiments.**

Two caveats belong next to that table. The *single best value found* is a max
statistic with large variance, and its difference is **not** significant at 10
seeds (p = 0.32). And this is VIM-2, the assay where the surrogate learns best.
What holds up is enrichment and efficiency — finding many more top variants,
sooner — which is the more useful quantity anyway: you want a shortlist of
leads, not one lucky hit.

![Learning curves](results/A4GRB6_PSEAI_Chen_2020_learning_curves.png)

### The benefit tracks surrogate quality

Running the identical pipeline on three assays gives the most useful result in
the project — and it is not "our method always wins".

| Assay | Surrogate ρ (held out) | Random, top-1% | Best guided method | Enrichment |
|---|---|---|---|---|
| VIM-2 β-lactamase | **0.561** | 4.7% | **UCB — 15.5%** | **3.3×** |
| TEM-1 β-lactamase | 0.476 | 4.0% | evolutionary — 10.8% | 2.7× |
| TPMT | 0.350 | 4.1% | evolutionary — 4.6% | 1.1× |

**Model-guided selection beats random exactly to the extent that the surrogate
can rank unseen variants**, and the relationship is monotone across all three.
Where the model is informative you get 3× enrichment; where it is not (TPMT,
ρ = 0.35) the loop degenerates into an expensive random search.

| TEM-1 β-lactamase | TPMT |
|---|---|
| ![TEM-1](results/BLAT_ECOLX_Firnberg_2014_learning_curves.png) | ![TPMT](results/TPMT_HUMAN_Matreyek_2018_learning_curves.png) |

### Results that do not flatter the method

**UCB is not universally best.** It dominates on VIM-2 (p = 0.002). On TEM-1 it
beats random (8.5% vs 4.0%, p = 0.031) but trails both greedy (10.0%) and the
evolutionary baseline (10.8%). On TPMT nothing beats random significantly — the
accurate statement there is "buys nothing measurable", not "is worse".

**The non-learning baseline is strong.** ESM-guided evolutionary search trains
no model at all, yet it is the best method on two of three assays. The
trainable surrogate's advantage is established **only on VIM-2**; reading this
as "learned surrogates beat ESM heuristics" over-reads the evidence.

**Exploration keeps the model honest.** Random selection sets the ceiling for
surrogate quality (ρ = 0.561) because its training set is unbiased, while
greedy decays to 0.464 — it only ever measures what the model already likes.
UCB holds 0.546 *and* delivers the best optimization outcome: the uncertainty
term paying for itself.

### An LLM structural prior improves the model, not the outcome

An optional label-free feature: an LLM is shown the wild-type sequence and the
assay type — not the protein's identity — and asked which positions are
functionally critical. Its answer becomes one scalar per variant, appended as a
feature column exactly as the ESM-2 zero-shot score is. One call per assay.

On TPMT it improves the representation, and the controls are what make that
readable:

| Representation | Held-out ρ |
|---|---|
| `esm2 + zero_shot` (what the benchmark uses) | 0.355 ± 0.041 |
| **`+ LLM prior, blind`** | **0.412 ± 0.016** |
| control: shuffled prior scores | 0.311 ± 0.055 |
| control: random critical positions | 0.353 ± 0.026 |

Shuffled scores land *below* baseline, so the gain is not an artefact of adding
a column; random critical positions land at baseline, so it is not "any sparse
prior helps"; and the blind prior recovers 94% of what it achieves when told
the protein's name, so it is reasoning rather than recall of a published assay.

**It does not convert into better optimization.** Every method's surrogate
improves (UCB 0.277 → 0.359), but none ends meaningfully ahead of random's 4.1%
top-1% recall and enrichment stays at ~1.1×. The prior is a **floor, not a
lift** — greedy and UCB had been *below* random because exploitative selection
degrades its own training set, and the prior restores them to roughly random.

That is evidence against reading the surrogate-quality relationship as a lever.
Extrapolating it predicted that lifting TPMT's ρ to ~0.41 would push enrichment
toward 2.5×. **That prediction was made in advance and it failed.** Either
ρ ≈ 0.39 is still below the threshold where guided selection pays here, or
whole-pool rank correlation is the wrong statistic for top-tail enrichment.

![Agent prior comparison](results/TPMT_HUMAN_Matreyek_2018_agentprior_comparison.png)

Off by default; needs a Gemini key in `.env`. Full detail in
[HANDOFF.md](HANDOFF.md).

---

## Quickstart

Requires Python 3.11+ and Node 18+. CPU-only works; CUDA and Apple MPS are used
automatically when present.

```bash
git clone https://github.com/IkshitGupta1502/drucion.git && cd drucion
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py
```

`download_data.py` fetches ProteinGym v1.3 (43 MB compressed, ~1 GB expanded,
217 assays) plus AlphaFold2 wild-type structures for the 3D explorer (19 MB;
`--skip-structures` to omit).

Build the dashboard and start the server:

```bash
cd frontend && npm install && npm run build && cd ..
.venv/bin/python -m uvicorn drucion.api.app:app --port 8000
```

Open <http://localhost:8000>. For hot reload, run `npm run dev` in `frontend/`
with the API still on 8000; that origin is already CORS-allow-listed.

### Reproduce the benchmark

```bash
.venv/bin/python scripts/precompute_embeddings.py A4GRB6_PSEAI_Chen_2020
.venv/bin/python scripts/run_benchmark.py --assay A4GRB6_PSEAI_Chen_2020 --seeds 10
```

About 11 minutes on an M2 laptop. Every number in this README regenerates via
`scripts/report.py --assay A4GRB6_PSEAI_Chen_2020`.

### Generate a bench protocol

```bash
.venv/bin/python scripts/make_protocol.py --assay A4GRB6_PSEAI_Chen_2020 \
    --variants H24C A42G D233H
```

Produces a wet-lab protocol for the proposed batch. Every value is labelled
**sourced** (cited record), **standard** (general technique) or **estimated** —
and an estimate cannot exist without an experiment that would confirm it.
Examples in `results/*_protocol.md`.

### Tests

```bash
.venv/bin/python -m pytest
```

326 tests, ~4 min. Tests marked `real_data` skip if ProteinGym is not
downloaded.

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

**Representations.** Mean-pooling a 266-residue protein barely moves when one
residue changes, so the default featurizer is *site-resolved*:
`[h_mut(site), h_mut(site) − h_wt(site)]` from ESM-2's final hidden state,
PCA-reduced to 24 components, with the masked-marginal zero-shot score appended.

Held-out ρ on TPMT (fit on 200 random variants, tested on the remainder):

| Representation | without zero-shot | with zero-shot |
|---|---|---|
| **ESM-2 site-resolved** (default) | 0.300 | **0.355** |
| ESM-2 mean-pooled | 0.261 | 0.318 |
| Mutation one-hot (control) | **−0.003** | 0.121 |

The one-hot control carries no pretrained knowledge and sits at ρ ≈ 0.00 —
which is how we know the signal comes from ESM-2, not the pipeline.

**Surrogate.** BoTorch `SingleTaskGP` in double precision. A random-forest
ensemble is a documented fallback, selected automatically if GP fitting raises,
and the active surrogate is always reported. Honest note: RF is slightly *more*
accurate on point predictions (ρ 0.41 vs 0.36 on TPMT at n=200), but the GP's
calibrated uncertainty is what UCB and EI need.

**Acquisition.** UCB (`μ + βσ`), analytic EI, greedy and random, scored over the
*finite* pool — ProteinGym can only reveal fitness for variants actually
assayed, so this is ranking, not continuous optimisation. Batch selection is a
diversity-filtered top-k, which stops one promising cluster consuming a whole
batch. The filter is deliberately **not** applied to the random baseline;
`tests/test_loop.py::TestRandomBaselineIsUniform` pins that.

**Evolutionary baseline.** ESM-2 ranks candidates by zero-shot plausibility and
elite variants focus the search. No surrogate — which makes it the right
control for "is the trainable model doing anything?"

---

## Scientific integrity

The benchmark is only meaningful if the hidden labels stay hidden. Enforced in
code and pinned by `tests/test_no_leakage.py` (19 tests):

- `CandidatePool` has no attribute that holds fitness.
- `ExperimentalOracle.query()` is the only path from a candidate index to a
  number, and it charges the budget.
- A static test asserts acquisition, surrogate and featurizer modules never
  reference any label-reading API by name.
- Surrogate accuracy uses a fixed 500-variant holdout removed from the pool
  entirely: never trained on, never selectable, never charged.
- Budget is read back from the oracle's own counter, not tracked separately.

**Fairness.** Per seed, every method gets the same pool, features, initial
measurements, budget and a fresh oracle. Only the selection rule differs.

**PCA caveat.** PCA is fit on the whole pool's features. Unsupervised and
label-free, applied identically to every method — but transductive, and
documented here rather than buried.

**Representation choice** was made once on TPMT and applied unchanged
elsewhere, so VIM-2 is a clean held-out application of that choice.

**Generated variants.** The evolutionary engine can propose substitutions never
assayed. Those are surfaced as *unvalidated* and never assigned an invented
score.

---

## Dashboard

Four views, all driven by the live API:

- **Setup** — pick any of 217 assays, see provenance and validation report,
  configure method, budget and seed.
- **Live optimization** — step or run with SSE streaming; each pick's
  prediction and uncertainty shown beside the revealed measurement.
- **Benchmark** — learning curves with ±1 s.d. bands, metrics and seed-paired
  significance tables.
- **Protein explorer** — 3D wild-type fold with probed residues coloured by
  measured fitness, scrubbable round by round.

Model predictions (purple) and real measurements (green) are visually distinct
everywhere.

---

## Repository layout

```
backend/drucion/
  data/proteingym.py     # loader, validator, CandidatePool, ExperimentalOracle
  embeddings/            # ESM-2 features, zero-shot scores, LLM structural prior
  models/surrogate.py    # GP (BoTorch) + RF ensemble fallback
  optimization/          # acquisition, evolutionary baseline, the closed loop
  evaluation/            # metrics, publication-quality plots
  experiments/           # multi-method benchmark, SQLite persistence
  protocol/              # bench protocol generation with per-value provenance
  api/                   # FastAPI app, schemas, session manager, SSE
frontend/src/            # React + TypeScript + Plotly dashboard
scripts/                 # download_data, run_benchmark, representation_check,
                         #   make_protocol, report
tests/                   # 326 tests
results/                 # benchmark CSV/JSON/PNG (committed)
```

## API

Interactive docs at `/docs`. The loop is driven by `POST /api/experiments`,
then `/step` or `/run`, with progress on `/stream` (Server-Sent Events) and
saved benchmarks at `GET /api/results/precomputed`.

---

## Provenance and attribution

**ProteinGym** (Notin et al., NeurIPS 2023 D&B) — MIT. Primary assays: Chen et
al. (VIM-2, 2020), Matreyek et al. (TPMT, 2018), Firnberg et al. (TEM-1, 2014).

**ESM-2** (Lin et al., *Science* 2023) — `facebook/esm2_t6_8M_UR50D`, MIT.
**BoTorch / GPyTorch** (Balandat et al., NeurIPS 2020) — MIT.

**EvoEmu** (<https://github.com/ibrahimaasim77/EvoEmu>) — the inspiration for
this project's evolutionary search. EvoEmu publishes no licence file, so **no
EvoEmu code is copied here**: its ESM-2 masked-substitution proposal logic and
elite-consensus search were re-implemented independently, with attribution in
the module docstrings. Drucion's contribution relative to EvoEmu is the
trainable surrogate and the experimental feedback loop — EvoEmu performs
inference-driven search with no learning.

---

## Limitations

- **Retrospective, not prospective.** Drucion selects from variants already
  assayed. It does not predict fitness for genuinely novel sequences.
- **Single substitutions only** by default; epistasis is not exercised, though
  multi-mutant assays load fine (`singles_only=False`).
- **The small ESM-2.** The 8M checkpoint was chosen so the demo runs on a
  laptop. Larger checkpoints would likely raise surrogate accuracy and, per the
  mechanism above, the size of the win.
- **Ten seeds.** Enough for the paired tests reported, not for small
  differences in the max statistic.
- **Assay-dependent benefit, no universal winner.** Three assays. UCB is best on
  VIM-2, the evolutionary baseline on TEM-1 and TPMT. Three points show the
  surrogate-quality relationship is monotone; they are not enough to fit it.
- **Structure is for orientation, not prediction.** The 3D view places real
  measurements on the wild-type AlphaFold2 model. Nothing here predicts
  structural consequence, and the backbone deliberately never morphs.
- **The LLM prior is tested on one assay.** A single prior draw on TPMT, the
  assay where ESM-2 is weakest — the case with the most room for an outside
  prior to help. Read +0.057 ρ as an upper bound, not a typical value.

## What is not finished

- Structure-aware *features* — spatial proximity is visible in the 3D view but
  not fed back into the model.
- Larger ESM-2 checkpoints are configurable but unbenchmarked.
- The benchmark runs sequentially; parallelising seeds would cut wall time
  roughly fivefold.
- The LLM prior has not been replicated on VIM-2 or TEM-1, and is used only as
  a feature column — not as the GP's prior mean or as per-position lengthscales.
- A **logbook** for entering your own measurements, so the loop can run
  prospectively against a real bench rather than a replayed assay.
