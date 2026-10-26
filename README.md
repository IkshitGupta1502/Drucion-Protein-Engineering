# Drucion

**Closed-loop, uncertainty-aware active learning for protein engineering.**

Drucion learns which protein variant to measure next. It embeds variants with
ESM-2, fits a Gaussian-process surrogate on the measurements it has already
paid for, estimates its own uncertainty, picks the most informative batch of
untested variants, reveals their real measured fitness, retrains, and repeats.

Every fitness value it sees is a **real, previously published
deep-mutational-scanning measurement** from
[ProteinGym](https://github.com/OATML-Markslab/ProteinGym). This is a
retrospective simulation of experimental selection — no wet-lab experiment is
performed, and no fitness value is ever invented.

📊 **[Pitch deck and demo video](https://docs.google.com/presentation/d/1Et8YwEuiCaiBXIXighHtmpEg4-1I3bkKS3Wipd1Ur5U/edit?usp=sharing)**

---

## Headline result

**VIM-2 β-lactamase** (5,004 measured variants), 200 measurements — 4% of the
pool — averaged over 10 seeds:

| Method | Best measured | Normalised AUC | Top-1% found | Measurements to 99th pct |
|---|---|---|---|---|
| Random selection | 2.264 ± 0.434 | 0.737 ± 0.086 | 4.7% | 92 |
| ESM-guided evolutionary | 1.820 ± 0.493 | 0.722 ± 0.103 | 8.2% | 89 |
| Greedy surrogate | 2.110 ± 0.552 | 0.769 ± 0.101 | 9.2% | 74 |
| Bayesian AL (EI) | 2.206 ± 0.414 | 0.774 ± 0.084 | 10.0% | 78 |
| **Bayesian AL (UCB)** | **2.519 ± 0.384** | **0.821 ± 0.075** | **15.5%** | **68** |

Methods share identical initial measurements per seed, so seeds are paired.
UCB vs. random, paired Wilcoxon over 10 seeds: top-1% recall wins **10/10
seeds, +10.8 points, p = 0.002**; AUC 7/10, p = 0.027; measurements to the 99th
percentile −24, p = 0.031.

**Efficiency:** random spends all 200 measurements to reach a best fitness of
2.264. UCB reaches that level after **140 — 30% fewer experiments.**

Two caveats belong beside that table. The *single best value found* is a max
statistic with large variance and is **not** significant at 10 seeds (p = 0.32).
And this is VIM-2, the assay where the surrogate learns best. What holds up is
enrichment and efficiency — more top variants, sooner — which is the more
useful quantity anyway: you want a shortlist of leads, not one lucky hit.

![Learning curves](results/A4GRB6_PSEAI_Chen_2020_learning_curves.png)

### The benefit tracks surrogate quality

The same pipeline on three assays gives the most useful result here — and it is
not "our method always wins".

| Assay | Surrogate ρ (held out) | Random, top-1% | Best guided method | Enrichment |
|---|---|---|---|---|
| VIM-2 β-lactamase | **0.561** | 4.7% | **UCB — 15.5%** | **3.3×** |
| TEM-1 β-lactamase | 0.476 | 4.0% | evolutionary — 10.8% | 2.7× |
| TPMT | 0.350 | 4.1% | evolutionary — 4.6% | 1.1× |

**Model-guided selection beats random exactly to the extent that the surrogate
can rank unseen variants**, monotonically across all three. Where the model is
informative you get 3× enrichment; where it is not (TPMT, ρ = 0.35) the loop
degenerates into an expensive random search.

### Results that do not flatter the method

**UCB is not universally best.** It dominates on VIM-2 (p = 0.002). On TEM-1 it
beats random (8.5% vs 4.0%, p = 0.031) but trails greedy (10.0%) and the
evolutionary baseline (10.8%). On TPMT nothing beats random significantly — the
accurate statement is "buys nothing measurable", not "is worse".

**The non-learning baseline is strong.** ESM-guided evolutionary search trains
no model at all, yet wins on two of three assays. The trainable surrogate's
advantage is established **only on VIM-2**.

**Exploration keeps the model honest.** Random selection sets the ceiling for
surrogate quality (ρ = 0.561) because its training set is unbiased, while
greedy decays to 0.464 — it only measures what the model already likes. UCB
holds 0.546 *and* delivers the best outcome: the uncertainty term paying for
itself.

### An LLM structural prior improves the model, not the outcome

An optional label-free feature. An LLM is shown the wild-type sequence and the
assay type — **not** the protein's identity — and asked which positions are
functionally critical and which substitutions are tolerated. Its answer becomes
one scalar per variant, appended as a feature column exactly as the ESM-2
zero-shot score is. One API call per assay, not per variant, because it is
asked for *structure* rather than for predicted fitness.

On TPMT it improves the representation, and the controls are what make that
readable:

| Representation | Held-out ρ |
|---|---|
| `esm2 + zero_shot` (what the benchmark uses) | 0.355 ± 0.041 |
| **`+ LLM prior, blind`** | **0.412 ± 0.016** |
| control: shuffled prior scores | 0.311 ± 0.055 |
| control: random critical positions | 0.353 ± 0.026 |

Shuffled scores land *below* baseline, so the gain is not an artefact of adding
a column. Random critical positions land at baseline, so it is not "any sparse
prior helps". And the blind prior recovers 94% of what it achieves when told
the protein's name — so it is reasoning, not recall of a published assay. Every
position it proposes must carry the residue it claims is there, checked against
the real sequence; 34/35 matched named, 14/14 blind.

**It does not convert into better optimization.** Every method's surrogate
improves (UCB ρ 0.277 → 0.359), but none ends meaningfully ahead of random's
4.1% top-1% recall and enrichment stays at ~1.1×. The prior is a **floor, not a
lift** — greedy and UCB had been *below* random because exploitative selection
degrades its own training set, and the prior restores them to roughly random.

That is evidence against reading the surrogate-quality relationship as a lever.
Extrapolating it predicted that lifting TPMT's ρ to ~0.41 would push enrichment
toward 2.5×. **That prediction was made in advance and it failed.** Either
ρ ≈ 0.39 is still below the threshold where guided selection pays here, or
whole-pool rank correlation is the wrong statistic for top-tail enrichment.
These runs cannot separate the two, and only TPMT was tested.

![Agent prior comparison](results/TPMT_HUMAN_Matreyek_2018_agentprior_comparison.png)

Off by default; needs a Gemini key in `.env` (see `.env.example`):

```bash
scripts/run_benchmark.py --assay TPMT_HUMAN_Matreyek_2018 --seeds 10 --agent-prior
```

---

## Quickstart

Python 3.11+ and Node 18+. CPU-only works; CUDA and Apple MPS used when present.

```bash
git clone https://github.com/IkshitGupta1502/drucion-protein-engineering.git
cd drucion-protein-engineering
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py      # ProteinGym v1.3, ~1 GB expanded
cd frontend && npm install && npm run build && cd ..
.venv/bin/python -m uvicorn drucion.api.app:app --port 8000
```

Open <http://localhost:8000>.

**Reproduce the benchmark** (~11 min on an M2 laptop):

```bash
.venv/bin/python scripts/precompute_embeddings.py A4GRB6_PSEAI_Chen_2020
.venv/bin/python scripts/run_benchmark.py --assay A4GRB6_PSEAI_Chen_2020 --seeds 10
```

Every number above regenerates via `scripts/report.py --assay <id>`.

**Generate a bench protocol** for a proposed batch:

```bash
.venv/bin/python scripts/make_protocol.py --assay A4GRB6_PSEAI_Chen_2020 \
    --variants H24C A42G D233H
```

Every value is labelled **sourced** (cited record), **standard** (general
technique) or **estimated** — and an estimate cannot exist without an
experiment that would confirm it. Examples in `results/*_protocol.md`.

**Tests:** `.venv/bin/python -m pytest` — 326 tests, ~4 min.

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
PCA-reduced to 24 components, with the masked-marginal zero-shot score
appended. Held-out ρ on TPMT:

| Representation | without zero-shot | with zero-shot |
|---|---|---|
| **ESM-2 site-resolved** (default) | 0.300 | **0.355** |
| ESM-2 mean-pooled | 0.261 | 0.318 |
| Mutation one-hot (control) | **−0.003** | 0.121 |

The one-hot control carries no pretrained knowledge and sits at ρ ≈ 0.00 —
which is how we know the signal comes from ESM-2, not the pipeline.

**Surrogate.** BoTorch `SingleTaskGP` in double precision, with a random-forest
ensemble as a documented fallback if GP fitting raises. Honest note: RF is
slightly *more* accurate on point predictions (ρ 0.41 vs 0.36 on TPMT at
n=200), but the GP's calibrated uncertainty is what UCB and EI need.

**Acquisition.** UCB (`μ + βσ`), analytic EI, greedy and random, scored over
the *finite* pool — ProteinGym only reveals fitness for variants actually
assayed, so this is ranking, not continuous optimisation. Batch selection is a
diversity-filtered top-k, deliberately **not** applied to the random baseline;
`tests/test_loop.py::TestRandomBaselineIsUniform` pins that.

**Evolutionary baseline.** ESM-2 ranks candidates by zero-shot plausibility and
elite variants focus the search. No surrogate — which makes it the right
control for "is the trainable model doing anything?"

---

## Scientific integrity

The benchmark means nothing if the hidden labels leak. Enforced in code, pinned
by `tests/test_no_leakage.py`:

- `CandidatePool` has no attribute holding fitness.
- `ExperimentalOracle.query()` is the only path from a candidate index to a
  number, and it charges the budget.
- A static test asserts acquisition, surrogate and featurizer modules never
  reference a label-reading API by name.
- Surrogate accuracy uses a fixed 500-variant holdout removed from the pool:
  never trained on, never selectable, never charged.
- Budget is read from the oracle's own counter, not tracked separately.

Per seed, every method gets the same pool, features, initial measurements,
budget and a fresh oracle — only the selection rule differs. PCA is fit on the
whole pool's features: unsupervised and label-free, applied identically to
every method, but transductive, and documented here rather than buried. The
representation was chosen once on TPMT and applied unchanged elsewhere, so
VIM-2 is a clean held-out application of that choice.

---

## Dashboard

Four views, driven by the live API:

- **Setup** — pick any of 217 assays, see provenance and validation report,
  configure method, budget and seed.
- **Live optimization** — step or run with SSE streaming; each pick's
  prediction and uncertainty beside the revealed measurement.
- **Benchmark** — learning curves with ±1 s.d. bands, metrics and seed-paired
  significance tables.
- **Protein explorer** — 3D wild-type fold with probed residues coloured by
  measured fitness, scrubbable round by round.

Model predictions (purple) and real measurements (green) are visually distinct
everywhere. Interactive API docs at `/docs`.

---

## Repository layout

```
backend/drucion/
  data/proteingym.py   # loader, validator, CandidatePool, ExperimentalOracle
  embeddings/          # ESM-2 features, zero-shot scores, LLM structural prior
  models/surrogate.py  # GP (BoTorch) + RF ensemble fallback
  optimization/        # acquisition, evolutionary baseline, the closed loop
  evaluation/          # metrics, publication-quality plots
  experiments/         # multi-method benchmark, SQLite persistence
  protocol/            # bench protocol generation with per-value provenance
  api/                 # FastAPI app, schemas, session manager, SSE
frontend/src/          # React + TypeScript + Plotly dashboard
scripts/               # download_data, run_benchmark, representation_check,
                       #   make_protocol, report
tests/                 # 326 tests
results/               # benchmark CSV/JSON/PNG (committed)
```

---

## Limitations

- **Retrospective, not prospective.** Drucion selects from variants already
  assayed; it does not predict fitness for genuinely novel sequences.
- **Single substitutions only** by default; epistasis is not exercised, though
  multi-mutant assays load fine (`singles_only=False`).
- **The small ESM-2.** The 8M checkpoint was chosen so the demo runs on a
  laptop. Larger checkpoints would likely raise surrogate accuracy and, per the
  mechanism above, the size of the win.
- **Ten seeds** — enough for the paired tests reported, not for small
  differences in the max statistic.
- **No universal winner.** Three assays. UCB is best on VIM-2, the evolutionary
  baseline on TEM-1 and TPMT. Three points show the surrogate-quality
  relationship is monotone; they are not enough to fit it.
- **Structure is for orientation, not prediction.** The 3D view places real
  measurements on the wild-type AlphaFold2 model; nothing predicts structural
  consequence, and the backbone deliberately never morphs.
- **The LLM prior is tested on one assay** — a single draw on TPMT, where ESM-2
  is weakest and an outside prior has the most room to help. Read +0.057 ρ as
  an upper bound, not a typical value.

## Attribution

**ProteinGym** (Notin et al., NeurIPS 2023 D&B) — MIT. Primary assays: Chen et
al. (VIM-2, 2020), Matreyek et al. (TPMT, 2018), Firnberg et al. (TEM-1, 2014).
**ESM-2** (Lin et al., *Science* 2023) — MIT. **BoTorch / GPyTorch** (Balandat
et al., NeurIPS 2020) — MIT.

**EvoEmu** (<https://github.com/ibrahimaasim77/EvoEmu>) inspired this project's
evolutionary search. EvoEmu publishes no licence file, so **no EvoEmu code is
copied here**: its ESM-2 masked-substitution proposal logic and elite-consensus
search were re-implemented independently, with attribution in the module
docstrings. Drucion's contribution relative to EvoEmu is the trainable
surrogate and the experimental feedback loop.
