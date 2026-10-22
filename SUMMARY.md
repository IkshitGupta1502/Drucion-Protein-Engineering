# Drucion — what was built, what was tested, and what it shows

A candid account of the project: the system, the evidence behind each claim,
the things that did not work, and the parts that are unfinished.

---

## 1. What was built

A working closed-loop active-learning system for protein engineering, plus the
benchmark harness and dashboard needed to evaluate it honestly.

**Scientific core** (`backend/drucion/`)

| Component | What it does |
|---|---|
| `data/proteingym.py` | Loads and validates real ProteinGym DMS assays; splits each into a label-free `CandidatePool` and a budget-enforcing `ExperimentalOracle` |
| `embeddings/esm.py` | ESM-2 embeddings — whole-sequence mean pooling, site-resolved features, and masked-marginal zero-shot scores — all disk-cached |
| `embeddings/features.py` | Featurizers (`esm2`, `esm2_mean`, `onehot`) with PCA reduction |
| `models/surrogate.py` | BoTorch `SingleTaskGP` in double precision, plus a random-forest ensemble fallback that is reported whenever it activates |
| `optimization/acquisition.py` | UCB, analytic EI, greedy, random; diversity-filtered batch selection over a finite pool |
| `optimization/evolutionary.py` | ESM-guided evolutionary baseline with elite-focused proposal and unvalidated-proposal accounting |
| `optimization/loop.py` | `ClosedLoopOptimizer` — fit, predict, acquire, measure, retrain, repeat |
| `evaluation/metrics.py` | AUC, regret, top-percentile recall, evals-to-threshold, seed-paired Wilcoxon tests, budget-to-match |
| `experiments/benchmark.py` | Controlled multi-method, multi-seed comparison with an enforced fairness contract |
| `experiments/store.py` | SQLite persistence for experiments, rounds and benchmarks |
| `api/` | FastAPI app, Pydantic schemas, threaded session manager, Server-Sent Events |

**Dashboard** (`frontend/`) — React + TypeScript + Vite + Plotly, four views:
Setup, Live optimization, Benchmark, Protein explorer.

**Scripts** — `download_data.py`, `precompute_embeddings.py`,
`run_benchmark.py`, `representation_check.py`.

---

## 2. What was actually run and verified

Everything below was executed, not just written.

- **Data.** ProteinGym v1.3 downloaded (217 substitution assays, ~1 GB). The
  primary assay, VIM-2 β-lactamase, loads 5,004 single variants with **zero**
  validation failures: no alphabet violations, no length mismatches, no
  mutation/sequence inconsistencies, no duplicates.
- **Embeddings.** ESM-2 (`esm2_t6_8M_UR50D`) run over three full assays on
  Apple MPS at ~70 sequences/s; embeddings and zero-shot scores cached to disk.
- **Benchmarks.** Two saved assays × 5 methods × 10 seeds = 100
  complete closed-loop runs with 15 rounds each, each consuming exactly its 200-measurement budget.
  Results in `results/` as JSON, CSV and PNG.
- **Tests.** 258 tests pass in ~4 min, including an integration test that runs
  ten complete optimization rounds and 19 tests dedicated to proving hidden
  labels cannot reach the optimizer.
- **Application.** Server started, dashboard loaded in a browser, an experiment
  created and stepped through the UI, all four views confirmed rendering real
  data. Every README command in this document was executed.

---

## 3. Results

### 3.1 The headline claim (one assay of three)

On VIM-2 β-lactamase — the assay where the surrogate learns best — with 200
measurements (4% of the pool) over 10 seeds:

- **3.3× enrichment of top-1% variants** — 15.5% found vs. random's 4.7%,
  winning on **all 10 of 10 seeds** (paired Wilcoxon **p = 0.002**).
- **30% fewer experiments** — UCB reaches random's final best fitness after 140
  measurements instead of 200.
- **Better anytime performance** — normalised AUC 0.821 vs 0.737
  (7/10 seeds, **p = 0.027**).
- **Faster to threshold** — 24 fewer measurements to reach the 99th percentile
  (**p = 0.031**).

### 3.2 The claim that does *not* hold

**The single best variant found is not significantly better** (2.519 vs 2.264,
p = 0.32). That statistic is a maximum over a heavy-tailed distribution and is
dominated by luck at ten seeds. We report it and decline to claim it.

### 3.3 Across three assays — including where we lose

The identical pipeline on three assays produced the most useful finding in the
project, and it is not "our method always wins":

| Assay | Surrogate ρ (held out) | Random, top-1% | Best guided method | Enrichment |
|---|---|---|---|---|
| VIM-2 β-lactamase | 0.561 | 4.7% | **UCB — 15.5%** | 3.3× |
| TEM-1 β-lactamase | 0.476 | 4.0% | evolutionary — 10.8% | 2.7× |
| TPMT | 0.350 | 4.1% | evolutionary — 4.6% | 1.1× |

**Model-guided selection beats random exactly to the extent the surrogate can
rank unseen variants**, monotonically across all three assays.

Two results that do not flatter the method, reported because they are true:

**UCB is not universally best.** It dominates on VIM-2 (p = 0.002). On TEM-1 it
beats random on top-1% recall (8.5% vs 4.0%, paired p = 0.031) but sits behind
greedy (10.0%) and the evolutionary baseline (10.8%). On TPMT nothing beats
random significantly — none of UCB's differences reach significance in either
direction (best p = 0.73, AUC p = 0.16, top-1% p = 0.13), so the accurate
phrasing is "buys nothing measurable", not "is worse".

**The non-learning baseline is strong.** ESM-guided evolutionary search trains
no model, yet is the best method on two of three assays, and matched random's
final result with 30% fewer measurements on TEM-1 and 15% fewer on TPMT. Our
trainable surrogate's advantage over a well-built zero-shot ESM baseline is
established **only on VIM-2**, not in general. Reading this work as "learned
surrogates beat ESM heuristics" would over-read the evidence.

The TPMT runs do still show the uncertainty term doing its job: greedy degrades
its own model hardest (final ρ 0.253 vs random's 0.350) while UCB holds 0.277,
falling back toward exploration when the model is uninformative.

### 3.4 Supporting findings

- **The pretrained model matters.** Held-out ρ at n=200 on TPMT: ESM-2
  site-resolved 0.300, mean-pooled 0.261, one-hot control **−0.003**. The
  control carries no pretrained knowledge and predicts nothing, so the signal
  is coming from ESM-2 rather than from the pipeline.
- **Representation matters more than expected.** Whole-sequence mean pooling is
  weak for single mutants — one changed residue barely moves an average over
  266. Site-resolved features gained +0.04, and appending the zero-shot score
  gained a further ~+0.05 (0.355 vs 0.300). Full ablation in
  `results/representation_check.json`.
- **Exploration keeps the model honest.** Final held-out surrogate ρ: random
  0.561, UCB 0.546, EI 0.521, evolutionary 0.518, greedy 0.464. Uniformly
  random measurement trains the best general model — its data is unbiased — so
  random sets the ceiling. Greedy only ever measures what the model already
  likes, so its training set degrades and its model decays to 0.464. UCB gives
  up almost nothing against the ceiling (−0.015) while delivering by far the
  best optimization outcome: the uncertainty term buys optimization
  performance at close to zero cost in model quality.

---

## 4. Honesty and integrity measures

- Measurements are **real published DMS data**, framed throughout as a
  retrospective simulation of experimental selection, never as new wet-lab work.
- `CandidatePool` has four fields and no fitness attribute;
  `ExperimentalOracle.query()` is the only path from an index to a number, and
  it charges budget.
- A static test asserts the acquisition, surrogate and featurizer modules never
  name a label-reading API. The one `peek_for_evaluation` call site is pinned
  by test.
- Surrogate accuracy is measured on a 500-variant holdout excluded from the
  candidate pool: never trained on, never selectable, never charged.
- Every method gets the same pool, features, initial measurements, budget and a
  fresh oracle. Only the selection rule differs.
- Generated variants outside the measured assay are reported as *unvalidated*
  and never receive an invented score.
- Budget overruns and surrogate fallbacks are recorded in the result JSON and
  surfaced in the dashboard rather than silently absorbed.

Documented and *not* hidden: PCA is fit transductively on the whole pool's
features (label-free, applied identically to all methods), and the
representation choice was made once on TPMT then applied unchanged to VIM-2.

---

## 5. Limitations

- **Retrospective, not prospective.** Selection is restricted to variants that
  were already assayed; the pool is pre-filtered in ways a real campaign's
  would not be.
- **Single substitutions.** Epistasis is not exercised by default.
- **Small ESM-2.** The 8M checkpoint was chosen to keep the demo laptop-sized.
- **Ten seeds.** Adequate for the paired tests reported; not for small
  differences in the maximum.
- **Three assays.** The benefit is demonstrably assay-dependent, and on one of
  the three there is none. Three points is enough to show the surrogate-quality
  relationship is monotone, not enough to fit it.
- **No universal winner.** UCB is best on one assay, the non-learning
  evolutionary baseline on two. We do not claim a best method.
- **No structural modelling.** BioEmu was deliberately kept out of the critical
  path.

---

## 6. What remains incomplete

- **Structure-aware features.** The 3D explorer shows where the search looked
  on the real wild-type fold, but the surrogate itself still sees only
  sequence-derived ESM-2 features — spatial proximity between successful
  mutations is visible to the eye and not fed back into the model.
- **Larger ESM-2 checkpoints** are configurable but unbenchmarked. Per the
  mechanism in §3.3, a better representation should *increase* the
  active-learning advantage — the most valuable next experiment.
- **Parallel benchmark execution.** Runs are sequential; parallelising across
  seeds would cut wall time roughly fivefold.
- **Multi-mutant assays.** They load correctly but were not benchmarked.
- **Frontend bundle is still large.** Plotly is split into its own chunk so
  the ~22 kB app shell renders first. Chart views are lazy-loaded, keeping the
  ~4.8 MB charting bundle (1.5 MB gzipped) off the initial setup-page load.

---

## 7. Reproducing everything

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/precompute_embeddings.py A4GRB6_PSEAI_Chen_2020
.venv/bin/python scripts/run_benchmark.py --assay A4GRB6_PSEAI_Chen_2020 --seeds 10
.venv/bin/python -m pytest
cd frontend && npm install && npm run build && cd ..
.venv/bin/python -m uvicorn drucion.api.app:app --port 8000
```

Roughly 15 minutes end to end on an M2 laptop, most of it the benchmark.
