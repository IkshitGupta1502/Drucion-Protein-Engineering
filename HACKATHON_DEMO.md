# Drucion — 3-minute demo walkthrough

**Track:** Self-Accelerating AI for Drug Development

**One-line pitch:** Drucion is an AI that learns which protein experiment to
run next. On real β-lactamase data it finds 3.3× more top-performing variants
than random screening using 30% fewer measurements — and across three assays it
shows exactly when that advantage holds and when it vanishes.

---

## Before you start

```bash
.venv/bin/python -m uvicorn drucion.api.app:app --port 8000
```

Open <http://localhost:8000>. Make sure `results/` contains the saved benchmark
JSON (it is committed, so this should just work). Have the **Benchmark** tab
pre-loaded in a second browser tab — it is the closing argument and you do not
want to wait for it to render live.

---

## 0:00 — 0:30 · The problem

> "Protein engineering is bottlenecked by measurement. Every variant you test
> costs real money and real time, and the space is astronomically large. The
> question is not *can we predict fitness* — it's **which experiment should we
> run next?**
>
> Drucion answers that with a closed loop: predict, choose, measure, learn,
> repeat. And we validate it against real published lab data, so we can prove
> the loop actually works rather than just asserting it."

**Say the honesty line early — it earns credibility for everything after:**

> "Everything you'll see is a retrospective simulation on real deep-mutational-
> scanning measurements from ProteinGym. We're not claiming new wet-lab
> experiments. The measurements already exist; we hide them, and the agent has
> to spend budget to see them."

---

## 0:30 — 1:00 · Setup tab — the data is real

Point at the **Dataset & provenance** card.

> "This is VIM-2 β-lactamase — the enzyme that makes bacteria resistant to
> carbapenems, our antibiotics of last resort. 5,004 single mutants, every one
> experimentally measured by Chen et al. in 2020.
>
> We validate the file on load — every sequence must match its mutation spec,
> the alphabet must be canonical, no duplicates. That report is on screen.
> And this panel" — point at *Fitness range* — "is flagged 'hidden from the
> optimizer', because it is. The optimizer only ever sees mutations and
> sequences; the scores live behind a budgeted oracle."

Point at the budget line: **200 measurements out of 5,004** — 4% of the pool.

Click **Start experiment**.

---

## 1:00 — 2:00 · Live optimization — watch it learn

The run prepares (embeddings are cached, so it's quick), then lands on the Live
tab.

Click **Step one round**. Then narrate the variant table:

> "Here's the loop in one screen. For each variant the model picked, you see
> what it **predicted** — in purple — its **uncertainty**, and then the real
> measurement it just paid to reveal, in green. Predictions were made *before*
> the measurement. Those two colours are never mixed anywhere in this app.
>
> Now the important part: it retrains on everything it has measured, and that
> changes what it picks next."

Click **Step** two or three more times. Point at the two charts:

> "Left: best measured fitness climbing as budget is spent. Right — this is the
> one I care about — the surrogate's accuracy on a **held-out set that's
> excluded from the candidate pool entirely**. Never trained on, never
> selectable, costs no budget. It climbs from about 0.40 to 0.55 Spearman over
> the run. The model is genuinely getting better at ranking proteins it has
> never seen, from feedback alone. That's the self-accelerating part."

Click **Run to completion** and let it stream while you move on.

---

## 2:00 — 2:40 · Benchmark tab — the proof

Switch to the pre-loaded **Benchmark** tab. This chart is the centrepiece.

> "Five methods, ten random seeds, identical candidate pool, identical budget,
> and — critically — for each seed every method starts from the *same* initial
> measurements. The only thing that differs is which variant each one chooses
> to measure next.
>
> Orange is our Bayesian active learner. Grey is random screening. Random
> spends its entire 200-measurement budget to reach this level" — trace the
> grey endpoint — "and UCB gets there at 140. **30% fewer experiments.**"

Scroll to the metrics table:

> "And here's the metric that matters most for a real campaign: top-1% variants
> discovered. Random finds 4.7%. We find 15.5% — a 3.3× enrichment. You don't
> want one lucky hit, you want a shortlist of leads."

Scroll to the **seed-paired comparison**:

> "Because seeds are paired by design, we can test properly. Enrichment
> wins on ten seeds out of ten, p = 0.002. AUC, p = 0.027. Measurements
> saved, p = 0.031.
>
> And look at the last row — **best single variant found, p = 0.32. Not
> significant.** That's a max statistic over ten seeds; it's noisy, and we're
> not going to claim it. The result we stand behind is enrichment and
> efficiency."

That last beat is deliberate. Judges notice teams that mark their own limits.

---

## 2:40 — 3:00 · The result that doesn't flatter us, and the close

Switch the benchmark dropdown through **TEM-1** and **TPMT**.

> "We ran three assays, and the honest headline is a relationship, not a win.

Point at the three-assay table in the README, or just say the numbers:

> "Surrogate accuracy 0.56, 0.48, 0.35. Enrichment over random: 3.3×, 2.7×,
> 1.1×. **The benefit tracks how well the model can actually rank variants** —
> monotonically. Where the model knows something, steering by it pays off.
> Where it doesn't, the loop degenerates into an expensive random search, and
> on TPMT it buys nothing measurable.
>
> And two things that don't go our way. On TEM-1 our Bayesian learner is
> beaten by our own ESM baseline — the one that trains nothing. That baseline
> is the best method on two of the three assays. So the claim we can actually
> defend is 'model-guided selection beats random when the model has signal',
> not 'our learned surrogate beats everything'. We're not going to crown a
> winner off ten seeds."

Close:

> "So: a working closed loop on real experimental data, a 3.3× enrichment with
> 30% fewer measurements on the assay where the model learns, a monotone
> account of when that transfers and when it doesn't, and a baseline that
> beats us twice — reported rather than buried. 258 tests, 19 of them
> specifically pinning that the hidden labels stay hidden, because a benchmark
> that leaks is worth nothing."

---

## If you have 30 more seconds

Open the **Protein explorer**:

- Wild type vs. best discovered sequence, substitution highlighted in red.
- Predicted-vs-measured scatter, coloured by round — later rounds pull toward
  the diagonal.
- Which residue positions the search concentrated budget on.

---

## Likely questions

**"Isn't this just retrospective? Would it work prospectively?"**
It's retrospective by necessity — that's the only way to get ground truth for
every candidate and run a controlled comparison. The loop itself is unchanged
for a prospective campaign: swap `ExperimentalOracle.query` for a LIMS call.
What we've validated is the selection policy, which is the part that transfers.

**"How do we know the labels aren't leaking?"**
`CandidatePool` is a frozen dataclass with four fields and no fitness
attribute. The oracle keeps scores private and `query()` is the only path to a
number. A static test asserts the acquisition, surrogate and featurizer modules
never reference a label-reading API by name. 19 tests cover this.

**"Why is the GP better than the random forest if the forest predicts better?"**
It isn't, on point accuracy — the forest is ρ 0.41 vs the GP's 0.36, and that's
in the README. The GP is there for *calibrated uncertainty*, which is what UCB
and EI consume. On TPMT you can see greedy exploitation of a weak model do
worse than random while UCB degrades gracefully; that's the uncertainty
earning its place.

**"What did you take from EvoEmu?"**
The framing and the evolutionary search idea. EvoEmu ships no licence file, so
we copied no code — the ESM-2 proposal logic and elite-consensus search are
reimplemented with attribution in the module docstrings. Our contribution on top
is the trainable surrogate and the feedback loop: EvoEmu does inference-driven
search with no learning.

**"Why the 8M ESM-2?"**
So the whole thing runs on a laptop. It's one config line to swap. We'd expect a
larger checkpoint to raise surrogate accuracy — and per our own mechanism, that
should *increase* the active-learning advantage.

---

## Demo failure modes

| Problem | Fix |
|---|---|
| Benchmark tab is empty | `results/*_benchmark.json` missing — run `scripts/run_benchmark.py`, or re-pull the repo |
| Experiment stuck on "preparing" | Embeddings not cached for that assay; run `scripts/precompute_embeddings.py <assay>` beforehand |
| Slow first load | ESM-2 weights downloading from HuggingFace — do this once before the demo |
| Port 8000 busy | `--port 8001`, and the frontend dev proxy follows |

**Pre-demo checklist:** server running, embeddings cached for VIM-2, TEM-1 and
TPMT,
`results/` populated, Benchmark tab pre-opened in a second browser tab.
