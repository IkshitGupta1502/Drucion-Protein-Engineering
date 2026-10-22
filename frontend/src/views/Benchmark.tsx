import { useEffect, useMemo, useState } from 'react';
import {
  api,
  METHOD_COLORS,
  METHOD_LABELS,
  type BenchmarkPayload,
} from '../api';
import { Plot } from '../components/Plot';
import { bandTraces } from '../components/plotTraces';

export function Benchmark() {
  const [saved, setSaved] = useState<Record<string, BenchmarkPayload>>({});
  const [key, setKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Live benchmark job
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [seeds, setSeeds] = useState(5);

  useEffect(() => {
    api
      .precomputed()
      .then((r) => {
        setSaved(r.benchmarks);
        const keys = Object.keys(r.benchmarks);
        if (keys.length) setKey(keys[0]);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      try {
        const job = await api.benchmark(jobId);
        setJobProgress(job.progress);
        setJobStatus(job.status);
        if (job.status === 'done' && job.result) {
          const name = `${job.result.assay_id} (just run)`;
          setSaved((prev) => ({ ...prev, [name]: job.result as BenchmarkPayload }));
          setKey(name);
          setJobId(null);
        } else if (job.status === 'error') {
          setError(job.error ?? 'Benchmark failed');
          setJobId(null);
        }
      } catch {
        /* keep polling */
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [jobId]);

  const payload = key ? saved[key] : null;

  const startBenchmark = async () => {
    setError(null);
    try {
      const job = await api.createBenchmark({
        assay_id: payload?.assay_id ?? 'A4GRB6_PSEAI_Chen_2020',
        methods: ['random', 'evolutionary', 'greedy', 'ei', 'ucb'],
        n_seeds: seeds,
        loop: { n_initial: 50, batch_size: 10, n_rounds: 15 },
      });
      setJobId(job.id);
      setJobStatus('running');
      setJobProgress(0);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="empty">
          <span className="spinner" /> Loading saved benchmark results…
        </div>
      </div>
    );
  }

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="plain-explanation"><strong>Which search approach works best?</strong><p>These saved examples compare different ways of choosing variants with the same measurement budget. Higher scores found earlier are better. Results depend on the protein and settings; a guided search won’t always beat random sampling.</p><p className="note">These are example runs, separate from your current search. You can inspect them without starting any computation.</p></div>
      {error && <div className="alert err">{error}</div>}

      <div className="card">
        <div className="card-head" style={{ marginBottom: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div className="card-title" style={{ marginBottom: 0 }}>
              Saved example
            </div>
            {Object.keys(saved).length > 0 && (
              <select
                aria-label="Saved benchmark"
                value={key ?? ''}
                onChange={(e) => setKey(e.target.value)}
                style={{ width: 'auto' }}
              >
                {Object.keys(saved).map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            )}
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <label className="note" htmlFor="seeds">
              Repeat runs
            </label>
            <input
              id="seeds"
              type="number"
              min={1}
              max={20}
              value={seeds}
              onChange={(e) => setSeeds(Number(e.target.value))}
              style={{ width: 70 }}
            />
            <button className="btn" onClick={startBenchmark} disabled={!!jobId || !Number.isInteger(seeds) || seeds < 1 || seeds > 20}>
              {jobId ? 'Comparing…' : 'Run a new comparison'}
            </button>
          </div>
        </div>
        {jobId && (
          <div style={{ marginTop: 14 }}>
            <div className="note" style={{ marginBottom: 6 }}>
              {jobStatus} — {Math.round(jobProgress * 100)}% of runs complete. This executes
              every method across every seed under identical budgets; it takes a few minutes.
            </div>
            <div className="progress">
              <div className="progress-bar" style={{ width: `${jobProgress * 100}%` }} />
            </div>
          </div>
        )}
      </div>

      {!payload ? (
        <div className="card">
          <div className="empty">
            No saved results yet. Run <code>python scripts/run_benchmark.py</code> or press
            “Run a new comparison”.
          </div>
        </div>
      ) : (
        <BenchmarkReport payload={payload} />
      )}
    </div>
  );
}

function BenchmarkReport({ payload }: { payload: BenchmarkPayload }) {
  const traces = useMemo(() => {
    const out: Plotly.Data[] = [];
    for (const method of payload.methods) {
      const c = payload.curves[method];
      if (!c) continue;
      out.push(
        ...bandTraces(
          c.n_measured,
          c.mean,
          c.std,
          METHOD_LABELS[method] ?? method,
          METHOD_COLORS[method] ?? '#888',
        ),
      );
    }
    const x = payload.curves[payload.methods[0]]?.n_measured ?? [];
    out.push({
      x,
      y: x.map(() => payload.pool_stats.pool_max),
      mode: 'lines',
      type: 'scatter',
      line: { color: '#526459', width: 1.2, dash: 'dot' },
      name: `Pool maximum (${payload.pool_stats.pool_max.toFixed(2)})`,
      hoverinfo: 'skip',
    });
    out.push({
      x,
      y: x.map(() => payload.pool_stats.p99),
      mode: 'lines',
      type: 'scatter',
      line: { color: '#637069', width: 1, dash: 'dash' },
      name: `Pool 99th pct (${payload.pool_stats.p99.toFixed(2)})`,
      hoverinfo: 'skip',
    });
    return out;
  }, [payload]);

  const surrogateTraces = useMemo(() => {
    const out: Plotly.Data[] = [];
    for (const method of payload.methods) {
      const c = payload.surrogate_curves?.[method];
      if (!c) continue;
      out.push({
        x: c.n_measured,
        y: c.mean,
        mode: 'lines',
        type: 'scatter',
        name: METHOD_LABELS[method] ?? method,
        line: { color: METHOD_COLORS[method] ?? '#888', width: 2 },
        hovertemplate: '%{x} measurements<br>ρ = %{y:.3f}<extra></extra>',
      });
    }
    return out;
  }, [payload]);

  const nSeeds = payload.seeds.length;
  const eff = payload.efficiency;

  return (
    <>
      <div className="benchmark-context"><span><strong>{payload.methods.length}</strong> strategies</span><span><strong>{nSeeds}</strong> random seeds</span><span><strong>{payload.curves[payload.methods[0]]?.n_measured.slice(-1)[0] ?? '—'}</strong> measurements per run</span><span><strong>{String(payload.feature_info.featurizer ?? '—')}</strong> features</span></div>
      <div className="card">
        <div className="card-head">
          <div className="card-title">
            Best measured fitness vs. cumulative experimental evaluations
          </div>
          <span className="badge">
            {nSeeds} seeds · identical pool, budget and starting measurements
          </span>
        </div>
        <Plot
          height={440}
          data={traces}
          layout={{
            xaxis: { title: { text: 'Cumulative experimental measurements revealed' } },
            yaxis: { title: { text: 'Best measured fitness (DMS score)' } },
            margin: { l: 62, r: 20, t: 10, b: 70 },
          }}
        />
        <div className="note">
          Shaded bands are ±1 s.d. across {nSeeds} seeds. For each seed every method starts
          from the <em>same</em> randomly drawn initial measurements and is allowed the same
          number of oracle queries, so the only difference between curves is which variant
          each method chooses to measure next.
        </div>
      </div>

      {eff && (
        <div className="card">
          <div className="card-title">Experimental efficiency</div>
          <div className="note" style={{ marginBottom: 14 }}>
            Random selection spends its whole budget to reach a best measured fitness of{' '}
            <strong>{eff.target.toFixed(3)}</strong>. How many measurements does each method
            need to match that?
          </div>
          <div className="stats">
            {payload.methods.map((m) => {
              const e = eff.methods[m];
              if (!e) return null;
              const isBaseline = m === eff.baseline;
              return (
                <div className="stat" key={m}>
                  <div className="stat-label">
                    {METHOD_LABELS[m] ?? m}
                    {isBaseline && ' (baseline)'}
                  </div>
                  <div
                    className="stat-value"
                    style={{ color: METHOD_COLORS[m], fontSize: 20 }}
                  >
                    {isBaseline
                      ? e.final_value.toFixed(3)
                      : (e.budget_to_match ?? 'never')}
                  </div>
                  <div className="stat-sub">
                    {isBaseline
                      ? 'the level every other method is measured against'
                      : e.savings_fraction != null
                        ? `${Math.round(e.savings_fraction * 100)}% fewer experiments`
                        : 'did not reach the target within budget'}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <div className="card-title">Summary metrics (mean ± s.d. across seeds)</div>
          <span className="badge">pool max {payload.pool_stats.pool_max.toFixed(2)}</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Method</th>
                <th className="num">Best measured</th>
                <th className="num">Normalised AUC</th>
                <th className="num">Final regret</th>
                <th className="num">Top-1% found</th>
                <th className="num">Top-10% found</th>
                <th className="num">Evals to 99th pct</th>
                <th className="num">Surrogate ρ</th>
              </tr>
            </thead>
            <tbody>
              {payload.methods.map((m) => {
                const s = payload.summary[m];
                if (!s) return null;
                const evals = s.evals_to_threshold_mean;
                const hit = s.threshold_hit_rate ?? 0;
                return (
                  <tr key={m} className={m === 'ucb' ? 'highlight' : undefined}>
                    <td>
                      <span
                        className="swatch"
                        style={{ background: METHOD_COLORS[m], marginRight: 8 }}
                      />
                      {METHOD_LABELS[m] ?? m}
                    </td>
                    <td className="num">
                      {fmt(s.best_measured_mean)} ± {fmt(s.best_measured_std)}
                    </td>
                    <td className="num">
                      {fmt(s.auc_normalised_mean, 3)} ± {fmt(s.auc_normalised_std, 3)}
                    </td>
                    <td className="num">
                      {fmt(s.final_regret_mean)} ± {fmt(s.final_regret_std)}
                    </td>
                    <td className="num">{pct(s.top1pct_recall_mean)}</td>
                    <td className="num">{pct(s.top10pct_recall_mean)}</td>
                    <td className="num">
                      {evals != null ? `${Math.round(evals)}` : 'never'}
                      {evals != null && hit < 1 && (
                        <span style={{ color: 'var(--text-faint)' }}>
                          {' '}
                          ({Math.round(hit * 100)}%)
                        </span>
                      )}
                    </td>
                    <td className="num">
                      {s.surrogate_spearman_mean != null
                        ? fmt(s.surrogate_spearman_mean, 3)
                        : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="note" style={{ marginTop: 12 }}>
          Normalised AUC rescales the area under the best-so-far curve so 0 is the pool mean
          and 1 is the pool maximum; it rewards reaching high fitness early. Regret is pool
          maximum minus best discovered. Surrogate ρ is Spearman correlation on a fixed
          holdout that is excluded from the candidate pool entirely.
        </div>
      </div>

      {payload.paired && <PairedTable payload={payload} />}

      {surrogateTraces.length > 0 && (
        <div className="card">
          <div className="card-title">
            Does the surrogate get better as measurements accumulate?
          </div>
          <Plot
            height={320}
            data={surrogateTraces}
            layout={{
              xaxis: { title: { text: 'Training measurements' } },
              yaxis: { title: { text: 'Held-out Spearman ρ' } },
              margin: { l: 58, r: 20, t: 10, b: 70 },
            }}
          />
          <div className="note">
            Measured on a fixed random holdout that is never trained on, never selectable and
            never charged to budget — so this is a clean out-of-sample accuracy curve.
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">Run configuration &amp; honesty notes</div>
        <dl className="kv">
          <dt>Assay</dt>
          <dd>{payload.assay_id}</dd>
          <dt>Seeds</dt>
          <dd>{payload.seeds.join(', ')}</dd>
          <dt>Representation</dt>
          <dd>
            {String(payload.feature_info.featurizer)} → {String(payload.feature_info.n_features)} features
            {payload.feature_info.includes_zero_shot ? ' (incl. ESM-2 zero-shot score)' : ''}
          </dd>
          <dt>Evaluation holdout</dt>
          <dd>{String(payload.feature_info.holdout_size ?? 0)} variants</dd>
          <dt>Wall time</dt>
          <dd>{payload.wall_time_s.toFixed(0)} s</dd>
        </dl>
        {payload.notes.length > 0 && (
          <div className="alert warn" style={{ marginTop: 14 }}>
            <strong>Run notes:</strong>
            <ul style={{ margin: '6px 0 0', paddingLeft: 20 }}>
              {[...new Set(payload.notes)].map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </div>
        )}
        <div className="note" style={{ marginTop: 12 }}>
          Every number on this page comes from real ProteinGym measurements and is written to{' '}
          <code>results/</code> as CSV and JSON by <code>scripts/run_benchmark.py</code>.
          Results are reported as measured, including where a baseline beats the proposed
          method.
        </div>
      </div>
    </>
  );
}

function PairedTable({ payload }: { payload: BenchmarkPayload }) {
  const paired = payload.paired!;
  const metrics = [
    ['best_measured', 'Best measured'],
    ['auc_normalised', 'Normalised AUC'],
    ['top1pct_recall', 'Top-1% found'],
    ['evals_to_threshold', 'Evals to 99th pct'],
  ] as const;

  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">
          Seed-paired comparison against {METHOD_LABELS[paired.baseline] ?? paired.baseline}
        </div>
        <span className="badge">Wilcoxon signed-rank</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Metric</th>
              <th className="num">Seeds won</th>
              <th className="num">Mean difference</th>
              <th className="num">p-value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(paired.methods).flatMap(([method, entry]) =>
              metrics.map(([mkey, mlabel], i) => {
                const m = entry.metrics[mkey];
                if (!m) return null;
                const better = m.lower_is_better
                  ? m.mean_difference < 0
                  : m.mean_difference > 0;
                const sig = m.wilcoxon_p != null && m.wilcoxon_p < 0.05;
                return (
                  <tr key={`${method}-${mkey}`}>
                    {i === 0 ? (
                      <td rowSpan={metrics.length}>
                        <span
                          className="swatch"
                          style={{ background: METHOD_COLORS[method], marginRight: 8 }}
                        />
                        {METHOD_LABELS[method] ?? method}
                      </td>
                    ) : null}
                    <td>{mlabel}</td>
                    <td className="num">
                      {m.wins}/{m.n}
                      {m.ties > 0 && (
                        <span style={{ color: 'var(--text-faint)' }}> ({m.ties} tied)</span>
                      )}
                    </td>
                    <td
                      className="num"
                      style={{ color: better ? 'var(--measured)' : 'var(--danger)' }}
                    >
                      {m.mean_difference > 0 ? '+' : ''}
                      {m.mean_difference.toFixed(3)}
                    </td>
                    <td className="num" style={{ fontWeight: sig ? 700 : 400 }}>
                      {m.wilcoxon_p != null ? m.wilcoxon_p.toFixed(3) : '—'}
                      {sig && ' *'}
                    </td>
                  </tr>
                );
              }),
            )}
          </tbody>
        </table>
      </div>
      <div className="note" style={{ marginTop: 12 }}>
        Because every method starts each seed from identical measurements, seeds are paired
        and a paired test is the appropriate one — it removes the large seed-to-seed variance
        that makes the unpaired standard deviations look uninformative. With{' '}
        {payload.seeds.length} seeds these p-values are indicative rather than confirmatory,
        and are shown for every method including those that lose.
      </div>
    </div>
  );
}

function fmt(v: number | null | undefined, digits = 3): string {
  return v == null || Number.isNaN(v) ? '—' : v.toFixed(digits);
}

function pct(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(1)}%`;
}
