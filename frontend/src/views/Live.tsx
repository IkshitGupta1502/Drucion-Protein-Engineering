import { BudgetEstimate } from '../components/BudgetEstimate';
import type { ExperimentResults } from '../api';
import { useEffect, useMemo, useRef, useState } from 'react';
import { METHOD_LABELS, type ExperimentState, type RoundRecord } from '../api';
import { Plot } from '../components/Plot';
import { BenchLoop } from '../components/BenchLoop';

interface Props {
  budgetEstimate?: ExperimentResults['budget_estimate'];
  onSetup: () => void;
  onResults: () => void;
  experiment: ExperimentState | null;
  rounds: RoundRecord[];
  busy: boolean;
  onStep: () => void;
  onRun: () => void;
  onPause: () => void;
  onReset: () => void;
  embedProgress: { done: number; total: number } | null;
}

export function Live({
  budgetEstimate,
  onSetup,
  onResults,
  experiment,
  rounds,
  busy,
  onStep,
  onRun,
  onPause,
  onReset,
  embedProgress,
}: Props) {
  if (!experiment) {
    return (
      <div className="card">
        <div className="empty">
          <div className="empty-glyph" aria-hidden="true">↗</div><h2>Your next discovery starts with a run.</h2><p>Try a ready-made dataset. Drucion will choose candidates, check their results, and show you what it learns.</p><button className="btn btn-primary" onClick={onSetup}>Start my first search →</button>
        </div>
      </div>
    );
  }

  const latest = rounds.length ? rounds[rounds.length - 1] : null;
  const preparing = experiment.status === 'preparing' || experiment.status === 'created';
  const running = experiment.status === 'running';
  const done = experiment.status === 'done';
  const curve = experiment.learning_curve;
  const failed = experiment.status === 'error';
  const initialBest = curve?.best_measured[0];
  const improvement = initialBest != null && experiment.best_measured != null ? experiment.best_measured - initialBest : null;
  const explanation = failed
    ? { title: 'Your search needs attention.', text: 'The search stopped before it could finish. Review the error below, then try again or choose another dataset.' }
    : preparing
      ? { title: 'Getting to know your protein.', text: 'We’re preparing protein features and checking the starting measurements. The first run on a dataset can take longer; you can stay on this page while it prepares.' }
      : done
        ? { title: 'Your search is complete.', text: `${experiment.n_measured} measurements checked across ${experiment.round} learning rounds. ${improvement != null && improvement > 0 ? `The best score improved by ${improvement.toFixed(3)} from your starting sample.` : 'No better score was found than the best variant in your starting sample. That can happen with a limited search.'}` }
        : running || busy
          ? { title: 'Choosing, checking, and learning.', text: 'Drucion is using the results so far to choose the next batch. It then reveals those measurements and updates its model for the next round.' }
          : { title: experiment.round ? 'Ready for your next round.' : 'Your starting sample is ready.', text: `We’ve checked ${experiment.n_measured} variants. Continue the search to complete the remaining rounds, or check one batch at a time to see each decision.` };


  return (
    <div className="grid" style={{ gap: 16 }}>
      <section className={`search-guide ${done ? 'complete' : ''}`} aria-labelledby="search-status">
        <div className="guide-copy" role="status"><span className="eyebrow">{done ? 'YOUR RESULTS ARE READY' : failed ? 'SEARCH PAUSED BY AN ERROR' : 'WHAT’S HAPPENING NOW'}</span><h2 id="search-status">{explanation.title}</h2><p>{explanation.text}</p></div>
        {done && <button className="btn btn-primary" onClick={onResults}>View my results →</button>}
        {failed && <button className="btn" onClick={onSetup}>Change search settings</button>}
        <ol className="search-stages" aria-label="Search progress"><li className={!preparing ? 'reached' : 'current'}><span>1</span>Prepare your starting sample</li><li className={done ? 'reached' : running || busy ? 'current' : ''}><span>2</span>Learn across {experiment.n_rounds} rounds</li><li className={done ? 'reached' : ''}><span>3</span>Review your candidates</li></ol>
      </section>
      <ControlBar
        experiment={experiment}
        busy={busy}
        preparing={preparing}
        running={running}
        done={done}
        onStep={onStep}
        onRun={onRun}
        onPause={onPause}
        onReset={onReset}
      />

      {experiment.error && <div className="alert err">{experiment.error}</div>}

      {preparing && (
        <div className="card">
          <div className="card-title">Preparing</div>
          <div style={{ marginBottom: 10 }}>
            <span className="spinner" />{' '}
            {embedProgress
              ? `Preparing protein features — ${embedProgress.done.toLocaleString()} / ${embedProgress.total.toLocaleString()} sequences`
              : 'Checking your dataset and starting measurements…'}
          </div>
          {embedProgress && (
            <div className="progress">
              <div
                className="progress-bar"
                style={{ width: `${(embedProgress.done / embedProgress.total) * 100}%` }}
              />
            </div>
          )}
        </div>
      )}

      <StatRow experiment={experiment} latest={latest} />
      <BudgetEstimate estimate={budgetEstimate} />

      <div className="grid chart-grid">
        <div className="card">
          <div className="card-head">
            <div className="card-title">Is the search finding better candidates?</div>
            <div>
              <span className="legend-key">
                <span className="swatch" style={{ background: '#147d59' }} /> measured
              </span>
            </div>
          </div>
          {curve && curve.n_measured.length > 1 ? (
            <Plot
              height={300}
              data={[
                {
                  x: curve.n_measured,
                  y: curve.best_measured,
                  mode: 'lines+markers',
                  type: 'scatter',
                  line: { color: '#147d59', width: 2.4, shape: 'hv' },
                  marker: { size: 6, color: '#147d59' },
                  name: 'Best measured',
                  hovertemplate: '%{x} measurements<br>best = %{y:.3f}<extra></extra>',
                },
              ]}
              layout={{
                xaxis: { title: { text: 'Cumulative measurements' } },
                yaxis: { title: { text: 'Best measured score' } },
                showlegend: false,
              }}
            />
          ) : (
            <div className="empty">Run a round to see the learning curve.</div>
          )}
        </div>

        <div className="card">
          <div className="card-head">
            <div className="card-title">How well is the model learning?</div>
            <span className="badge">Checked on unseen variants</span>
          </div>
          {rounds.some((r) => r.holdout_spearman != null) ? (
            <Plot
              height={300}
              data={[
                {
                  x: rounds.map((r) => r.n_measured - r.selected.length),
                  y: rounds.map((r) => r.holdout_spearman),
                  mode: 'lines+markers',
                  type: 'scatter',
                  line: { color: '#7659a5', width: 2.2 },
                  marker: { size: 6, color: '#7659a5' },
                  hovertemplate:
                    '%{x} training measurements<br>Spearman ρ = %{y:.3f}<extra></extra>',
                },
              ]}
              layout={{
                xaxis: { title: { text: 'Training measurements' } },
                yaxis: { title: { text: 'Ranking accuracy (−1 to 1)' } },
                showlegend: false,
              }}
            />
          ) : (
            <div className="empty">
              Run a round to evaluate the model against the held-out variants.
            </div>
          )}
        </div>
      </div>

      <div className="plain-explanation"><strong>How to read your progress</strong><p>A higher measured score means better performance in this dataset. The first chart tracks your best result so far. The second checks how well the model ranks variants it hasn’t learned from; closer to 1 is better. Neither chart guarantees the next batch will improve.</p></div>
      <RoundPanel rounds={rounds} />
      <BenchLoop experiment={experiment} />
    </div>
  );
}

function ControlBar({
  experiment,
  busy,
  preparing,
  running,
  done,
  onStep,
  onRun,
  onPause,
  onReset,
}: {
  experiment: ExperimentState;
  busy: boolean;
  preparing: boolean;
  running: boolean;
  done: boolean;
  onStep: () => void;
  onRun: () => void;
  onPause: () => void;
  onReset: () => void;
}) {
  const badgeClass =
    experiment.status === 'error'
      ? 'err'
      : done
        ? 'ok'
        : running || preparing
          ? 'run'
          : 'warn';
  return (
    <div className="card">
      <div className="card-head" style={{ marginBottom: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <span className={`badge ${badgeClass}`}>{preparing ? 'Preparing' : done ? 'Complete' : running || busy ? 'Searching' : experiment.status === 'error' ? 'Needs attention' : 'Ready to continue'}</span>
          <strong>{METHOD_LABELS[experiment.acquisition] ?? experiment.acquisition}</strong>
          <span className="note">
            Round {experiment.round} of {experiment.n_rounds} ·{' '}
            {experiment.n_measured}/{experiment.budget} measurements
          </span>
        </div>
        <div className="btn-row">
          <button className="btn" onClick={onStep} disabled={busy || preparing || running || done || experiment.status === 'error'}>
            Check one batch
          </button>
          {running ? (
            <button className="btn" onClick={onPause}>
              Pause after this round
            </button>
          ) : (
            <button
              className="btn btn-primary"
              onClick={onRun}
              disabled={busy || preparing || done || experiment.status === 'error'}
            >
              Continue search
            </button>
          )}
          <button className="btn" onClick={onReset} disabled={preparing || running || busy}>
            Start over
          </button>
        </div>
      </div>
      <div className="progress" style={{ marginTop: 14 }}>
        <div
          className="progress-bar"
          style={{ width: `${(experiment.n_measured / Math.max(experiment.budget, 1)) * 100}%` }}
        />
      </div>
    </div>
  );
}

function StatRow({
  experiment,
  latest,
}: {
  experiment: ExperimentState;
  latest: RoundRecord | null;
}) {
  const improved = useRef<number | null>(null);
  const [flash, setFlash] = useState(false);
  useEffect(() => {
    const best = experiment.best_measured ?? null;
    if (improved.current != null && best != null && best > improved.current) {
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 1200);
      return () => clearTimeout(t);
    }
    improved.current = best;
  }, [experiment.best_measured]);

  const meanUncertainty = latest?.selected.length
    ? latest.selected.reduce((s, v) => s + (v.predicted_uncertainty ?? 0), 0) /
      latest.selected.length
    : null;

  return (
    <div className="stats">
      <div className={`stat ${flash ? 'flash' : ''}`}>
        <div className="stat-label">Best score found</div>
        <div className="stat-value measured">
          {experiment.best_measured != null ? experiment.best_measured.toFixed(3) : '—'}
        </div>
        <div className="stat-sub">{experiment.best_mutant ?? 'no variant yet'}</div>
      </div>
      <div className="stat">
        <div className="stat-label">Variants checked</div>
        <div className="stat-value">{experiment.n_measured}</div>
        <div className="stat-sub">of {experiment.budget} budget</div>
      </div>
      <div className="stat">
        <div className="stat-label">Round</div>
        <div className="stat-value">
          {experiment.round}
          <span style={{ color: 'var(--text-faint)', fontSize: 15 }}>/{experiment.n_rounds}</span>
        </div>
        <div className="stat-sub">{experiment.pool_size?.toLocaleString()} candidates in pool</div>
      </div>
      <div className="stat">
        <div className="stat-label">Model uncertainty</div>
        <div className="stat-value predicted">
          {meanUncertainty != null ? meanUncertainty.toFixed(3) : '—'}
        </div>
        <div className="stat-sub">uncertainty on last batch</div>
      </div>
      <div className="stat">
        <div className="stat-label">Ranking accuracy</div>
        <div className="stat-value">
          {latest?.holdout_spearman != null ? latest.holdout_spearman.toFixed(3) : '—'}
        </div>
        <div className="stat-sub">
          {latest?.surrogate_info && 'n_train' in latest.surrogate_info
            ? `trained on ${String(latest.surrogate_info.n_train)}`
            : 'unseen variants · closer to 1 is better'}
        </div>
      </div>
    </div>
  );
}

function RoundPanel({ rounds }: { rounds: RoundRecord[] }) {
  const [selected, setSelected] = useState<number | null>(null);
  const shown = useMemo(() => {
    if (!rounds.length) return null;
    const idx = selected ?? rounds.length - 1;
    return rounds[Math.min(idx, rounds.length - 1)];
  }, [rounds, selected]);

  if (!shown) {
    return (
      <div className="card">
        <div className="card-title">Your latest candidates</div>
        <div className="empty">
          Your candidates will appear after the first round. Each row shows what the model expected and the actual result it revealed.
        </div>
      </div>
    );
  }

  const bestInRound = Math.max(...shown.selected.map((v) => v.measured_fitness));

  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">
          Round {shown.round} — what we expected and what we found
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <span className="legend-key">
            <span className="swatch" style={{ background: '#7659a5' }} /> model prediction
          </span>
          <span className="legend-key">
            <span className="swatch" style={{ background: '#147d59' }} /> real measurement
          </span>
          <select
            aria-label="Selected round"
            value={shown.round}
            onChange={(e) => {
              const r = Number(e.target.value);
              setSelected(rounds.findIndex((x) => x.round === r));
            }}
            style={{ width: 'auto' }}
          >
            {rounds.map((r) => (
              <option key={r.round} value={r.round}>
                Round {r.round}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="table-wrap scroll-y">
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th className="num">Position</th>
              <th className="num" title="The priority assigned by the search approach; it is not a measured result.">Selection priority</th>
              <th className="num">Expected score</th>
              <th className="num">Uncertainty</th>
              <th className="num">Actual score</th>
              <th className="num">Difference</th>
            </tr>
          </thead>
          <tbody>
            {shown.selected.map((v) => {
              const err =
                v.predicted_fitness != null ? v.measured_fitness - v.predicted_fitness : null;
              return (
                <tr
                  key={v.index}
                  className={v.measured_fitness === bestInRound ? 'best-row' : undefined}
                >
                  <td className="mono">{v.mutant}</td>
                  <td className="num">{v.positions.join(', ')}</td>
                  <td className="num">{v.acquisition_score.toFixed(3)}</td>
                  <td className="num predicted">
                    {v.predicted_fitness != null ? v.predicted_fitness.toFixed(3) : '—'}
                  </td>
                  <td className="num predicted">
                    {v.predicted_uncertainty != null ? v.predicted_uncertainty.toFixed(3) : '—'}
                  </td>
                  <td className="num measured">{v.measured_fitness.toFixed(3)}</td>
                  <td className="num" style={{ color: 'var(--text-faint)' }}>
                    {err != null ? (err > 0 ? '+' : '') + err.toFixed(3) : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="note" style={{ marginTop: 12 }}>
        Predictions were made <em>before</em> these measurements were revealed. If you run
        another round, the model learns from all {shown.n_measured} measurements before
        choosing the next batch.
        {shown.proposal_report && (
          <>
            {' '}
            This method also generated{' '}
            {String(shown.proposal_report.n_proposed ?? 0)} novel variants, of which{' '}
            {String(shown.proposal_report.n_unvalidated ?? 0)} are not in the measured assay
            and are therefore left unvalidated — they are never assigned an invented score.
          </>
        )}
      </div>
    </div>
  );
}
