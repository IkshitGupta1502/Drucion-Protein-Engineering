import { useEffect, useMemo, useRef, useState } from 'react';
import { METHOD_LABELS, type ExperimentState, type RoundRecord } from '../api';
import { Plot } from '../components/Plot';

interface Props {
  onSetup: () => void;
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
  onSetup,
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
          <div className="empty-glyph" aria-hidden="true">↗</div><h2>Your next discovery starts with a run.</h2><p>Choose an assay and a measurement budget to follow the model as it learns.</p><button className="btn btn-primary" onClick={onSetup}>Configure an experiment →</button>
        </div>
      </div>
    );
  }

  const latest = rounds.length ? rounds[rounds.length - 1] : null;
  const preparing = experiment.status === 'preparing' || experiment.status === 'created';
  const running = experiment.status === 'running';
  const done = experiment.status === 'done';
  const curve = experiment.learning_curve;

  return (
    <div className="grid" style={{ gap: 16 }}>
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
              ? `Computing ESM-2 embeddings for the candidate pool — ${embedProgress.done.toLocaleString()} / ${embedProgress.total.toLocaleString()} sequences`
              : 'Loading and validating the assay…'}
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

      <div className="grid chart-grid">
        <div className="card">
          <div className="card-head">
            <div className="card-title">Best measured fitness vs. budget</div>
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
                yaxis: { title: { text: 'Best measured DMS score' } },
                showlegend: false,
              }}
            />
          ) : (
            <div className="empty">Run a round to see the learning curve.</div>
          )}
        </div>

        <div className="card">
          <div className="card-head">
            <div className="card-title">Surrogate accuracy per round</div>
            <span className="badge">held-out set</span>
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
                yaxis: { title: { text: 'Spearman ρ' } },
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

      <RoundPanel rounds={rounds} />
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
          <span className={`badge ${badgeClass}`}>{experiment.status}</span>
          <strong>{METHOD_LABELS[experiment.acquisition] ?? experiment.acquisition}</strong>
          <span className="note">
            {experiment.assay_id} · round {experiment.round}/{experiment.n_rounds} ·{' '}
            {experiment.n_measured}/{experiment.budget} measurements
            {experiment.surrogate_kind && <> · surrogate: {experiment.surrogate_kind}</>}
          </span>
        </div>
        <div className="btn-row">
          <button className="btn" onClick={onStep} disabled={busy || preparing || running || done}>
            Step one round
          </button>
          {running ? (
            <button className="btn" onClick={onPause}>
              Pause
            </button>
          ) : (
            <button
              className="btn btn-primary"
              onClick={onRun}
              disabled={busy || preparing || done}
            >
              Run to completion
            </button>
          )}
          <button className="btn" onClick={onReset} disabled={preparing}>
            Reset
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
        <div className="stat-label">Best measured fitness</div>
        <div className="stat-value measured">
          {experiment.best_measured != null ? experiment.best_measured.toFixed(3) : '—'}
        </div>
        <div className="stat-sub">{experiment.best_mutant ?? 'no variant yet'}</div>
      </div>
      <div className="stat">
        <div className="stat-label">Measurements spent</div>
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
        <div className="stat-label">Mean predicted s.d.</div>
        <div className="stat-value predicted">
          {meanUncertainty != null ? meanUncertainty.toFixed(3) : '—'}
        </div>
        <div className="stat-sub">uncertainty on last batch</div>
      </div>
      <div className="stat">
        <div className="stat-label">Surrogate ρ (holdout)</div>
        <div className="stat-value">
          {latest?.holdout_spearman != null ? latest.holdout_spearman.toFixed(3) : '—'}
        </div>
        <div className="stat-sub">
          {latest?.surrogate_info && 'n_train' in latest.surrogate_info
            ? `trained on ${String(latest.surrogate_info.n_train)}`
            : 'held-out Spearman'}
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
        <div className="card-title">Selected variants</div>
        <div className="empty">
          Step the loop to see which variants the model chose, what it predicted, and what the
          real measurement turned out to be.
        </div>
      </div>
    );
  }

  const bestInRound = Math.max(...shown.selected.map((v) => v.measured_fitness));

  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">
          Round {shown.round} — selected variants and revealed measurements
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
              <th className="num">Acquisition</th>
              <th className="num">Predicted fitness</th>
              <th className="num">Predicted s.d.</th>
              <th className="num">Measured fitness</th>
              <th className="num">Error</th>
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
        Predictions were made <em>before</em> these measurements were revealed. After this
        round the surrogate is retrained on all {shown.n_measured} measurements, which is what
        changes the next round’s choices.
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
