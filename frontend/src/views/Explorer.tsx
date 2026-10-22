import { useMemo } from 'react';
import { METHOD_LABELS, type ExperimentResults } from '../api';
import { Plot } from '../components/Plot';

export function Explorer({ results, onSetup }: { results: ExperimentResults | null; onSetup: () => void }) {
  if (!results || !results.wild_type || !results.best_sequence) {
    return (
      <div className="card">
        <div className="empty">
          <div className="empty-glyph" aria-hidden="true">⌁</div><h2>Every variant has a story.</h2><p>Start an experiment to compare sequences, explore mutations, and inspect measured results.</p><button className="btn btn-primary" onClick={onSetup}>Configure an experiment →</button>
        </div>
      </div>
    );
  }

  const mutatedPositions = new Set(results.differences.map((d) => d.position));

  return (
    <div className="grid" style={{ gap: 16 }}>
      <BestVariant results={results} />

      <div className="grid chart-grid">
        <PredictionVsMeasurement results={results} />
        <MutationMap results={results} />
      </div>

      <div className="card">
        <div className="card-head">
          <div className="card-title">Sequence comparison</div>
          <span className="legend-key">
            <span className="swatch" style={{ background: 'var(--danger)' }} /> substituted
            residue
          </span>
        </div>
        <div style={{ marginBottom: 6 }} className="note">
          Wild type ({results.wild_type.length} aa)
        </div>
        <div className="seq">
          {splitHighlight(results.wild_type, mutatedPositions)}
        </div>
        <div style={{ margin: '14px 0 6px' }} className="note">
          Best discovered variant — <code>{results.best_mutant}</code>
        </div>
        <div className="seq">
          {splitHighlight(results.best_sequence, mutatedPositions)}
        </div>
      </div>
    </div>
  );
}

function BestVariant({ results }: { results: ExperimentResults }) {
  // Find the round in which the best variant was measured, and what the
  // surrogate predicted for it at that moment.
  const found = useMemo(() => {
    for (const r of results.rounds ?? []) {
      for (const v of r.selected) {
        if (v.mutant === results.best_mutant) return { round: r.round, variant: v };
      }
    }
    return null;
  }, [results.rounds, results.best_mutant]);

  const stats = results.pool_stats;

  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">Best discovered variant</div>
        <span className="badge">{METHOD_LABELS[results.acquisition] ?? results.acquisition}</span>
      </div>
      <div className="stats">
        <div className="stat">
          <div className="stat-label">Variant</div>
          <div className="stat-value" style={{ fontSize: 20 }}>
            {results.best_mutant ?? '—'}
          </div>
          <div className="stat-sub">
            {results.differences.length} substitution
            {results.differences.length === 1 ? '' : 's'} from wild type
          </div>
        </div>
        <div className="stat">
          <div className="stat-label">Measured fitness</div>
          <div className="stat-value measured">
            {results.best_measured != null ? results.best_measured.toFixed(3) : '—'}
          </div>
          <div className="stat-sub">real DMS measurement</div>
        </div>
        <div className="stat">
          <div className="stat-label">Model prediction</div>
          <div className="stat-value predicted">
            {found?.variant.predicted_fitness != null
              ? found.variant.predicted_fitness.toFixed(3)
              : '—'}
          </div>
          <div className="stat-sub">
            {found?.variant.predicted_uncertainty != null
              ? `± ${found.variant.predicted_uncertainty.toFixed(3)} before measuring`
              : 'prediction before measurement'}
          </div>
        </div>
        <div className="stat">
          <div className="stat-label">Found at round</div>
          <div className="stat-value">{found ? found.round : '0'}</div>
          <div className="stat-sub">
            {found ? 'during active learning' : 'in the initial random design'}
          </div>
        </div>
        <div className="stat">
          <div className="stat-label">Measurements spent</div>
          <div className="stat-value">{results.n_measured}</div>
          <div className="stat-sub">of {results.budget} budget</div>
        </div>
        {stats && (
          <div className="stat">
            <div className="stat-label">Percentile in assay</div>
            <div className="stat-value">
              {results.best_measured != null && stats.pool_max > stats.pool_min
                ? `${percentileLabel(results.best_measured, stats)}`
                : '—'}
            </div>
            <div className="stat-sub">pool max {stats.pool_max.toFixed(2)}</div>
          </div>
        )}
      </div>
      <div className="note" style={{ marginTop: 14 }}>
        Green values are real experimental measurements from the published assay. Purple
        values are model predictions made before the measurement was revealed. They are never
        mixed.
      </div>
    </div>
  );
}

function PredictionVsMeasurement({ results }: { results: ExperimentResults }) {
  const points = useMemo(() => {
    const pred: number[] = [];
    const meas: number[] = [];
    const text: string[] = [];
    const round: number[] = [];
    for (const r of results.rounds ?? []) {
      for (const v of r.selected) {
        if (v.predicted_fitness == null) continue;
        pred.push(v.predicted_fitness);
        meas.push(v.measured_fitness);
        text.push(v.mutant);
        round.push(r.round);
      }
    }
    return { pred, meas, text, round };
  }, [results]);

  if (!points.pred.length) {
    return (
      <div className="card">
        <div className="card-title">Predicted vs. measured</div>
        <div className="empty">No surrogate predictions recorded for this run.</div>
      </div>
    );
  }

  const lo = Math.min(...points.pred, ...points.meas);
  const hi = Math.max(...points.pred, ...points.meas);

  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">Predicted vs. measured fitness</div>
        <span className="badge">{points.pred.length} selected variants</span>
      </div>
      <Plot
        height={330}
        data={[
          {
            x: [lo, hi],
            y: [lo, hi],
            mode: 'lines',
            type: 'scatter',
            line: { color: '#637069', width: 1, dash: 'dash' },
            name: 'perfect prediction',
            hoverinfo: 'skip',
          },
          {
            x: points.pred,
            y: points.meas,
            mode: 'markers',
            type: 'scatter',
            text: points.text,
            marker: {
              size: 9,
              color: points.round,
              colorscale: 'Viridis',
              showscale: true,
              colorbar: { title: { text: 'round' }, thickness: 10, len: 0.8 },
              line: { color: '#0b0f16', width: 1 },
            },
            name: 'variants',
            hovertemplate:
              '<b>%{text}</b><br>predicted %{x:.3f}<br>measured %{y:.3f}<extra></extra>',
          },
        ]}
        layout={{
          xaxis: { title: { text: 'Predicted fitness (before measuring)' } },
          yaxis: { title: { text: 'Measured fitness (real DMS)' } },
          showlegend: false,
          margin: { l: 58, r: 10, t: 10, b: 50 },
        }}
      />
      <div className="note">
        Points on the dashed line are perfect predictions. Later rounds (lighter) should sit
        closer to it as the surrogate learns from accumulated measurements.
      </div>
    </div>
  );
}

function MutationMap({ results }: { results: ExperimentResults }) {
  const data = useMemo(() => {
    const byPosition = new Map<number, { best: number; count: number }>();
    for (const r of results.rounds ?? []) {
      for (const v of r.selected) {
        for (const p of v.positions) {
          const cur = byPosition.get(p);
          if (!cur) byPosition.set(p, { best: v.measured_fitness, count: 1 });
          else {
            cur.count += 1;
            cur.best = Math.max(cur.best, v.measured_fitness);
          }
        }
      }
    }
    const positions = [...byPosition.keys()].sort((a, b) => a - b);
    return {
      positions,
      best: positions.map((p) => byPosition.get(p)!.best),
      count: positions.map((p) => byPosition.get(p)!.count),
    };
  }, [results]);

  if (!data.positions.length) {
    return (
      <div className="card">
        <div className="card-title">Explored positions</div>
        <div className="empty">No rounds recorded yet.</div>
      </div>
    );
  }

  const bestPositions = new Set(results.differences.map((d) => d.position));

  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">Where the search looked</div>
        <span className="badge">{data.positions.length} positions probed</span>
      </div>
      <Plot
        height={330}
        data={[
          {
            x: data.positions,
            y: data.best,
            mode: 'markers',
            type: 'scatter',
            text: data.positions.map(
              (p, i) => `position ${p} · ${data.count[i]} variant(s) measured`,
            ),
            marker: {
              size: data.positions.map((p) => (bestPositions.has(p) ? 14 : 8)),
              color: data.positions.map((p) =>
                bestPositions.has(p) ? '#147d59' : '#286f8d',
              ),
              line: { color: '#0b0f16', width: 1 },
            },
            hovertemplate: '%{text}<br>best measured here = %{y:.3f}<extra></extra>',
          },
        ]}
        layout={{
          xaxis: { title: { text: 'Residue position' } },
          yaxis: { title: { text: 'Best measured fitness at position' } },
          showlegend: false,
          margin: { l: 58, r: 10, t: 10, b: 50 },
        }}
      />
      <div className="note">
        Green marks the position(s) mutated in the best variant found. Clustering indicates
        the surrogate concentrating the budget on a promising region of the protein.
      </div>
    </div>
  );
}

function splitHighlight(seq: string, positions: Set<number>) {
  const out: React.ReactNode[] = [];
  let buffer = '';
  for (let i = 0; i < seq.length; i += 1) {
    const pos = i + 1;
    if (positions.has(pos)) {
      if (buffer) {
        out.push(buffer);
        buffer = '';
      }
      out.push(
        <span className="mut" key={pos} title={`position ${pos}`}>
          {seq[i]}
        </span>,
      );
    } else {
      buffer += seq[i];
    }
  }
  if (buffer) out.push(buffer);
  return out;
}

function percentileLabel(value: number, stats: Record<string, number>): string {
  if (value >= stats.p99) return '≥ 99th';
  if (value >= stats.p90) return '≥ 90th';
  if (value >= stats.pool_mean) return '> mean';
  return '< mean';
}
