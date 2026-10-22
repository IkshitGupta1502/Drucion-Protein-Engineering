import { useEffect, useMemo, useState } from 'react';
import { api, type AssayDetail, type AssaySummary } from '../api';

import { type ExperimentSpec } from '../experimentSpec';

/** Assays we have precomputed ESM-2 embeddings for; others run slower. */
const FEATURED = ['A4GRB6_PSEAI_Chen_2020', 'TPMT_HUMAN_Matreyek_2018', 'BLAT_ECOLX_Firnberg_2014'];

interface Props {
  spec: ExperimentSpec;
  onChange: (spec: ExperimentSpec) => void;
  onStart: () => void;
  starting: boolean;
  error: string | null;
}

export function Setup({ spec, onChange, onStart, starting, error }: Props) {
  const [assays, setAssays] = useState<AssaySummary[]>([]);
  const [detail, setDetail] = useState<AssayDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    api
      .assays()
      .then(setAssays)
      .catch((e: Error) => setListError(e.message));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoadingDetail(true);
    setDetail(null);
    setDetailError(null);
    api
      .assay(spec.assay_id, true)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((e: Error) => { if (!cancelled) setDetailError(e.message); })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });
    return () => {
      cancelled = true;
    };
  }, [spec.assay_id]);

  const sorted = useMemo(() => {
    const featured = assays.filter((a) => FEATURED.includes(a.assay_id));
    const rest = assays
      .filter((a) => !FEATURED.includes(a.assay_id))
      .sort((a, b) => a.assay_id.localeCompare(b.assay_id));
    return { featured, rest };
  }, [assays]);

  const set = (patch: Partial<ExperimentSpec>) => onChange({ ...spec, ...patch });
  const setLoop = (patch: Partial<ExperimentSpec['loop']>) =>
    onChange({ ...spec, loop: { ...spec.loop, ...patch } });

  const budget = spec.loop.n_initial + spec.loop.batch_size * spec.loop.n_rounds;
  const poolSize = detail?.pool_size ?? 0;
  const validNumbers = Number.isInteger(spec.loop.n_initial) && spec.loop.n_initial >= 2 && spec.loop.n_initial <= 2000 && Number.isInteger(spec.loop.batch_size) && spec.loop.batch_size >= 1 && spec.loop.batch_size <= 500 && Number.isInteger(spec.loop.n_rounds) && spec.loop.n_rounds >= 1 && spec.loop.n_rounds <= 200 && Number.isInteger(spec.loop.seed) && spec.loop.seed >= 0 && spec.loop.seed <= 10000 && spec.loop.ucb_beta >= 0 && spec.loop.ucb_beta <= 20;
  const overBudget = poolSize > 0 && budget >= poolSize;

  return (
    <div className="grid setup-grid">
      <div className="setup-controls">
        <div className="card">
          <div className="section-heading"><span className="step-label">02</span><div><h2>Configure your run</h2><p>Define how the model explores.</p></div></div>

          {listError && <div className="alert err">Could not load assays: {listError}</div>}
          {error && <div className="alert err" role="alert">{error}</div>}
          {detailError && <div className="alert err" role="alert">Could not load assay: {detailError}</div>}

          <div className="field">
            <label htmlFor="assay">ProteinGym assay</label>
            <select
              id="assay"
              value={spec.assay_id}
              onChange={(e) => set({ assay_id: e.target.value })}
            >
              <optgroup label="Featured (embeddings precomputed)">
                {sorted.featured.map((a) => (
                  <option key={a.assay_id} value={a.assay_id}>
                    {a.assay_id} — {a.molecule_name}
                  </option>
                ))}
              </optgroup>
              <optgroup label={`All available (${sorted.rest.length})`}>
                {sorted.rest.map((a) => (
                  <option key={a.assay_id} value={a.assay_id}>
                    {a.assay_id} — {a.molecule_name}
                  </option>
                ))}
              </optgroup>
            </select>
            <div className="field-hint">
              Published experimental data. Featured assays have cached protein features.
            </div>
          </div>

          <div className="field">
            <label htmlFor="acq">Selection method</label>
            <select
              id="acq"
              value={spec.loop.acquisition}
              onChange={(e) =>
                setLoop({ acquisition: e.target.value as ExperimentSpec['loop']['acquisition'] })
              }
            >
              <option value="ucb">Bayesian active learning — UCB</option>
              <option value="ei">Bayesian active learning — Expected Improvement</option>
              <option value="greedy">Greedy surrogate (pure exploitation)</option>
              <option value="evolutionary">ESM-guided evolutionary (no surrogate)</option>
              <option value="random">Random selection (baseline)</option>
            </select>
          </div>

          <div className="row">
            <div className="field">
              <label htmlFor="init">Initial measurements</label>
              <input
                id="init"
                type="number"
                min={2}
                max={2000}
                value={spec.loop.n_initial}
                onChange={(e) => setLoop({ n_initial: Number(e.target.value) })}
              />
            </div>
            <div className="field">
              <label htmlFor="batch">Batch size per round</label>
              <input
                id="batch"
                type="number"
                min={1}
                max={500}
                value={spec.loop.batch_size}
                onChange={(e) => setLoop({ batch_size: Number(e.target.value) })}
              />
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label htmlFor="rounds">Rounds</label>
              <input
                id="rounds"
                type="number"
                min={1}
                max={200}
                value={spec.loop.n_rounds}
                onChange={(e) => setLoop({ n_rounds: Number(e.target.value) })}
              />
            </div>
            <div className="field">
              <label htmlFor="seed">Random seed</label>
              <input
                id="seed"
                type="number"
                min={0}
                max={10000}
                value={spec.loop.seed}
                onChange={(e) => setLoop({ seed: Number(e.target.value) })}
              />
            </div>
          </div>

          <details className="advanced-settings"><summary>Model & exploration settings<span>Advanced</span></summary><div className="advanced-content">
          <div className="row">
            <div className="field">
              <label htmlFor="feat">Protein representation</label>
              <select
                id="feat"
                value={spec.featurizer}
                onChange={(e) =>
                  set({ featurizer: e.target.value as ExperimentSpec['featurizer'] })
                }
              >
                <option value="esm2">ESM-2 site-resolved</option>
                <option value="esm2_mean">ESM-2 mean-pooled</option>
                <option value="onehot">Mutation one-hot (control)</option>
              </select>
            </div>
            <div className="field">
              <label htmlFor="sur">Surrogate model</label>
              <select
                id="sur"
                value={spec.surrogate}
                onChange={(e) => set({ surrogate: e.target.value as 'gp' | 'rf' })}
              >
                <option value="gp">Gaussian process (BoTorch)</option>
                <option value="rf">Random-forest ensemble</option>
              </select>
            </div>
          </div>

          {spec.loop.acquisition === 'ucb' && (
            <div className="field">
              <label htmlFor="beta">UCB exploration weight (β = {spec.loop.ucb_beta})</label>
              <input
                id="beta"
                type="number"
                step={0.5}
                min={0}
                max={20}
                value={spec.loop.ucb_beta}
                onChange={(e) => setLoop({ ucb_beta: Number(e.target.value) })}
              />
              <div className="field-hint">
                Acquisition = posterior mean + β × posterior s.d. Higher explores more.
              </div>
            </div>
          )}

          </div></details>
          <div className="budget-summary">
            <span className="eyebrow">TOTAL MEASUREMENT BUDGET</span><div className="budget-number">{validNumbers ? budget.toLocaleString() : '—'} <span>measurements</span></div>
            {poolSize > 0 && <> out of {poolSize.toLocaleString()} measured variants</>}
            {overBudget && (
              <div style={{ marginTop: 6 }}>
                That budget exceeds the candidate pool — reduce rounds or batch size.
              </div>
            )}
          </div>

          {!validNumbers && <p className="alert err" role="alert">Enter whole numbers within the measurement, round, and seed limits shown in each field.</p>}
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={onStart}
            disabled={starting || overBudget || !validNumbers || loadingDetail || !detail || !!detailError}
          >
            {starting ? <><span className="spinner" /> Preparing…</> : 'Start experiment →'}
          </button>
        </div>
      </div>

      <div className="setup-dataset">
        <AssayCard detail={detail} loading={loadingDetail} />
      </div>
    </div>
  );
}

function AssayCard({ detail, loading }: { detail: AssayDetail | null; loading: boolean }) {
  if (loading && !detail) {
    return (
      <div className="card">
        <div className="card-title">Dataset</div>
        <div className="empty">
          <span className="spinner" /> Loading and validating assay…
        </div>
      </div>
    );
  }
  if (!detail) {
    return (
      <div className="card">
        <div className="card-title">Dataset</div>
        <div className="empty">Select an assay.</div>
      </div>
    );
  }

  const v = detail.validation ?? {};
  const stats = detail.pool_stats;

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="card">
        <div className="card-head">
          <div className="section-heading"><span className="step-label">01</span><div><h2>The protein landscape</h2><p>Your source of experimental truth.</p></div></div>
          <span className="badge ok">Real measurements</span>
        </div>
        <h3 className="protein-title">{detail.molecule_name}</h3>
        <div className="note" style={{ marginBottom: 14 }}>
          {detail.title}
          {detail.first_author && <> — {detail.first_author} et al., {detail.year}</>}
        </div>
        <div className="dataset-metrics"><div><strong>{detail.pool_size?.toLocaleString() ?? '—'}</strong><span>Measured variants</span></div><div><strong>{detail.seq_len}</strong><span>Amino acids</span></div><div><strong>{detail.year}</strong><span>Publication year</span></div></div>
        <dl className="kv">
          <dt>Assay ID</dt>
          <dd>{detail.assay_id}</dd>
          <dt>UniProt</dt>
          <dd>{detail.uniprot_id}</dd>
          <dt>Organism</dt>
          <dd>{detail.source_organism}</dd>
          <dt>Selection</dt>
          <dd>{detail.selection_assay || detail.selection_type}</dd>
          <dt>Sequence length</dt>
          <dd>{detail.seq_len} aa</dd>
          <dt>Candidate pool</dt>
          <dd>{detail.pool_size?.toLocaleString()} measured single variants</dd>
          <dt>Source</dt>
          <dd>{detail.source}</dd>
        </dl>
        <div className="note" style={{ marginTop: 14 }}>
          All fitness values are previously published DMS measurements. Drucion performs a
          retrospective simulation of experimental selection: no new wet-lab experiment is
          run, and measurements stay hidden from the optimizer until it spends budget on
          them.
        </div>
      </div>

      <details className="card disclosure"><summary>Data validation <span>View checks</span></summary>
        <div className="table-wrap">
          <table>
            <tbody>
              {[
                ['Rows in source file', v.rows_in_file],
                ['Dropped: missing values', v.dropped_missing],
                ['Dropped: multi-mutants', v.dropped_multiple_mutants],
                ['Dropped: non-canonical residues', v.dropped_bad_alphabet],
                ['Dropped: length mismatch', v.dropped_length_mismatch],
                ['Dropped: mutation/sequence mismatch', v.dropped_mutation_sequence_inconsistent],
                ['Dropped: duplicate sequences', v.dropped_duplicate_sequences],
                ['Final pool size', v.pool_size],
              ].map(([label, value]) => (
                <tr key={label as string}>
                  <td>{label}</td>
                  <td className="num">{value ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      {stats && (
        <div className="card">
          <div className="card-head">
            <div className="card-title">Fitness range in this assay</div>
            <span className="badge warn">Reference only — hidden from the optimizer</span>
          </div>
          <div className="stats">
            <Stat label="Pool minimum" value={stats.pool_min.toFixed(2)} />
            <Stat label="Pool mean" value={stats.pool_mean.toFixed(2)} />
            <Stat label="90th pct" value={stats.p90.toFixed(2)} />
            <Stat label="99th pct" value={stats.p99.toFixed(2)} />
            <Stat label="Pool maximum" value={stats.pool_max.toFixed(2)} />
          </div>
          <div className="note" style={{ marginTop: 12 }}>
            Shown so you can judge how much headroom exists. These statistics are computed
            after the fact for evaluation; the optimizer never receives them.
          </div>
        </div>
      )}

      <details className="card disclosure"><summary>Wild-type sequence <span>{detail.seq_len} residues</span></summary><div className="seq">{detail.wild_type}</div></details>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}
