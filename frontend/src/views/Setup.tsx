import { useEffect, useState } from 'react';
import { api, type AssayDetail, type AssaySummary } from '../api';
import { type ExperimentSpec } from '../experimentSpec';

const FEATURED = ['A4GRB6_PSEAI_Chen_2020', 'TPMT_HUMAN_Matreyek_2018', 'BLAT_ECOLX_Firnberg_2014'];
const SAMPLE_NAMES: Record<string, string> = {
  A4GRB6_PSEAI_Chen_2020: 'VIM-2 enzyme · Recommended first run',
  TPMT_HUMAN_Matreyek_2018: 'TPMT protein · Try a different challenge',
  BLAT_ECOLX_Firnberg_2014: 'TEM-1 enzyme · Explore another dataset',
};
interface Props {
  spec: ExperimentSpec;
  onChange: (spec: ExperimentSpec) => void;
  onStart: (automatic: boolean) => void;
  onPreview: () => void;
  starting: boolean;
  active: boolean;
  onResume: () => void;
  error: string | null;
}

export function Setup({ spec, onChange, onStart, onPreview, starting, active, onResume, error }: Props) {
  const [assays, setAssays] = useState<AssaySummary[]>([]);
  const [detail, setDetail] = useState<AssayDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [automatic, setAutomatic] = useState(true);
  useEffect(() => {
    let cancelled = false;
    setLoading(true); setLoadError(null); setDetail(null);
    Promise.all([api.assays(), api.assay(spec.assay_id, true)])
      .then(([list, selected]) => { if (!cancelled) { setAssays(list); setDetail(selected); } })
      .catch((e: Error) => { if (!cancelled) setLoadError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [spec.assay_id, retry]);
  const setLoop = (patch: Partial<ExperimentSpec['loop']>) => onChange({ ...spec, loop: { ...spec.loop, ...patch } });
  const budget = spec.loop.n_initial + spec.loop.batch_size * spec.loop.n_rounds;
  const starter = spec.loop.n_initial === 50 && spec.loop.batch_size === 10 && spec.loop.n_rounds === 3;
  const extended = spec.loop.n_initial === 50 && spec.loop.batch_size === 10 && spec.loop.n_rounds === 15;
  const valid = Number.isInteger(spec.loop.n_initial) && spec.loop.n_initial >= 2 && spec.loop.n_initial <= 2000
    && Number.isInteger(spec.loop.batch_size) && spec.loop.batch_size >= 1 && spec.loop.batch_size <= 500
    && Number.isInteger(spec.loop.n_rounds) && spec.loop.n_rounds >= 1 && spec.loop.n_rounds <= 200
    && Number.isInteger(spec.loop.seed) && spec.loop.seed >= 0 && spec.loop.seed <= 10000
    && Number.isFinite(spec.loop.ucb_beta) && spec.loop.ucb_beta >= 0 && spec.loop.ucb_beta <= 20;
  const tooLarge = !!detail?.pool_size && budget >= detail.pool_size;
  const featured = assays.filter(a => FEATURED.includes(a.assay_id));
  const rest = assays.filter(a => !FEATURED.includes(a.assay_id));
  return (
    <>
      {active && <div className="resume-banner"><div><strong>You have a run in progress.</strong><p>Pick up where you left off before starting another.</p></div><button className="btn" onClick={onResume}>Return to my run →</button></div>}
      <div className="onboarding-grid">
        <section className="welcome-panel" aria-labelledby="welcome-title">
          <span className="badge ok">Ready-to-run protein datasets</span>
          <h2 id="welcome-title">Spend your next experiment <br/>on a better candidate.</h2>
          <p className="welcome-copy">Drucion helps you choose which protein variants to test next. It learns from each result and uses that feedback to guide the next batch.</p>
          <button className="btn btn-primary mobile-start-link" onClick={() => { const target = document.getElementById('start-title'); target?.scrollIntoView({ block: 'start' }); target?.focus({ preventScroll: true }); }}>Set up my first search ↓</button>
          <ol className="journey-list">
            <li><span>1</span><div><h3>Choose your starting point</h3><p>Use a ready-made dataset. No files to upload or model to train yourself.</p></div></li>
            <li><span>2</span><div><h3>Watch the search learn</h3><p>Drucion starts with a small sample, picks promising variants, and checks their results.</p></div></li>
            <li><span>3</span><div><h3>See what you found</h3><p>Get the best variant found, its measured score, and a clear view of your progress.</p></div></li>
          </ol>
          <div className="demo-explainer"><strong>Try it with real results without a lab</strong><p>This demo reveals published protein measurements as it runs. It uses no lab equipment and does not spend money or perform new experiments.</p></div>
          <button className="text-button" onClick={onPreview}>Just looking? Explore example results →</button>
        </section>
        <section className="card start-card" aria-labelledby="start-title">
          <div className="card-head"><h2 id="start-title" tabIndex={-1}>Start your first search</h2><span className="badge">No setup required</span></div>
          <p className="start-intro">We’ve picked a starting dataset and balanced search settings. You can change them anytime before you start.</p>
          {loadError && <div className="alert err" role="alert"><strong>We couldn’t load the datasets.</strong><p>Check that the Drucion server is running, then try again.</p><details><summary>Technical details</summary>{loadError}</details><button className="btn" onClick={() => setRetry(v => v + 1)}>Try again</button></div>}
          {error && <div className="alert err" role="alert">{error}</div>}
          <div className="field"><label htmlFor="assay">1. Choose a protein dataset</label><select id="assay" value={spec.assay_id} disabled={loading} onChange={e => onChange({ ...spec, assay_id: e.target.value })}>
            {assays.length === 0 && <option value={spec.assay_id}>Loading datasets…</option>}
            <optgroup label="Ready-to-run examples">{featured.map(a => <option key={a.assay_id} value={a.assay_id}>{SAMPLE_NAMES[a.assay_id]}</option>)}</optgroup>
            <optgroup label="More datasets">{rest.map(a => <option key={a.assay_id} value={a.assay_id}>{a.molecule_name} · {a.assay_id}</option>)}</optgroup>
          </select><div className="field-hint">{loading ? 'Checking the available measurements…' : detail ? `${detail.pool_size?.toLocaleString()} protein variants available to explore. A variant is a version of a protein with a small sequence change.` : 'Choose a dataset to continue.'}</div></div>
          <fieldset className="plan-picker"><legend>2. Choose how far to explore</legend><div className="plan-options">
            <button type="button" className={`plan-option ${starter ? 'selected' : ''}`} aria-pressed={starter} onClick={() => setLoop({ n_initial: 50, batch_size: 10, n_rounds: 3 })}><span>Quick start</span><strong>80 <small>measurements</small></strong><p>3 learning rounds · Try the workflow</p></button>
            <button type="button" className={`plan-option ${extended ? 'selected' : ''}`} aria-pressed={extended} onClick={() => setLoop({ n_initial: 50, batch_size: 10, n_rounds: 15 })}><span>Explore further</span><strong>200 <small>measurements</small></strong><p>15 learning rounds · A deeper search</p></button>
          </div>{!starter && !extended && <p className="field-hint">Custom plan selected below.</p>}</fieldset>
          <details className="advanced-settings"><summary>Customize search settings<span>Optional</span></summary><div className="advanced-content">
            <p className="note">The defaults are ready to use. These controls let you change how candidates are chosen and how much data the search uses.</p>
            <div className="field"><label htmlFor="acq">Search approach</label><select id="acq" value={spec.loop.acquisition} onChange={e => setLoop({ acquisition: e.target.value as ExperimentSpec['loop']['acquisition'] })}><option value="ucb">Balanced search (UCB) — recommended</option><option value="ei">Expected improvement (EI)</option><option value="greedy">Highest predicted score first</option><option value="evolutionary">Sequence-guided search</option><option value="random">Random sampling — comparison baseline</option></select><p className="field-hint">Balanced search weighs promising scores against opportunities the model is less certain about.</p></div>
            <div className="row"><NumberField id="init" label="Starting sample" value={spec.loop.n_initial} min={2} max={2000} onChange={v => setLoop({ n_initial: v })}/><NumberField id="batch" label="Variants per round" value={spec.loop.batch_size} min={1} max={500} onChange={v => setLoop({ batch_size: v })}/></div>
            <div className="row"><NumberField id="rounds" label="Learning rounds" value={spec.loop.n_rounds} min={1} max={200} onChange={v => setLoop({ n_rounds: v })}/><NumberField id="seed" label="Random seed" value={spec.loop.seed} min={0} max={10000} onChange={v => setLoop({ seed: v })}/></div>
            <div className="field"><label htmlFor="feat">Protein representation</label><select id="feat" value={spec.featurizer} onChange={e => onChange({ ...spec, featurizer: e.target.value as ExperimentSpec['featurizer'] })}><option value="esm2">ESM-2 site-resolved (recommended)</option><option value="esm2_mean">ESM-2 mean-pooled</option><option value="onehot">Mutation one-hot</option></select></div>
            <div className="field"><label htmlFor="sur">Prediction model</label><select id="sur" value={spec.surrogate} onChange={e => onChange({ ...spec, surrogate: e.target.value as 'gp' | 'rf' })}><option value="gp">Gaussian process (recommended)</option><option value="rf">Random forest</option></select></div>
            {spec.loop.acquisition === 'ucb' && <NumberField id="beta" label="Exploration weight" value={spec.loop.ucb_beta} min={0} max={20} step={0.5} onChange={v => setLoop({ ucb_beta: v })}/>}
          </div></details>
          <label className="check-field"><input type="checkbox" checked={automatic} onChange={e => setAutomatic(e.target.checked)}/><span>Run all rounds automatically<small>Turn this off to review each batch before continuing.</small></span></label>
          <div className="run-receipt"><strong>Here’s what will happen</strong><p>{valid ? <>Start with {spec.loop.n_initial} known results, then check {spec.loop.batch_size} more variants in each of {spec.loop.n_rounds} rounds. That’s <b>{budget} measurements total</b>.</> : 'Set valid search limits to continue.'}</p></div>
          {(!valid || tooLarge) && <p className="alert err" role="alert">{tooLarge ? 'This dataset is too small for the selected plan. Reduce the starting sample or number of rounds.' : 'Check the number limits in Customize search settings.'}</p>}
          <button className="btn btn-primary start-search" onClick={() => onStart(automatic)} disabled={starting || loading || !detail || !!loadError || !valid || tooLarge || active}>{starting ? 'Starting your search…' : loading ? 'Loading your dataset…' : active ? 'A search is already in progress' : 'Start my search →'}</button>
          <p className="cta-note">{automatic ? 'We’ll handle the next steps. You can pause between rounds.' : 'You’ll choose when to run each round.'} Preparation can take longer the first time a dataset is used.</p>
        </section>
      </div>
      <details className="dataset-details"><summary>About the data & how results are measured<span>For a closer look</span></summary><AssayCard detail={detail} loading={loading}/></details>
    </>
  );
}
function NumberField({ id, label, value, min, max, step = 1, onChange }: { id: string; label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void }) {
  return <div className="field"><label htmlFor={id}>{label}</label><input id={id} type="number" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))}/><span className="field-hint">{min}–{max}</span></div>;
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
