import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import {
  api,
  type ExperimentResults,
  type ExperimentState,
  type RoundRecord,
} from './api';
import { Setup } from './views/Setup';
import { DEFAULT_SPEC, type ExperimentSpec } from './experimentSpec';
const Live = lazy(() => import('./views/Live').then(m => ({ default: m.Live })));
const Benchmark = lazy(() => import('./views/Benchmark').then(m => ({ default: m.Benchmark })));
const Explorer = lazy(() => import('./views/Explorer').then(m => ({ default: m.Explorer })));

type Tab = 'setup' | 'live' | 'benchmark' | 'explorer';

const TABS: { id: Tab; label: string }[] = [
  { id: 'setup', label: 'Experiment setup' },
  { id: 'live', label: 'Live optimization' },
  { id: 'benchmark', label: 'Benchmark' },
  { id: 'explorer', label: 'Protein explorer' },
];

export default function App() {
  const [tab, setTab] = useState<Tab>(() => {
    const hash = window.location.hash.slice(1);
    return TABS.some(t => t.id === hash) ? hash as Tab : 'setup';
  });
  const navigate = (next: Tab) => { window.location.hash = next; setTab(next); };
  useEffect(() => {
    const sync = () => { const hash = window.location.hash.slice(1); setTab(TABS.some(t => t.id === hash) ? hash as Tab : 'setup'); };
    window.addEventListener('hashchange', sync);
    return () => window.removeEventListener('hashchange', sync);
  }, []);
  const [spec, setSpec] = useState<ExperimentSpec>(DEFAULT_SPEC);
  const [experiment, setExperiment] = useState<ExperimentState | null>(null);
  const [rounds, setRounds] = useState<RoundRecord[]>([]);
  const [results, setResults] = useState<ExperimentResults | null>(null);
  const [busy, setBusy] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [device, setDevice] = useState<string>('');
  const [embedProgress, setEmbedProgress] = useState<{ done: number; total: number } | null>(
    null,
  );
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    api
      .health()
      .then((h) => setDevice(h.device))
      .catch(() => undefined);
  }, []);

  const refreshResults = useCallback((id: string) => {
    api
      .results(id)
      .then(setResults)
      .catch(() => undefined);
  }, []);

  /** Subscribe to the server's event stream for a given experiment. */
  const connect = useCallback(
    (id: string) => {
      sourceRef.current?.close();
      const src = new EventSource(api.streamUrl(id));
      sourceRef.current = src;

      src.addEventListener('snapshot', (e) => {
        setExperiment(JSON.parse((e as MessageEvent).data));
      });
      src.addEventListener('ready', (e) => {
        setExperiment(JSON.parse((e as MessageEvent).data));
        setEmbedProgress(null);
      });
      src.addEventListener('embedding_progress', (e) => {
        setEmbedProgress(JSON.parse((e as MessageEvent).data));
      });
      src.addEventListener('round', (e) => {
        const record: RoundRecord = JSON.parse((e as MessageEvent).data);
        setRounds((prev) =>
          prev.some((r) => r.round === record.round) ? prev : [...prev, record],
        );
        api.experiment(id).then(setExperiment).catch(() => undefined);
      });
      src.addEventListener('status', () => {
        api.experiment(id).then(setExperiment).catch(() => undefined);
      });
      src.addEventListener('finished', (e) => {
        setExperiment(JSON.parse((e as MessageEvent).data));
        refreshResults(id);
      });
      src.addEventListener('error', (e) => {
        const data = (e as MessageEvent).data;
        if (data) {
          try {
            setError(JSON.parse(data).error);
          } catch {
            /* transport-level error: the browser retries on its own */
          }
        }
      });
      return src;
    },
    [refreshResults],
  );

  useEffect(() => () => sourceRef.current?.close(), []);

  const start = async () => {
    setStarting(true);
    setError(null);
    setRounds([]);
    setResults(null);
    setEmbedProgress(null);
    try {
      const created = await api.createExperiment(spec);
      setExperiment(created);
      connect(created.id);
      navigate('live');
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  };

  const step = async () => {
    if (!experiment) return;
    setBusy(true);
    setError(null);
    try {
      const record = await api.step(experiment.id);
      setRounds((prev) =>
        prev.some((r) => r.round === record.round) ? prev : [...prev, record],
      );
      setExperiment(await api.experiment(experiment.id));
      refreshResults(experiment.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const run = async () => {
    if (!experiment) return;
    setError(null);
    try {
      setExperiment(await api.run(experiment.id));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const pause = async () => {
    if (!experiment) return;
    await api.pause(experiment.id).catch(() => undefined);
  };

  const reset = async () => {
    if (!experiment) return;
    setRounds([]);
    setResults(null);
    setError(null);
    try {
      const fresh = await api.reset(experiment.id);
      setExperiment(fresh);
      connect(fresh.id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const experimentId = experiment?.id;

  // Keep the explorer in sync once rounds land.
  useEffect(() => {
    if (experimentId && rounds.length) refreshResults(experimentId);
  }, [experimentId, rounds.length, refreshResults]);

  return (
    <div className="app">
      <a className="skip-link" href="#main-content" onClick={(event) => { event.preventDefault(); document.getElementById('main-content')?.focus(); }}>Skip to content</a>
      <aside className="sidebar">
        <a className="brand" href="#setup" aria-label="Drucion home">
          <span className="brand-symbol" aria-hidden="true">d</span>
          <span className="brand-name">Drucion<span className="brand-sub">Protein engineering</span></span>
        </a>
        <div className="nav-heading">WORKSPACE</div>
        <nav className="tabs" aria-label="Workspace navigation">
          {TABS.map((t, i) => (
            <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`}
              aria-current={tab === t.id ? 'page' : undefined} onClick={() => navigate(t.id)}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
                {i === 0 ? <><rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/></> : i === 1 ? <path d="M3 12h4l3-7 4 14 3-7h4"/> : i === 2 ? <><path d="M4 4v16h16M8 15V9m5 6V5m5 10v-4"/></> : <><path d="M7 3c0 8 10 10 10 18M17 3c0 8-10 10-10 18M8 6h8M8 18h8M10 10h4M10 14h4"/></>}
              </svg>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-note"><span className="eyebrow">THE EXPERIMENT LOOP</span><p>Predict. Measure.<br/>Learn. Repeat.</p><span>Find better variants with a focused measurement budget.</span></div>
        <div className="sidebar-footer"><span className={`status-dot ${device ? 'online' : ''}`} />{device ? 'Local engine connected' : 'Connecting to engine…'}<small>ProteinGym · Retrospective experiments</small></div>
      </aside>
      <div className="workspace">
        <header className="topbar"><span>Research workspace <span className="breadcrumb-divider">/</span> <strong>{TABS.find(t => t.id === tab)?.label}</strong></span><span className="environment-label">LOCAL WORKSPACE</span></header>
        <main className="main" id="main-content" tabIndex={-1}>
          <div className="page-heading"><div><div className="eyebrow">{tab === 'setup' ? 'DESIGN YOUR NEXT EXPERIMENT' : 'EXPERIMENT WORKSPACE'}</div><h1>{tab === 'setup' ? 'Design an experiment.' : tab === 'live' ? 'Live optimization.' : tab === 'benchmark' ? 'Compare the evidence.' : 'Explore your variants.'}</h1><p>{tab === 'setup' ? 'Choose a protein, set a budget, and let each measurement inform the next.' : tab === 'live' ? 'Watch predictions become observations, one round at a time.' : tab === 'benchmark' ? 'Compare selection strategies under the same experimental conditions.' : 'Explore sequence changes and the measurements behind each discovery.'}</p></div><span className="page-index">0{TABS.findIndex(t => t.id === tab) + 1} / 04</span></div>
          <Suspense fallback={<div className="empty" role="status"><span className="spinner" /> Loading workspace…</div>}>
        {tab === 'setup' && (
          <Setup
            spec={spec}
            onChange={setSpec}
            onStart={start}
            starting={starting}
            error={error}
          />
        )}
        {tab === 'live' && (
          <>
            {error && <div className="alert err">{error}</div>}
            <Live
              onSetup={() => navigate('setup')}
              experiment={experiment}
              rounds={rounds}
              busy={busy}
              onStep={step}
              onRun={run}
              onPause={pause}
              onReset={reset}
              embedProgress={embedProgress}
            />
          </>
        )}
        {tab === 'benchmark' && <Benchmark />}
        {tab === 'explorer' && <Explorer results={results} onSetup={() => navigate('setup')} />}
          </Suspense>
          <footer className="workspace-footer"><span>Drucion / Research edition</span><span>Published measurements. Transparent predictions.</span></footer>
        </main>
      </div>
    </div>
  );
}
