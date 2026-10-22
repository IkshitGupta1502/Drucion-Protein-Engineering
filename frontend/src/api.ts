/** Typed client for the Drucion API. */

const BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

export interface AssaySummary {
  assay_id: string;
  molecule_name: string;
  source_organism: string;
  selection_type: string;
  selection_assay: string;
  seq_len: number;
  n_mutants: number;
  n_single_mutants: number;
  year: string;
  available: boolean;
}

export interface AssayDetail extends Omit<AssaySummary, 'available'> {
  uniprot_id: string;
  first_author: string;
  title: string;
  n_multiple_mutants: number;
  wild_type: string;
  region_mutated: string;
  source: string;
  license: string;
  validation?: Record<string, number>;
  pool_size?: number;
  pool_stats?: Record<string, number>;
}

export interface Variant {
  index: number;
  mutant: string;
  positions: number[];
  predicted_fitness: number | null;
  predicted_uncertainty: number | null;
  acquisition_score: number;
  measured_fitness: number;
}

export interface RoundRecord {
  round: number;
  n_measured: number;
  best_measured: number;
  mean_measured: number;
  selected: Variant[];
  surrogate_kind: string;
  surrogate_info: Record<string, unknown>;
  surrogate_spearman: number | null;
  surrogate_rmse: number | null;
  holdout_spearman: number | null;
  holdout_rmse: number | null;
  proposal_report: Record<string, unknown> | null;
  wall_time_s: number;
}

export interface ExperimentState {
  id: string;
  status: string;
  assay_id: string;
  acquisition: string;
  round: number;
  n_rounds: number;
  n_measured: number;
  budget: number;
  best_measured?: number | null;
  best_mutant?: string | null;
  pool_size?: number | null;
  surrogate_kind?: string | null;
  features?: Record<string, unknown> | null;
  learning_curve?: { n_measured: number[]; best_measured: number[] } | null;
  error?: string | null;
  config?: Record<string, unknown> | null;
}

export interface ExperimentResults {
  id: string;
  status: string;
  assay_id: string;
  acquisition: string;
  n_measured: number;
  budget: number;
  best_measured: number | null;
  best_mutant: string | null;
  wild_type?: string;
  best_sequence?: string;
  differences: { position: number; wild_type_aa: string; variant_aa: string }[];
  learning_curve?: { n_measured: number[]; best_measured: number[] } | null;
  rounds: RoundRecord[];
  pool_stats?: Record<string, number> | null;
}

export interface BenchmarkCurve {
  n_measured: number[];
  mean: number[];
  std: number[];
}

export interface PairedMetric {
  n: number;
  wins: number;
  ties: number;
  mean_difference: number;
  wilcoxon_p: number | null;
  lower_is_better: boolean;
}

export interface BenchmarkPayload {
  assay_id: string;
  methods: string[];
  seeds: number[];
  pool_stats: Record<string, number>;
  feature_info: Record<string, unknown>;
  config: Record<string, unknown>;
  runs: Record<string, unknown>[];
  curves: Record<string, BenchmarkCurve>;
  surrogate_curves?: Record<string, { n_measured: number[]; mean: number[]; std: number[] }>;
  summary: Record<string, Record<string, number | null>>;
  paired?: {
    baseline: string;
    methods: Record<string, { n_paired_seeds: number; metrics: Record<string, PairedMetric> }>;
  };
  efficiency?: {
    baseline: string;
    target: number;
    methods: Record<
      string,
      { budget_to_match: number | null; savings_fraction: number | null; final_value: number }
    >;
  };
  notes: string[];
  wall_time_s: number;
}

export interface BenchmarkJob {
  id: string;
  status: string;
  assay_id: string;
  progress: number;
  result: BenchmarkPayload | null;
  error: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; device: string; torch: string }>('/api/health'),
  assays: () => request<AssaySummary[]>('/api/assays'),
  assay: (id: string, full = false) =>
    request<AssayDetail>(`/api/assays/${id}${full ? '?full=true' : ''}`),

  createExperiment: (body: unknown) =>
    request<ExperimentState>('/api/experiments', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  experiment: (id: string) => request<ExperimentState>(`/api/experiments/${id}`),
  step: (id: string) => request<RoundRecord>(`/api/experiments/${id}/step`, { method: 'POST' }),
  run: (id: string) => request<ExperimentState>(`/api/experiments/${id}/run`, { method: 'POST' }),
  pause: (id: string) => request<unknown>(`/api/experiments/${id}/pause`, { method: 'POST' }),
  reset: (id: string) => request<ExperimentState>(`/api/experiments/${id}/reset`, { method: 'POST' }),
  results: (id: string) => request<ExperimentResults>(`/api/experiments/${id}/results`),

  createBenchmark: (body: unknown) =>
    request<BenchmarkJob>('/api/benchmarks', { method: 'POST', body: JSON.stringify(body) }),
  benchmark: (id: string) => request<BenchmarkJob>(`/api/benchmarks/${id}`),
  precomputed: () =>
    request<{ benchmarks: Record<string, BenchmarkPayload> }>('/api/results/precomputed'),

  streamUrl: (id: string) => `${BASE}/api/experiments/${id}/stream`,
};

export const METHOD_LABELS: Record<string, string> = {
  random: 'Random selection',
  evolutionary: 'ESM-guided evolutionary',
  greedy: 'Greedy surrogate',
  ei: 'Bayesian AL (EI)',
  ucb: 'Bayesian AL (UCB)',
};

/** Okabe-Ito colour-blind-safe palette, shared with the Python plots. */
export const METHOD_COLORS: Record<string, string> = {
  random: '#657269',
  evolutionary: '#986414',
  greedy: '#28718c',
  ei: '#167154',
  ucb: '#b44d29',
};
