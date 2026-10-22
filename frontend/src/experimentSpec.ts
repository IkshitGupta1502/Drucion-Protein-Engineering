export interface ExperimentSpec {
  assay_id: string;
  pool_size: number | null;
  featurizer: 'esm2' | 'esm2_mean' | 'onehot';
  surrogate: 'gp' | 'rf';
  n_components: number;
  loop: {
    n_initial: number;
    batch_size: number;
    n_rounds: number;
    seed: number;
    acquisition: 'random' | 'greedy' | 'ucb' | 'ei' | 'evolutionary';
    ucb_beta: number;
    diversity_quantile: number;
  };
}

export const DEFAULT_SPEC: ExperimentSpec = {
  assay_id: 'A4GRB6_PSEAI_Chen_2020',
  pool_size: null,
  featurizer: 'esm2',
  surrogate: 'gp',
  n_components: 24,
  loop: {
    n_initial: 50,
    batch_size: 10,
    n_rounds: 15,
    seed: 0,
    acquisition: 'ucb',
    ucb_beta: 2.0,
    diversity_quantile: 0.05,
  },
};
