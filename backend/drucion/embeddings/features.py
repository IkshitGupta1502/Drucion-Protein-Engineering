"""Featurizers that turn a label-free candidate pool into a surrogate input matrix.

Two representations are provided so the benchmark can separate *where the gain
comes from*:

``esm2``
    Mean-pooled ESM-2 embeddings, PCA-reduced. Carries pretrained
    evolutionary knowledge.
``onehot``
    One-hot encoding of the mutation itself (position x substituted residue),
    PCA-reduced. Carries no pretrained knowledge, only the identity of the
    change.

Running the same active-learning loop on both tells us how much of the
improvement is active learning versus the pretrained protein language model.

PCA is fit on the **features of the whole candidate pool**, which is
label-free and therefore does not leak the oracle's held-out fitness values.
This is standard transductive unsupervised preprocessing; it is applied
identically to every method so the comparison stays fair.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from ..config import EmbeddingConfig
from ..data.proteingym import AMINO_ACIDS, CandidatePool, parse_mutant

logger = logging.getLogger(__name__)

_AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def onehot_mutation_features(pool: CandidatePool) -> np.ndarray:
    """Sparse-ish one-hot encoding of each variant's substitutions.

    Layout: for every mutated position observed anywhere in the pool, 20
    columns (one per substituted residue). Columns for positions the pool never
    mutates are omitted, which keeps the matrix compact without losing
    information.
    """
    all_positions = sorted(
        {s.position for m in pool.mutants for s in parse_mutant(m)}
    )
    pos_index = {p: i for i, p in enumerate(all_positions)}
    X = np.zeros((len(pool), len(all_positions) * len(AMINO_ACIDS)), dtype=np.float32)
    for row, mutant in enumerate(pool.mutants):
        for sub in parse_mutant(mutant):
            col = pos_index[sub.position] * len(AMINO_ACIDS) + _AA_INDEX[sub.mut_aa]
            X[row, col] = 1.0
    return X


@dataclass
class FeatureSet:
    """Reduced features for a candidate pool, plus provenance for the UI."""

    X: np.ndarray  # (pool_size, n_features) float64
    featurizer: str
    raw_dim: int
    n_components: int
    explained_variance_ratio: float
    includes_zero_shot: bool = False
    #: ESM-2 masked-marginal zero-shot scores, kept for the evolutionary
    #: baseline and for the UI. ``None`` when not computed.
    zero_shot: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "featurizer": self.featurizer,
            "raw_dim": self.raw_dim,
            "n_components": self.n_components,
            "n_features": int(self.X.shape[1]),
            "explained_variance_ratio": round(self.explained_variance_ratio, 4),
            "includes_zero_shot": self.includes_zero_shot,
        }


def build_features(
    pool: CandidatePool,
    config: EmbeddingConfig | None = None,
    progress=None,
) -> FeatureSet:
    """Compute and PCA-reduce features for every variant in ``pool``."""
    config = config or EmbeddingConfig()

    if config.featurizer in ("esm2", "esm2_mean"):
        from .esm import ESM2Embedder

        embedder = ESM2Embedder(
            model_name=config.model_name,
            device=config.device,
            batch_size=config.batch_size,
        )
        if config.featurizer == "esm2":
            raw = embedder.embed_sites(
                pool.sequences,
                [pool.mutated_positions(i) for i in range(len(pool))],
                pool.wild_type,
                progress=progress,
            )
        else:
            raw = embedder.embed_sequences(pool.sequences, progress=progress)
    elif config.featurizer == "onehot":
        raw = onehot_mutation_features(pool)
        if progress:
            progress(len(pool), len(pool))
    else:  # pragma: no cover - guarded by config typing
        raise ValueError(f"Unknown featurizer: {config.featurizer!r}")

    raw = np.asarray(raw, dtype=np.float64)
    n_components = int(min(config.n_components, raw.shape[0], raw.shape[1]))
    pca = PCA(n_components=n_components, random_state=0)
    X = pca.fit_transform(raw)

    # Centre, then divide by a single global scale. Rescaling each component
    # to unit variance was measurably worse: it inflates low-variance noise
    # directions until they compete with the informative leading PCs.
    X = X - X.mean(axis=0)
    global_scale = float(X.std())
    if global_scale > 1e-12:
        X = X / global_scale

    zero_shot = None
    if config.include_zero_shot:
        from .esm import ESM2Embedder

        embedder = ESM2Embedder(
            model_name=config.model_name,
            device=config.device,
            batch_size=config.batch_size,
        )
        zero_shot = embedder.masked_marginal_scores(pool.wild_type, pool.mutants)
        zs = zero_shot - zero_shot.mean()
        zs_std = float(zs.std())
        if zs_std > 1e-12:
            zs = zs / zs_std
        # Put the zero-shot column on the same scale as the PCA block so the
        # GP's ARD kernel starts out treating it comparably.
        X = np.hstack([X, (zs * X.std()).reshape(-1, 1)])

    return FeatureSet(
        X=np.ascontiguousarray(X, dtype=np.float64),
        featurizer=config.featurizer,
        raw_dim=int(raw.shape[1]),
        n_components=n_components,
        explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
        includes_zero_shot=config.include_zero_shot,
        zero_shot=zero_shot,
    )
