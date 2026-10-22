"""Featurizer shapes, determinism and ESM-2 pooling correctness.

ESM-2 tests are marked ``real_data`` only in the sense that they download the
8M-parameter checkpoint on first run; they are otherwise hermetic. They are
kept small (a handful of short sequences) so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from drucion.config import EmbeddingConfig
from drucion.data.proteingym import AMINO_ACIDS, CandidatePool
from drucion.embeddings.features import build_features, onehot_mutation_features


@pytest.fixture
def tiny_pool():
    wt = "MKTAYIAKQRQ"
    mutants, seqs = [], []
    for pos, mut in [(2, "R"), (4, "G"), (6, "V"), (2, "E")]:
        mutants.append(f"{wt[pos - 1]}{pos}{mut}")
        seqs.append(wt[: pos - 1] + mut + wt[pos:])
    return CandidatePool(
        assay_id="TINY", wild_type=wt, mutants=tuple(mutants), sequences=tuple(seqs)
    )


class TestOneHotFeatures:
    def test_shape_covers_only_mutated_positions(self, tiny_pool):
        X = onehot_mutation_features(tiny_pool)
        # 3 distinct positions mutated (2, 4, 6) x 20 residues
        assert X.shape == (4, 3 * len(AMINO_ACIDS))

    def test_one_hot_per_single_mutant(self, tiny_pool):
        X = onehot_mutation_features(tiny_pool)
        assert np.all(X.sum(axis=1) == 1.0)

    def test_distinct_mutations_differ(self, tiny_pool):
        X = onehot_mutation_features(tiny_pool)
        # K2R and K2E are at the same position but different residues
        assert not np.array_equal(X[0], X[3])

    def test_is_deterministic(self, tiny_pool):
        assert np.array_equal(
            onehot_mutation_features(tiny_pool), onehot_mutation_features(tiny_pool)
        )


class TestBuildFeatures:
    def test_onehot_featureset_shape_and_metadata(self, tiny_pool):
        cfg = EmbeddingConfig(
            featurizer="onehot", n_components=3, include_zero_shot=False
        )
        fs = build_features(tiny_pool, cfg)
        assert fs.X.shape == (4, 3)
        assert fs.X.dtype == np.float64
        assert fs.featurizer == "onehot"
        assert fs.n_components == 3
        assert fs.includes_zero_shot is False
        assert fs.zero_shot is None
        assert 0.0 <= fs.explained_variance_ratio <= 1.0001

    def test_components_capped_by_data_size(self, tiny_pool):
        cfg = EmbeddingConfig(
            featurizer="onehot", n_components=999, include_zero_shot=False
        )
        fs = build_features(tiny_pool, cfg)
        assert fs.n_components <= len(tiny_pool)

    def test_features_are_centred(self, tiny_pool):
        cfg = EmbeddingConfig(
            featurizer="onehot", n_components=3, include_zero_shot=False
        )
        fs = build_features(tiny_pool, cfg)
        assert np.allclose(fs.X.mean(axis=0), 0.0, atol=1e-9)

    def test_all_values_finite(self, tiny_pool):
        cfg = EmbeddingConfig(
            featurizer="onehot", n_components=3, include_zero_shot=False
        )
        assert np.all(np.isfinite(build_features(tiny_pool, cfg).X))

    def test_unknown_featurizer_raises(self, tiny_pool):
        cfg = EmbeddingConfig(featurizer="nonsense", include_zero_shot=False)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown featurizer"):
            build_features(tiny_pool, cfg)

    def test_progress_callback_reports_completion(self, tiny_pool):
        seen = []
        cfg = EmbeddingConfig(
            featurizer="onehot", n_components=2, include_zero_shot=False
        )
        build_features(tiny_pool, cfg, progress=lambda d, t: seen.append((d, t)))
        assert seen[-1] == (len(tiny_pool), len(tiny_pool))

    def test_to_dict_is_serialisable(self, tiny_pool):
        import json

        cfg = EmbeddingConfig(
            featurizer="onehot", n_components=2, include_zero_shot=False
        )
        json.dumps(build_features(tiny_pool, cfg).to_dict())


@pytest.mark.real_data
class TestESM2:
    """Exercises the actual ESM-2 checkpoint on a few short sequences."""

    @pytest.fixture(scope="class")
    def embedder(self):
        from drucion.embeddings.esm import ESM2Embedder

        return ESM2Embedder(batch_size=4)

    def test_embedding_dimension_matches_checkpoint(self, embedder):
        assert embedder.embedding_dim == 320  # esm2_t6_8M_UR50D hidden size

    def test_mean_pooled_shape(self, embedder, tiny_pool):
        X = embedder.embed_sequences(tiny_pool.sequences, use_cache=False)
        assert X.shape == (len(tiny_pool), embedder.embedding_dim)
        assert np.all(np.isfinite(X))

    def test_identical_sequences_give_identical_embeddings(self, embedder):
        seq = "MKTAYIAKQRQ"
        X = embedder.embed_sequences([seq, seq], use_cache=False)
        assert np.allclose(X[0], X[1])

    def test_different_sequences_give_different_embeddings(self, embedder):
        X = embedder.embed_sequences(
            ["MKTAYIAKQRQ", "MKTAYIAKQRW"], use_cache=False
        )
        assert not np.allclose(X[0], X[1])

    def test_padding_does_not_change_pooled_embedding(self, embedder):
        """The pooling mask must exclude padding, or batching changes results."""
        short = "MKTAYIAK"
        alone = embedder.embed_sequences([short], use_cache=False)[0]
        # Batched with a much longer sequence, which forces padding.
        batched = embedder.embed_sequences(
            [short, "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"], use_cache=False
        )[0]
        assert np.allclose(alone, batched, atol=1e-4)

    def test_site_embedding_shape_is_double_width(self, embedder, tiny_pool):
        positions = [tiny_pool.mutated_positions(i) for i in range(len(tiny_pool))]
        X = embedder.embed_sites(
            tiny_pool.sequences, positions, tiny_pool.wild_type, use_cache=False
        )
        assert X.shape == (len(tiny_pool), 2 * embedder.embedding_dim)
        assert np.all(np.isfinite(X))

    def test_site_embedding_difference_block_is_nonzero(self, embedder, tiny_pool):
        """The second half is h_mut - h_wt; a real substitution must move it."""
        positions = [tiny_pool.mutated_positions(i) for i in range(len(tiny_pool))]
        X = embedder.embed_sites(
            tiny_pool.sequences, positions, tiny_pool.wild_type, use_cache=False
        )
        diff_block = X[:, embedder.embedding_dim :]
        assert np.all(np.linalg.norm(diff_block, axis=1) > 1e-3)

    def test_site_embedding_rejects_mismatched_lengths(self, embedder, tiny_pool):
        with pytest.raises(ValueError, match="same length"):
            embedder.embed_sites(
                tiny_pool.sequences, [[2]], tiny_pool.wild_type, use_cache=False
            )

    def test_zero_shot_scores_shape_and_sign(self, embedder, tiny_pool):
        scores = embedder.masked_marginal_scores(
            tiny_pool.wild_type, tiny_pool.mutants
        )
        assert scores.shape == (len(tiny_pool),)
        assert np.all(np.isfinite(scores))
        # A substitution away from the wild type is usually disfavoured, so
        # log p(mut) - log p(wt) should be mostly negative.
        assert float(np.mean(scores)) < 0.0

    def test_caching_returns_identical_values(self, embedder, tiny_pool):
        a = embedder.embed_sequences(tiny_pool.sequences, use_cache=True)
        b = embedder.embed_sequences(tiny_pool.sequences, use_cache=True)
        assert np.array_equal(a, b)
