"""Shared fixtures.

Unit tests run against small **synthetic** assay fixtures so they are fast and
hermetic. Tests that need the real ProteinGym data are marked ``real_data`` and
skip automatically when the download is absent — the performance claims in the
README come from those, not from the synthetic fixtures.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from drucion.config import PROTEINGYM_DIR, REFERENCE_CSV

AA = "ACDEFGHIKLMNPQRSTVWY"

#: A short, well-formed wild-type sequence for synthetic fixtures.
SYNTHETIC_WT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_data: requires the downloaded ProteinGym assays"
    )


def _synthetic_rows(wild_type: str, n: int, seed: int = 0):
    """Deterministic single substitutions with a smooth synthetic fitness.

    Fitness is a simple function of position and substituted residue, so a
    surrogate can genuinely learn it — which is what lets the loop tests assert
    meaningful behaviour rather than just shapes.
    """
    rng = np.random.default_rng(seed)
    rows = []
    seen = set()
    L = len(wild_type)
    while len(rows) < n:
        pos = int(rng.integers(1, L + 1))
        wt_aa = wild_type[pos - 1]
        mut_aa = AA[int(rng.integers(0, len(AA)))]
        if mut_aa == wt_aa:
            continue
        spec = f"{wt_aa}{pos}{mut_aa}"
        if spec in seen:
            continue
        seen.add(spec)
        seq = wild_type[: pos - 1] + mut_aa + wild_type[pos:]
        score = (
            np.sin(pos / 3.0)
            + 0.5 * (AA.index(mut_aa) / len(AA))
            + 0.05 * rng.standard_normal()
        )
        rows.append({"mutant": spec, "mutated_sequence": seq, "DMS_score": score})
    return rows


@pytest.fixture
def synthetic_assay(tmp_path: Path):
    """Write a synthetic assay CSV + reference row; return loader kwargs."""
    assay_id = "SYNTH_TEST_Fixture_2026"
    wild_type = SYNTHETIC_WT
    rows = _synthetic_rows(wild_type, n=500, seed=1)

    assay_dir = tmp_path / "assays"
    assay_dir.mkdir()
    csv_path = assay_dir / f"{assay_id}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mutant", "mutated_sequence", "DMS_score"])
        w.writeheader()
        w.writerows(rows)

    ref_path = tmp_path / "DMS_substitutions.csv"
    with ref_path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "DMS_id",
                "UniProt_ID",
                "molecule_name",
                "source_organism",
                "selection_assay",
                "selection_type",
                "first_author",
                "title",
                "year",
                "seq_len",
                "DMS_total_number_mutants",
                "DMS_number_single_mutants",
                "DMS_number_multiple_mutants",
                "target_seq",
                "region_mutated",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "DMS_id": assay_id,
                "UniProt_ID": "SYNTH1",
                "molecule_name": "Synthetic test protein",
                "source_organism": "in silico",
                "selection_assay": "synthetic",
                "selection_type": "Synthetic",
                "first_author": "Fixture",
                "title": "Synthetic fixture for unit tests",
                "year": "2026",
                "seq_len": len(wild_type),
                "DMS_total_number_mutants": len(rows),
                "DMS_number_single_mutants": len(rows),
                "DMS_number_multiple_mutants": 0,
                "target_seq": wild_type,
                "region_mutated": f"1-{len(wild_type)}",
            }
        )

    # The metadata reader memoises the reference table; clear it per test.
    from drucion.data import proteingym

    proteingym._reference_table.cache_clear()

    return {
        "assay_id": assay_id,
        "assay_dir": assay_dir,
        "reference_csv": str(ref_path),
        "wild_type": wild_type,
        "n_rows": len(rows),
    }


@pytest.fixture
def synthetic_loaded(synthetic_assay):
    from drucion.data.proteingym import load_assay

    return load_assay(
        synthetic_assay["assay_id"],
        assay_dir=synthetic_assay["assay_dir"],
        reference_csv=synthetic_assay["reference_csv"],
    )


@pytest.fixture
def synthetic_features(synthetic_loaded):
    """Cheap deterministic features: one-hot mutation encoding, PCA-reduced.

    Avoids loading ESM-2 in unit tests while exercising the same code path.
    """
    from drucion.config import EmbeddingConfig
    from drucion.embeddings.features import build_features

    cfg = EmbeddingConfig(
        featurizer="onehot", n_components=12, include_zero_shot=False
    )
    return build_features(synthetic_loaded.pool, cfg)


@pytest.fixture
def synthetic_learnable_features(synthetic_loaded):
    """Features that actually encode the synthetic fitness function.

    ``synthetic_features`` (PCA'd one-hot) deliberately carries almost no
    signal, which is right for shape and plumbing tests but makes it
    impossible to test whether the loop can *exploit* a learnable landscape.
    This fixture hands the surrogate the two quantities the synthetic score is
    built from — mutated position and substituted-residue index — so a
    successful run demonstrates the machinery works end to end.

    Real runs use ESM-2 features; see tests marked ``real_data`` and
    ``scripts/run_benchmark.py`` for anything resembling a performance claim.
    """
    from drucion.data.proteingym import parse_mutant
    from drucion.embeddings.features import FeatureSet

    pool = synthetic_loaded.pool
    rows = []
    for mutant in pool.mutants:
        (sub,) = parse_mutant(mutant)
        rows.append([sub.position, AA.index(sub.mut_aa)])
    X = np.asarray(rows, dtype=np.float64)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return FeatureSet(
        X=X,
        featurizer="synthetic-oracle-features",
        raw_dim=2,
        n_components=2,
        explained_variance_ratio=1.0,
    )


@pytest.fixture(scope="session")
def real_data_available() -> bool:
    return REFERENCE_CSV.exists() and PROTEINGYM_DIR.exists()


@pytest.fixture
def require_real_data(real_data_available):
    if not real_data_available:
        pytest.skip("ProteinGym data not downloaded (run scripts/download_data.py)")
