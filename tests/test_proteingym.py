"""ProteinGym parsing, validation and mutation/sequence consistency."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from drucion.data.proteingym import (
    AssayValidationError,
    ExperimentalOracle,
    apply_mutations,
    load_assay,
    parse_mutant,
)


class TestParseMutant:
    def test_single_substitution(self):
        (sub,) = parse_mutant("H24C")
        assert (sub.wt_aa, sub.position, sub.mut_aa) == ("H", 24, "C")

    def test_multiple_substitutions(self):
        subs = parse_mutant("H24C:A42G")
        assert [s.position for s in subs] == [24, 42]

    @pytest.mark.parametrize(
        "bad", ["", "24C", "HC", "H0C", "H24", "h24c", "HX24C", "H24B", "H-4C"]
    )
    def test_rejects_malformed(self, bad):
        with pytest.raises(AssayValidationError):
            parse_mutant(bad)


class TestApplyMutations:
    def test_applies_substitution(self):
        wt = "MKTAY"
        assert apply_mutations(wt, "K2R") == "MRTAY"

    def test_preserves_length(self):
        wt = "MKTAYIAK"
        assert len(apply_mutations(wt, "A4G")) == len(wt)

    def test_rejects_wrong_wild_type_residue(self):
        with pytest.raises(AssayValidationError, match="expects"):
            apply_mutations("MKTAY", "A2R")

    def test_rejects_out_of_range_position(self):
        with pytest.raises(AssayValidationError, match="outside"):
            apply_mutations("MKTAY", "Y99A")

    def test_multiple_substitutions_commute(self):
        wt = "MKTAYIAK"
        assert apply_mutations(wt, "K2R:A4G") == apply_mutations(wt, "A4G:K2R")


class TestLoadAssay:
    def test_loads_synthetic_assay(self, synthetic_loaded, synthetic_assay):
        assert len(synthetic_loaded.pool) == synthetic_assay["n_rows"]
        assert synthetic_loaded.pool.wild_type == synthetic_assay["wild_type"]
        assert len(synthetic_loaded.oracle) == len(synthetic_loaded.pool)

    def test_every_sequence_matches_its_mutation_spec(self, synthetic_loaded):
        """The core data-integrity invariant of the whole project."""
        pool = synthetic_loaded.pool
        for mutant, seq in zip(pool.mutants, pool.sequences):
            assert apply_mutations(pool.wild_type, mutant) == seq

    def test_all_sequences_have_wild_type_length(self, synthetic_loaded):
        wt_len = len(synthetic_loaded.pool.wild_type)
        assert all(len(s) == wt_len for s in synthetic_loaded.pool.sequences)

    def test_no_duplicate_sequences(self, synthetic_loaded):
        seqs = synthetic_loaded.pool.sequences
        assert len(set(seqs)) == len(seqs)

    def test_validation_report_populated(self, synthetic_loaded):
        report = synthetic_loaded.validation
        assert report["pool_size"] == len(synthetic_loaded.pool)
        assert report["dropped_missing"] == 0
        assert report["wild_type_length"] == len(synthetic_loaded.pool.wild_type)

    def test_subsampling_is_deterministic(self, synthetic_assay):
        kwargs = {
            "assay_dir": synthetic_assay["assay_dir"],
            "reference_csv": synthetic_assay["reference_csv"],
            "pool_size": 50,
        }
        a = load_assay(synthetic_assay["assay_id"], seed=7, **kwargs)
        b = load_assay(synthetic_assay["assay_id"], seed=7, **kwargs)
        assert a.pool.mutants == b.pool.mutants
        assert len(a.pool) == 50

    def test_missing_file_raises(self, synthetic_assay):
        with pytest.raises(FileNotFoundError):
            load_assay(
                "NOT_A_REAL_ASSAY",
                assay_dir=synthetic_assay["assay_dir"],
                reference_csv=synthetic_assay["reference_csv"],
            )

    def test_index_of_unmeasured_variant_is_none(self, synthetic_loaded):
        assert synthetic_loaded.pool.index_of("Z999Z") is None
        first = synthetic_loaded.pool.mutants[0]
        assert synthetic_loaded.pool.index_of(first) == 0


class TestCorruptedInput:
    """The loader must drop bad rows rather than silently trusting them."""

    def _write(self, tmp_path, rows, wild_type):
        assay_id = "CORRUPT_TEST_Fixture_2026"
        d = tmp_path / "assays"
        d.mkdir(exist_ok=True)
        with (d / f"{assay_id}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["mutant", "mutated_sequence", "DMS_score"]
            )
            w.writeheader()
            w.writerows(rows)
        ref = tmp_path / "ref.csv"
        with ref.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "DMS_id", "UniProt_ID", "molecule_name", "source_organism",
                    "selection_assay", "selection_type", "first_author", "title",
                    "year", "seq_len", "DMS_total_number_mutants",
                    "DMS_number_single_mutants", "DMS_number_multiple_mutants",
                    "target_seq", "region_mutated",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "DMS_id": assay_id, "UniProt_ID": "X", "molecule_name": "x",
                    "source_organism": "x", "selection_assay": "x",
                    "selection_type": "x", "first_author": "x", "title": "x",
                    "year": "2026", "seq_len": len(wild_type),
                    "DMS_total_number_mutants": len(rows),
                    "DMS_number_single_mutants": len(rows),
                    "DMS_number_multiple_mutants": 0,
                    "target_seq": wild_type, "region_mutated": "1",
                }
            )
        from drucion.data import proteingym

        proteingym._reference_table.cache_clear()
        return assay_id, d, str(ref)

    def test_drops_inconsistent_and_malformed_rows(self, tmp_path):
        wt = "MKTAYIAKQ"
        rows = [
            {"mutant": "K2R", "mutated_sequence": "MRTAYIAKQ", "DMS_score": 1.0},
            # sequence does not match the mutation spec
            {"mutant": "T3G", "mutated_sequence": "MKTAYIAKQ", "DMS_score": 0.5},
            # wrong length
            {"mutant": "A4G", "mutated_sequence": "MKTGYIAK", "DMS_score": 0.2},
            # non-canonical residue
            {"mutant": "Y5X", "mutated_sequence": "MKTAXIAKQ", "DMS_score": 0.1},
            # missing score
            {"mutant": "I6V", "mutated_sequence": "MKTAYVAKQ", "DMS_score": ""},
        ]
        assay_id, d, ref = self._write(tmp_path, rows, wt)
        assay = load_assay(assay_id, assay_dir=d, reference_csv=ref)
        assert assay.pool.mutants == ("K2R",)
        v = assay.validation
        assert v["dropped_mutation_sequence_inconsistent"] >= 1
        assert v["dropped_length_mismatch"] == 1
        assert v["dropped_missing"] == 1

    def test_all_rows_invalid_raises(self, tmp_path):
        wt = "MKTAYIAKQ"
        rows = [
            {"mutant": "T3G", "mutated_sequence": "MKTAYIAKQ", "DMS_score": 0.5}
        ]
        assay_id, d, ref = self._write(tmp_path, rows, wt)
        with pytest.raises(AssayValidationError, match="no valid variants"):
            load_assay(assay_id, assay_dir=d, reference_csv=ref)


class TestOracle:
    def test_rejects_non_finite_scores(self):
        with pytest.raises(AssayValidationError):
            ExperimentalOracle("x", [1.0, np.nan, 2.0])

    def test_query_returns_requested_values(self):
        oracle = ExperimentalOracle("x", [10.0, 20.0, 30.0])
        assert oracle.query([0, 2]).tolist() == [10.0, 30.0]

    def test_query_charges_each_variant_once(self):
        oracle = ExperimentalOracle("x", [1.0, 2.0, 3.0])
        oracle.query([0, 1])
        assert oracle.n_queries == 2
        oracle.query([1])  # repeat is free
        assert oracle.n_queries == 2
        oracle.query([2])
        assert oracle.n_queries == 3

    def test_out_of_range_query_raises(self):
        oracle = ExperimentalOracle("x", [1.0, 2.0])
        with pytest.raises(IndexError):
            oracle.query([5])
        with pytest.raises(IndexError):
            oracle.query([-1])

    def test_empty_query_is_free(self):
        oracle = ExperimentalOracle("x", [1.0, 2.0])
        assert oracle.query([]).size == 0
        assert oracle.n_queries == 0

    def test_peek_does_not_charge_budget(self):
        oracle = ExperimentalOracle("x", [1.0, 2.0, 3.0])
        assert oracle.peek_for_evaluation([0, 1]).tolist() == [1.0, 2.0]
        assert oracle.n_queries == 0

    def test_returned_arrays_are_copies(self):
        oracle = ExperimentalOracle("x", [1.0, 2.0])
        got = oracle.query([0])
        got[0] = 999.0
        assert oracle.query([0]).tolist() == [1.0]

    def test_top_quantile_indices(self):
        oracle = ExperimentalOracle("x", list(range(100)))
        top = oracle.top_quantile_indices(0.99)
        assert 99 in top
        assert 0 not in top


@pytest.mark.real_data
class TestRealProteinGym:
    """Runs against the actual downloaded assay files."""

    def test_primary_assay_loads_and_validates(self, require_real_data):
        assay = load_assay("A4GRB6_PSEAI_Chen_2020")
        assert len(assay.pool) > 4000
        assert assay.metadata.molecule_name == "Beta-lactamase VIM-2"
        assert assay.validation["dropped_mutation_sequence_inconsistent"] == 0
        assert assay.validation["dropped_bad_alphabet"] == 0
        assert assay.validation["dropped_length_mismatch"] == 0

    def test_real_sequences_match_mutation_specs(self, require_real_data):
        assay = load_assay("A4GRB6_PSEAI_Chen_2020")
        pool = assay.pool
        for mutant, seq in zip(pool.mutants[:500], pool.sequences[:500]):
            assert apply_mutations(pool.wild_type, mutant) == seq

    def test_secondary_assay_loads(self, require_real_data):
        assay = load_assay("TPMT_HUMAN_Matreyek_2018")
        assert len(assay.pool) > 3000
        assert assay.metadata.seq_len == len(assay.pool.wild_type)
