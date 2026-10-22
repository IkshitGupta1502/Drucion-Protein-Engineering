"""Wild-type structure loading, and the alignment guarantee the 3D view relies on.

The explorer highlights assay position *N* on structure residue *N*. If those
ever disagree, the picture is wrong in a way a viewer cannot detect — so the
loader refuses rather than rendering. These tests pin that refusal.
"""

from __future__ import annotations

import pytest

from drucion.data.structures import (
    StructureMisaligned,
    StructureUnavailable,
    has_structure,
    load_structure,
    read_ca_sequence,
    structure_path,
)


def _atom(serial: int, name: str, resname: str, chain: str, resnum: int) -> str:
    """Build one fixed-width PDB ATOM record.

    Columns matter here (1-indexed, per the PDB spec): 1-6 record name,
    7-11 serial, 13-16 atom name, 17 altLoc, 18-20 residue name, 22 chain,
    23-26 residue number, 31-38 x. Getting these wrong is the classic way to
    "parse" a PDB into nonsense, so the helper is written against the spec.
    """
    name = f"{name:<4s}"[:4]
    return (
        f"ATOM  "           # cols 1-6
        f"{serial:>5d}"      # cols 7-11
        f" "                 # col 12
        f"{name}"            # cols 13-16
        f" "                 # col 17 (altLoc)
        f"{resname:>3s}"     # cols 18-20
        f" "                 # col 21
        f"{chain}"           # col 22
        f"{resnum:>4d}"      # cols 23-26
        f"    "              # cols 27-30 (iCode + padding)
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}"  # cols 31-54
        f"{1.0:6.2f}{0.0:6.2f}"            # cols 55-66
    )


class TestReadCaSequence:
    def test_reads_ca_atoms_in_residue_order(self):
        pdb = "\n".join(
            [
                _atom(1, " CA ", "MET", "A", 1),
                _atom(2, " CA ", "LYS", "A", 2),
                _atom(3, " CA ", "THR", "A", 3),
            ]
        )
        assert read_ca_sequence(pdb) == {1: "M", 2: "K", 3: "T"}

    def test_ignores_non_ca_atoms(self):
        pdb = "\n".join(
            [
                _atom(1, " N  ", "MET", "A", 1),
                _atom(2, " CA ", "MET", "A", 1),
                _atom(3, " CB ", "MET", "A", 1),
            ]
        )
        assert read_ca_sequence(pdb) == {1: "M"}

    def test_ignores_non_atom_records(self):
        pdb = "\n".join(
            ["HEADER    SOMETHING", _atom(1, " CA ", "ALA", "A", 1), "END"]
        )
        assert read_ca_sequence(pdb) == {1: "A"}

    def test_reads_only_the_first_chain(self):
        """Merging chains would corrupt the numbering the view depends on."""
        pdb = "\n".join(
            [
                _atom(1, " CA ", "MET", "A", 1),
                _atom(2, " CA ", "LYS", "A", 2),
                _atom(3, " CA ", "TRP", "B", 1),
            ]
        )
        assert read_ca_sequence(pdb) == {1: "M", 2: "K"}

    def test_unknown_residue_becomes_x(self):
        assert read_ca_sequence(_atom(1, " CA ", "UNK", "A", 1)) == {1: "X"}

    def test_empty_input_returns_empty(self):
        assert read_ca_sequence("") == {}


class TestStructurePath:
    def test_builds_path_from_uniprot_id(self, tmp_path):
        assert structure_path("FOO_BAR", tmp_path) == tmp_path / "FOO_BAR.pdb"

    def test_has_structure_reflects_disk(self, tmp_path):
        assert not has_structure("FOO_BAR", tmp_path)
        (tmp_path / "FOO_BAR.pdb").write_text("")
        assert has_structure("FOO_BAR", tmp_path)


@pytest.mark.real_data
class TestRealStructures:
    ASSAYS = [
        "A4GRB6_PSEAI_Chen_2020",
        "BLAT_ECOLX_Firnberg_2014",
        "TPMT_HUMAN_Matreyek_2018",
    ]

    @pytest.mark.parametrize("assay_id", ASSAYS)
    def test_structure_loads(self, assay_id, require_real_data):
        st = load_structure(assay_id)
        assert st.assay_id == assay_id
        assert st.pdb_text.startswith("ATOM")
        assert st.n_residues > 100

    @pytest.mark.parametrize("assay_id", ASSAYS)
    def test_numbering_starts_at_one_and_is_contiguous(
        self, assay_id, require_real_data
    ):
        st = load_structure(assay_id)
        assert st.first_residue == 1
        assert len(st.sequence) == st.n_residues

    @pytest.mark.parametrize("assay_id", ASSAYS)
    def test_structure_sequence_equals_assay_wild_type(
        self, assay_id, require_real_data
    ):
        """The guarantee the 3D highlighting depends on."""
        from drucion.data.proteingym import load_assay_metadata

        st = load_structure(assay_id)
        assert st.sequence == load_assay_metadata(assay_id).wild_type

    @pytest.mark.parametrize("assay_id", ASSAYS)
    def test_every_mutated_position_exists_in_the_structure(
        self, assay_id, require_real_data
    ):
        """Every position the optimizer can pick must be renderable."""
        from drucion.data.proteingym import load_assay, parse_mutant

        st = load_structure(assay_id)
        pool = load_assay(assay_id).pool
        positions = {s.position for m in pool.mutants for s in parse_mutant(m)}
        assert positions
        assert max(positions) <= st.n_residues
        assert min(positions) >= 1

    def test_wild_type_residue_matches_at_mutated_positions(
        self, require_real_data
    ):
        """Position N in a mutant spec must be residue N in the structure."""
        from drucion.data.proteingym import load_assay, parse_mutant

        assay_id = "A4GRB6_PSEAI_Chen_2020"
        st = load_structure(assay_id)
        pool = load_assay(assay_id).pool
        for mutant in pool.mutants[:300]:
            for sub in parse_mutant(mutant):
                assert st.sequence[sub.position - 1] == sub.wt_aa, mutant


class TestRefusals:
    def test_missing_structure_raises(self, tmp_path, require_real_data):
        load_structure.cache_clear()
        with pytest.raises(StructureUnavailable, match="No structure"):
            load_structure("A4GRB6_PSEAI_Chen_2020", str(tmp_path))
        load_structure.cache_clear()

    def test_misaligned_sequence_is_refused(self, tmp_path, require_real_data):
        """A structure whose residues disagree with the assay must not render."""
        from drucion.data.proteingym import load_assay_metadata

        meta = load_assay_metadata("A4GRB6_PSEAI_Chen_2020")
        wt = meta.wild_type
        three = {v: k for k, v in {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        }.items()}
        # Same length, but residue 5 is deliberately wrong.
        corrupted = list(wt)
        corrupted[4] = "W" if wt[4] != "W" else "A"
        lines = [
            _atom(i + 1, " CA ", three[aa], "A", i + 1)
            for i, aa in enumerate(corrupted)
        ]
        (tmp_path / f"{meta.uniprot_id}.pdb").write_text("\n".join(lines))

        load_structure.cache_clear()
        with pytest.raises(StructureMisaligned, match="differ"):
            load_structure("A4GRB6_PSEAI_Chen_2020", str(tmp_path))
        load_structure.cache_clear()

    def test_wrong_residue_numbering_is_refused(self, tmp_path, require_real_data):
        """Numbering that doesn't start at 1 would shift every highlight."""
        from drucion.data.proteingym import load_assay_metadata

        meta = load_assay_metadata("A4GRB6_PSEAI_Chen_2020")
        three = {
            "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
            "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
            "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
            "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
        }
        # Correct sequence, but numbered from 27 (a realistic construct offset).
        lines = [
            _atom(i + 1, " CA ", three[aa], "A", i + 27)
            for i, aa in enumerate(meta.wild_type)
        ]
        (tmp_path / f"{meta.uniprot_id}.pdb").write_text("\n".join(lines))

        load_structure.cache_clear()
        with pytest.raises(StructureMisaligned, match="numbered"):
            load_structure("A4GRB6_PSEAI_Chen_2020", str(tmp_path))
        load_structure.cache_clear()


@pytest.mark.real_data
class TestStructureEndpoint:
    def test_serves_pdb(self, require_real_data):
        from fastapi.testclient import TestClient

        from drucion.api.app import app

        with TestClient(app) as client:
            r = client.get("/api/assays/A4GRB6_PSEAI_Chen_2020/structure")
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("chemical/x-pdb")
            assert r.headers["X-Structure-Residues"] == "266"
            assert "AlphaFold2" in r.headers["X-Structure-Source"]
            assert r.text.startswith("ATOM")

    def test_unknown_assay_is_404(self, require_real_data):
        from fastapi.testclient import TestClient

        from drucion.api.app import app

        with TestClient(app) as client:
            assert client.get("/api/assays/NOPE/structure").status_code == 404

    def test_detail_reports_structure_availability(self, require_real_data):
        from fastapi.testclient import TestClient

        from drucion.api.app import app

        with TestClient(app) as client:
            body = client.get("/api/assays/A4GRB6_PSEAI_Chen_2020").json()
            assert body["has_structure"] is True
