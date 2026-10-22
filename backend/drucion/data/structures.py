"""Wild-type protein structures for the 3D explorer.

Structures are the AlphaFold2 models distributed with ProteinGym, one per
target protein. They are used **only to position residues in space** so the
dashboard can show where on the protein the optimizer spent its budget.

Scientific framing
------------------
Three things this module does *not* do, deliberately:

* It never produces a structure for a mutant. Only the wild type has one. A
  single substitution does not visibly change a backbone, and rendering an
  invented mutant fold would be fabricated data.
* It never predicts structure. The PDB files are read from disk as shipped.
* It never renders a structure whose residue numbering disagrees with the
  assay's mutation positions. :func:`load_structure` validates the alignment
  and refuses on mismatch, because highlighting residue *N* on a structure
  whose residue *N* is a different amino acid would be actively misleading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import STRUCTURES_DIR
from .proteingym import load_assay_metadata

logger = logging.getLogger(__name__)

#: Three-letter to one-letter amino acid codes, standard residues only.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class StructureUnavailable(FileNotFoundError):
    """No structure file on disk for this protein."""


class StructureMisaligned(ValueError):
    """The structure's residue numbering disagrees with the assay sequence."""


@dataclass(frozen=True)
class Structure:
    """A wild-type structure whose numbering is verified against the assay."""

    assay_id: str
    uniprot_id: str
    pdb_text: str
    n_residues: int
    first_residue: int
    #: Sequence read out of the structure, for the alignment check.
    sequence: str
    source: str = "AlphaFold2 model distributed with ProteinGym v1.3"

    def to_dict(self) -> dict:
        return {
            "assay_id": self.assay_id,
            "uniprot_id": self.uniprot_id,
            "n_residues": self.n_residues,
            "first_residue": self.first_residue,
            "source": self.source,
        }


def structure_path(uniprot_id: str, directory: Path | None = None) -> Path:
    base = Path(directory or STRUCTURES_DIR)
    return base / f"{uniprot_id}.pdb"


def has_structure(uniprot_id: str, directory: Path | None = None) -> bool:
    return structure_path(uniprot_id, directory).exists()


def read_ca_sequence(pdb_text: str) -> dict[int, str]:
    """Map residue number -> one-letter code, from CA atoms of the first chain.

    Only the first chain is read: the ProteinGym models are single-chain, and
    silently merging chains would corrupt the numbering.
    """
    out: dict[int, str] = {}
    chain: str | None = None
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        this_chain = line[21]
        if chain is None:
            chain = this_chain
        elif this_chain != chain:
            continue
        resname = line[17:20].strip()
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        out[resnum] = THREE_TO_ONE.get(resname, "X")
    return out


@lru_cache(maxsize=8)
def load_structure(
    assay_id: str, directory: str | None = None
) -> Structure:
    """Load and verify the wild-type structure for ``assay_id``.

    Raises
    ------
    StructureUnavailable
        No PDB file for this protein.
    StructureMisaligned
        The structure's residues do not match the assay's wild-type sequence,
        so mutation positions could not be mapped onto it safely.
    """
    meta = load_assay_metadata(assay_id)
    path = structure_path(meta.uniprot_id, Path(directory) if directory else None)
    if not path.exists():
        raise StructureUnavailable(
            f"No structure for {meta.uniprot_id} at {path}. "
            "Run `python scripts/download_data.py --structures`."
        )

    pdb_text = path.read_text()
    residues = read_ca_sequence(pdb_text)
    if not residues:
        raise StructureMisaligned(f"No CA atoms parsed from {path}")

    numbers = sorted(residues)
    sequence = "".join(residues[n] for n in numbers)
    wt = meta.wild_type

    # The dashboard highlights assay position P on structure residue P, so the
    # two must agree residue for residue. Anything else is refused rather than
    # silently drawn in the wrong place.
    if numbers != list(range(1, len(wt) + 1)):
        raise StructureMisaligned(
            f"{meta.uniprot_id}: structure residues are numbered "
            f"{numbers[0]}..{numbers[-1]} ({len(numbers)} residues) but the "
            f"assay sequence is 1..{len(wt)}"
        )
    mismatches = [
        (i + 1, wt[i], sequence[i]) for i in range(len(wt)) if wt[i] != sequence[i]
    ]
    if mismatches:
        preview = ", ".join(f"{p}:{a}≠{b}" for p, a, b in mismatches[:5])
        raise StructureMisaligned(
            f"{meta.uniprot_id}: {len(mismatches)} residue(s) differ between "
            f"the structure and the assay wild type ({preview})"
        )

    return Structure(
        assay_id=assay_id,
        uniprot_id=meta.uniprot_id,
        pdb_text=pdb_text,
        n_residues=len(numbers),
        first_residue=numbers[0],
        sequence=sequence,
    )
