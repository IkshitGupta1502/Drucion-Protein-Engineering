#!/usr/bin/env python
"""Download the ProteinGym substitution benchmark (DMS assays + reference table).

Downloads ~43 MB and expands to ~1 GB of assay CSVs under ``data/``.

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --force     # re-download even if present
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile

from drucion.config import DATA_DIR, PROTEINGYM_DIR, REFERENCE_CSV

VERSION = "v1.3"
ZIP_URL = (
    f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{VERSION}/"
    "DMS_ProteinGym_substitutions.zip"
)
REFERENCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
    "reference_files/DMS_substitutions.csv"
)


def _report(block: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(block * block_size, total)
    pct = 100 * done / total
    sys.stdout.write(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)")
    sys.stdout.flush()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    raw_dir = DATA_DIR / "proteingym_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if REFERENCE_CSV.exists() and not args.force:
        print(f"Reference table already present: {REFERENCE_CSV}")
    else:
        print(f"Downloading reference table from {REFERENCE_URL}")
        urllib.request.urlretrieve(REFERENCE_URL, REFERENCE_CSV)
        print(f"  saved to {REFERENCE_CSV}")

    if PROTEINGYM_DIR.exists() and any(PROTEINGYM_DIR.glob("*.csv")) and not args.force:
        n = len(list(PROTEINGYM_DIR.glob("*.csv")))
        print(f"Assays already present: {n} files in {PROTEINGYM_DIR}")
        return 0

    zip_path = raw_dir / "DMS_ProteinGym_substitutions.zip"
    print(f"Downloading ProteinGym {VERSION} substitution assays (~43 MB)")
    print(f"  {ZIP_URL}")
    urllib.request.urlretrieve(ZIP_URL, zip_path, reporthook=_report)
    print()

    print(f"Extracting to {raw_dir}")
    if PROTEINGYM_DIR.exists() and args.force:
        shutil.rmtree(PROTEINGYM_DIR)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()

    n = len(list(PROTEINGYM_DIR.glob("*.csv")))
    print(f"Done: {n} assay files available in {PROTEINGYM_DIR}")
    if n == 0:
        print("WARNING: no CSV files found — the archive layout may have changed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
