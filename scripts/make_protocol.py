#!/usr/bin/env python
"""Generate a bench protocol for one assay, with provenance on every number.

Values come from the assay's own published conditions and UniProt first; a
language model is asked only for what those did not supply, and anything it
returns is labelled an estimate and carries an experiment that would confirm
it. ``--no-llm`` omits estimates entirely.

Usage::

    python scripts/make_protocol.py --assay A4GRB6_PSEAI_Chen_2020
    python scripts/make_protocol.py --assay TPMT_HUMAN_Matreyek_2018 --no-llm
    python scripts/make_protocol.py --assay BLAT_ECOLX_Firnberg_2014 \
        --variants A42G H24C D233H --out protocol.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from drucion.config import RESULTS_DIR
from drucion.data.proteingym import load_assay_metadata
from drucion.protocol import generate_protocol, render_markdown


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assay", required=True)
    p.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="variant specs to include as this round's sample list",
    )
    p.add_argument(
        "--variants-from",
        type=Path,
        default=None,
        help="read the sample list from a file, one variant per line",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="records and standard technique only; no estimates",
    )
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write to this path (default: stdout)",
    )
    p.add_argument("--save", action="store_true", help=f"write into {RESULTS_DIR}")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    worklist = list(args.variants or [])
    if args.variants_from:
        worklist += [
            line.strip()
            for line in args.variants_from.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    metadata = load_assay_metadata(args.assay)
    protocol = generate_protocol(
        metadata,
        worklist=worklist,
        use_llm=not args.no_llm,
        model=args.model,
    )

    text = (
        json.dumps(protocol.to_dict(), indent=2)
        if args.json
        else render_markdown(protocol)
    )

    out = args.out
    if args.save and out is None:
        suffix = "json" if args.json else "md"
        out = RESULTS_DIR / f"{args.assay}_protocol.{suffix}"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)

    counts = protocol.tier_counts()
    print(
        f"[{counts['sourced']} sourced, {counts['estimated']} estimates] "
        f"compiled from: {protocol.generated_by}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
