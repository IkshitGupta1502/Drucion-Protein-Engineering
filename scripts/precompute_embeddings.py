#!/usr/bin/env python
"""Precompute and cache ESM-2 embeddings for one or more ProteinGym assays.

Usage::

    python scripts/precompute_embeddings.py BLAT_ECOLX_Stiffler_2015 TPMT_HUMAN_Matreyek_2018
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from drucion.config import EmbeddingConfig
from drucion.data.proteingym import load_assay
from drucion.embeddings.esm import ESM2Embedder


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("assays", nargs="+", help="ProteinGym DMS_id values")
    p.add_argument("--model", default=EmbeddingConfig().model_name)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pool-size", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    embedder = ESM2Embedder(model_name=args.model, batch_size=args.batch_size)

    for assay_id in args.assays:
        assay = load_assay(assay_id, pool_size=args.pool_size)
        n = len(assay.pool)
        print(f"[{assay_id}] pool={n} device={embedder.device}", flush=True)
        t0 = time.time()
        last = [0.0]

        def progress(done: int, total: int) -> None:
            now = time.time()
            if now - last[0] > 5 or done == total:
                last[0] = now
                rate = done / max(now - t0, 1e-9)
                print(
                    f"[{assay_id}] {done}/{total} ({rate:.1f} seq/s)",
                    flush=True,
                )

        X = embedder.embed_sequences(assay.pool.sequences, progress=progress)
        print(
            f"[{assay_id}] done shape={X.shape} in {time.time() - t0:.1f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
