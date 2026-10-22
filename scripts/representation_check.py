#!/usr/bin/env python
"""One-time representation sanity check (reported, not tuned against).

Fits the surrogate on a random label-free subset and measures Spearman
correlation on the disjoint remainder, for each candidate representation. This
is a *modelling* decision made once and documented; the active-learning
benchmark itself never sees labels outside the oracle's budget.

Usage::

    python scripts/representation_check.py --assay TPMT_HUMAN_Matreyek_2018
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from drucion.config import RESULTS_DIR, EmbeddingConfig, SurrogateConfig
from drucion.data.proteingym import load_assay
from drucion.embeddings.features import build_features
from drucion.models.surrogate import build_surrogate


def evaluate(X: np.ndarray, y: np.ndarray, n_train: int, seeds: int = 3) -> dict:
    rhos, rmses = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        tr, te = perm[:n_train], perm[n_train:]
        sur = build_surrogate(SurrogateConfig(), seed=seed)
        sur.fit(X[tr], y[tr])
        mean, _ = sur.predict(X[te])
        rhos.append(float(spearmanr(mean, y[te]).statistic))
        rmses.append(float(np.sqrt(np.mean((mean - y[te]) ** 2))))
    return {
        "spearman_mean": round(float(np.mean(rhos)), 4),
        "spearman_std": round(float(np.std(rhos)), 4),
        "rmse_mean": round(float(np.mean(rmses)), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assay", default="TPMT_HUMAN_Matreyek_2018")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--components", type=int, default=24)
    p.add_argument(
        "--featurizers", nargs="+", default=["esm2", "esm2_mean", "onehot"]
    )
    p.add_argument(
        "--save", action="store_true", help="write results/representation_check.json"
    )
    args = p.parse_args()

    assay = load_assay(args.assay)
    y = assay.oracle.all_scores_for_evaluation()
    results = {}
    # Each representation is evaluated with and without the appended ESM-2
    # zero-shot score, which isolates that feature's contribution and keeps
    # the one-hot row a genuine no-pretrained-knowledge control.
    for feat in args.featurizers:
        for with_zs in (False, True):
            key = f"{feat}+zero_shot" if with_zs else feat
            cfg = EmbeddingConfig(
                featurizer=feat,
                n_components=args.components,
                include_zero_shot=with_zs,
            )
            fs = build_features(assay.pool, cfg)
            results[key] = evaluate(fs.X, y, args.n_train) | {
                "raw_dim": fs.raw_dim,
                "n_features": int(fs.X.shape[1]),
                "explained_variance": round(fs.explained_variance_ratio, 3),
            }
            print(key, json.dumps(results[key]), flush=True)

    payload = {
        "assay": args.assay,
        "n_train": args.n_train,
        "n_components": args.components,
        "note": (
            "Held-out Spearman for each representation, fit on a random "
            "label-free subset and evaluated on the disjoint remainder. This "
            "is a one-time modelling decision, made on this assay and then "
            "applied unchanged elsewhere; the active-learning benchmark never "
            "sees labels outside its budget."
        ),
        "results": results,
    }
    print(json.dumps(payload, indent=2))

    if args.save:
        out = RESULTS_DIR / "representation_check.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
