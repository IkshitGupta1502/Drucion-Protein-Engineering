"""Drucion — closed-loop, uncertainty-aware active learning for protein engineering.

Drucion decides which protein variant to measure next. It embeds variants with
ESM-2, fits a Gaussian-process surrogate on the measurements it has already
paid for, estimates its own uncertainty, selects the most informative batch of
untested variants, reveals their real measured fitness, retrains, and repeats.

Every fitness value it sees is a real, previously published deep-mutational-
scanning measurement from ProteinGym. This is a **retrospective simulation of
experimental selection** — no wet-lab experiment is performed, and no fitness
value is ever invented.

Typical use::

    from drucion.config import ExperimentConfig
    from drucion.data.proteingym import load_assay
    from drucion.embeddings.features import build_features
    from drucion.optimization.loop import ClosedLoopOptimizer

    config = ExperimentConfig()                 # VIM-2, UCB, 50 + 10x15
    assay = load_assay(config.assay_id)         # pool (no labels) + sealed oracle
    features = build_features(assay.pool, config.embedding)
    optimizer = ClosedLoopOptimizer(assay, config, features=features)

    for record in optimizer.run():
        print(record.round, record.best_measured)

See ``README.md`` for results and ``SUMMARY.md`` for a candid account of what
was tested, what worked, and what did not.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
