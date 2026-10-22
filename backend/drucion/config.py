"""Central configuration for Drucion.

All paths are resolved relative to the repository root so the package works
from any working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DRUCION_DATA_DIR", REPO_ROOT / "data"))
PROTEINGYM_DIR = DATA_DIR / "proteingym_raw" / "DMS_ProteinGym_substitutions"
REFERENCE_CSV = DATA_DIR / "proteingym_raw" / "DMS_substitutions.csv"
STRUCTURES_DIR = DATA_DIR / "structures_raw" / "ProteinGym_AF2_structures"
CACHE_DIR = Path(os.environ.get("DRUCION_CACHE_DIR", DATA_DIR / "cache"))
RESULTS_DIR = Path(os.environ.get("DRUCION_RESULTS_DIR", REPO_ROOT / "results"))
DB_PATH = Path(os.environ.get("DRUCION_DB", DATA_DIR / "drucion.sqlite"))
#: Optional ``KEY=value`` file for API credentials, read as a fallback when the
#: environment does not already define them. Gitignored.
ENV_FILE = Path(os.environ.get("DRUCION_ENV_FILE", REPO_ROOT / ".env"))

AcquisitionName = Literal["random", "greedy", "ucb", "ei", "evolutionary"]

#: Methods compared in the benchmark, in presentation order.
ALL_METHODS: tuple[str, ...] = ("random", "evolutionary", "greedy", "ei", "ucb")


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string. CPU always works; CUDA/MPS used when present."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class EmbeddingConfig:
    """Protein representation settings."""

    #: "esm2" = site-resolved ESM-2 features (default), "esm2_mean" = whole-
    #: sequence mean pooling, "onehot" = mutation one-hot baseline.
    featurizer: Literal["esm2", "esm2_mean", "onehot"] = "esm2"
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    device: str = "auto"
    batch_size: int = 16
    #: PCA target dimensionality for the surrogate's input space.
    n_components: int = 24
    #: Append the ESM-2 masked-marginal zero-shot score as an extra feature.
    #: Label-free, and measurably improves surrogate accuracy (see
    #: results/representation_check.json).
    include_zero_shot: bool = True
    #: Append an LLM-proposed structural prior as a further feature column.
    #: Same category of input as the zero-shot score: label-free, derived from
    #: pretrained knowledge, one scalar per variant. Off by default because it
    #: needs a Gemini API key; see embeddings/agent_prior.py.
    include_agent_prior: bool = False
    agent_prior_model: str = "gemini-3.8-flash"
    #: Withhold the protein's identity from the prompt, leaving only sequence
    #: and assay type. ProteinGym assays are published, so a prior that only
    #: works non-blind is recall rather than reasoning. Defaults to True: on
    #: TPMT the blind prior recovers 0.412 held-out rho against the named
    #: prior's 0.424, so withholding the identity costs almost nothing and
    #: removes the memorisation confound entirely. See
    #: results/representation_check_agent.json.
    agent_prior_blind: bool = True
    #: "agent" uses the prior as returned; "shuffled" permutes its scores
    #: across variants (same distribution, no information) and
    #: "random_positions" reassigns its tolerance labels to random positions.
    #: The latter two are the controls that make a gain interpretable.
    agent_prior_mode: Literal["agent", "shuffled", "random_positions"] = "agent"


@dataclass
class SurrogateConfig:
    """Surrogate fitness model settings."""

    kind: Literal["gp", "rf"] = "gp"
    #: Fall back to the random-forest ensemble if GP fitting raises.
    allow_fallback: bool = True
    max_fit_points: int = 2000
    jitter: float = 1e-6


@dataclass
class LoopConfig:
    """Closed-loop active-learning schedule."""

    n_initial: int = 50
    batch_size: int = 10
    n_rounds: int = 15
    seed: int = 0
    acquisition: AcquisitionName = "ucb"
    #: UCB exploration weight (posterior mean + beta * sigma).
    ucb_beta: float = 2.0
    #: Batch diversity: minimum normalised feature distance between picks in a
    #: batch, as a quantile of pairwise pool distances. 0 disables the filter.
    diversity_quantile: float = 0.05


@dataclass
class ExperimentConfig:
    """Full specification of one closed-loop run."""

    assay_id: str = "A4GRB6_PSEAI_Chen_2020"
    #: Cap the candidate pool for speed; None uses every measured variant.
    pool_size: int | None = None
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        return cls(
            assay_id=d.get("assay_id", "A4GRB6_PSEAI_Chen_2020"),
            pool_size=d.get("pool_size"),
            embedding=EmbeddingConfig(**d.get("embedding", {})),
            surrogate=SurrogateConfig(**d.get("surrogate", {})),
            loop=LoopConfig(**d.get("loop", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load a configuration from a YAML file (see ``configs/default.yaml``)."""
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @property
    def total_budget(self) -> int:
        """Total number of oracle measurements this run is allowed to consume."""
        return self.loop.n_initial + self.loop.batch_size * self.loop.n_rounds
