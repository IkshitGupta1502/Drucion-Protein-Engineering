"""Pydantic request/response models for the Drucion API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

AcquisitionName = Literal["random", "greedy", "ucb", "ei", "evolutionary"]
FeaturizerName = Literal["esm2", "esm2_mean", "onehot"]


class AssaySummary(BaseModel):
    assay_id: str
    molecule_name: str
    source_organism: str
    selection_type: str
    selection_assay: str
    seq_len: int
    n_mutants: int
    n_single_mutants: int
    year: str
    available: bool


class AssayDetail(BaseModel):
    assay_id: str
    uniprot_id: str
    molecule_name: str
    source_organism: str
    selection_assay: str
    selection_type: str
    first_author: str
    title: str
    year: str
    seq_len: int
    n_mutants: int
    n_single_mutants: int
    n_multiple_mutants: int
    wild_type: str
    region_mutated: str
    source: str
    license: str
    #: Validation report from loading the real CSV.
    validation: dict[str, Any] | None = None
    pool_size: int | None = None
    #: Pool-level score statistics. Shown so users can see the achievable
    #: range; the optimizer never receives these.
    pool_stats: dict[str, float] | None = None


class LoopSettings(BaseModel):
    n_initial: int = Field(50, ge=2, le=2000)
    batch_size: int = Field(10, ge=1, le=500)
    n_rounds: int = Field(15, ge=1, le=200)
    seed: int = Field(0, ge=0, le=10_000)
    acquisition: AcquisitionName = "ucb"
    ucb_beta: float = Field(2.0, ge=0.0, le=20.0)
    diversity_quantile: float = Field(0.05, ge=0.0, le=0.5)


class ExperimentRequest(BaseModel):
    assay_id: str = "A4GRB6_PSEAI_Chen_2020"
    pool_size: int | None = Field(None, ge=50, le=200_000)
    featurizer: FeaturizerName = "esm2"
    surrogate: Literal["gp", "rf"] = "gp"
    n_components: int = Field(24, ge=2, le=256)
    loop: LoopSettings = Field(default_factory=LoopSettings)

    @field_validator("assay_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("assay_id must not be empty")
        return v


class VariantOut(BaseModel):
    index: int
    mutant: str
    positions: list[int]
    predicted_fitness: float | None
    predicted_uncertainty: float | None
    acquisition_score: float
    measured_fitness: float


class RoundOut(BaseModel):
    round: int
    n_measured: int
    best_measured: float
    mean_measured: float
    selected: list[VariantOut] = []
    surrogate_kind: str = ""
    surrogate_info: dict[str, Any] = {}
    surrogate_spearman: float | None = None
    surrogate_rmse: float | None = None
    holdout_spearman: float | None = None
    holdout_rmse: float | None = None
    proposal_report: dict[str, Any] | None = None
    wall_time_s: float = 0.0


class ExperimentOut(BaseModel):
    id: str
    status: str
    assay_id: str
    acquisition: str
    round: int
    n_rounds: int
    n_measured: int
    budget: int
    best_measured: float | None = None
    best_mutant: str | None = None
    pool_size: int | None = None
    surrogate_kind: str | None = None
    features: dict[str, Any] | None = None
    learning_curve: dict[str, list[float]] | None = None
    error: str | None = None
    config: dict[str, Any] | None = None


class ExperimentResults(BaseModel):
    id: str
    status: str
    assay_id: str
    acquisition: str
    n_measured: int
    budget: int
    best_measured: float | None
    best_mutant: str | None
    wild_type: str | None = None
    best_sequence: str | None = None
    #: Position-level diff between wild type and the best discovered variant.
    differences: list[dict[str, Any]] = []
    learning_curve: dict[str, list[float]] | None = None
    rounds: list[RoundOut] = []
    pool_stats: dict[str, float] | None = None


class BenchmarkRequest(BaseModel):
    assay_id: str = "A4GRB6_PSEAI_Chen_2020"
    methods: list[AcquisitionName] = ["random", "evolutionary", "greedy", "ei", "ucb"]
    n_seeds: int = Field(5, ge=1, le=30)
    pool_size: int | None = Field(None, ge=50, le=200_000)
    featurizer: FeaturizerName = "esm2"
    surrogate: Literal["gp", "rf"] = "gp"
    loop: LoopSettings = Field(default_factory=LoopSettings)
    holdout_size: int = Field(500, ge=0, le=5000)

    @field_validator("methods")
    @classmethod
    def _at_least_one(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one method is required")
        return list(dict.fromkeys(v))


class BenchmarkOut(BaseModel):
    id: str
    status: str
    assay_id: str
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
