"""Trainable surrogate fitness models with calibrated uncertainty.

Both surrogates implement the same interface::

    fit(X_observed, y_observed) -> self
    predict(X_candidates)       -> (mean, std)

They are trained **only** on variants whose measurements the oracle has
already revealed. Nothing here can see a hidden label.

``GPSurrogate``
    Exact Gaussian process (BoTorch ``SingleTaskGP``) in double precision with
    input normalisation and outcome standardisation. Principled posterior
    uncertainty, which is what the Bayesian acquisition functions need.

``RFSurrogate``
    Random-forest ensemble whose per-tree spread is used as the uncertainty
    estimate. Faster and numerically bulletproof; used as a documented
    fallback when exact GP fitting fails to converge, and available directly
    for very large budgets.
"""

from __future__ import annotations

import logging
import warnings
from typing import Protocol

import numpy as np

from ..config import SurrogateConfig

logger = logging.getLogger(__name__)


class Surrogate(Protocol):
    """Structural type every surrogate satisfies."""

    kind: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate": ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def _check_training_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    if X.shape[0] < 2:
        raise ValueError("Surrogate needs at least 2 observations to fit")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Training data contains non-finite values")
    return X, y


class GPSurrogate:
    """Exact GP regression over PCA-reduced protein features."""

    kind = "gp"

    def __init__(self, config: SurrogateConfig | None = None) -> None:
        self.config = config or SurrogateConfig()
        self._model = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self.n_train = 0
        self.fit_info: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        X, y = _check_training_data(X, y)

        # Subsample for tractability if the observed set grows very large.
        if X.shape[0] > self.config.max_fit_points:
            rng = np.random.default_rng(0)
            keep = rng.choice(X.shape[0], self.config.max_fit_points, replace=False)
            X, y = X[keep], y[keep]

        self.n_train = int(X.shape[0])
        # Degenerate targets (all observations identical) break standardisation.
        if float(np.std(y)) < 1e-12:
            self._model = None
            self._y_mean = float(np.mean(y))
            self._y_std = 0.0
            self.fit_info = {"degenerate_targets": True, "n_train": self.n_train}
            return self

        tX = torch.from_numpy(X).to(torch.float64)
        ty = torch.from_numpy(y).to(torch.float64).unsqueeze(-1)

        model = SingleTaskGP(
            train_X=tX,
            train_Y=ty,
            input_transform=Normalize(d=tX.shape[-1]),
            outcome_transform=Standardize(m=1),
        ).to(torch.float64)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
        model.eval()
        self._model = model
        self._torch = torch
        self.fit_info = {
            "n_train": self.n_train,
            "noise": float(model.likelihood.noise.mean().item()),
            "degenerate_targets": False,
        }
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if self._model is None:
            # Degenerate fit: constant mean, zero information.
            mean = np.full(X.shape[0], self._y_mean, dtype=np.float64)
            std = np.ones(X.shape[0], dtype=np.float64)
            return mean, std

        torch = self._torch
        tX = torch.from_numpy(X).to(torch.float64)
        with torch.no_grad():
            posterior = self._model.posterior(tX)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            var = posterior.variance.squeeze(-1).clamp_min(self.config.jitter)
            std = var.sqrt().cpu().numpy()
        return (
            np.nan_to_num(mean, nan=self._y_mean),
            np.nan_to_num(std, nan=1.0, posinf=1.0),
        )


class RFSurrogate:
    """Random-forest ensemble; uncertainty = spread across trees."""

    kind = "rf"

    def __init__(self, config: SurrogateConfig | None = None, seed: int = 0) -> None:
        self.config = config or SurrogateConfig()
        self.seed = seed
        self._model = None
        self._y_mean = 0.0
        self.n_train = 0
        self.fit_info: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFSurrogate":
        from sklearn.ensemble import RandomForestRegressor

        X, y = _check_training_data(X, y)
        self.n_train = int(X.shape[0])
        self._y_mean = float(np.mean(y))
        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            max_features=0.7,
            bootstrap=True,
            random_state=self.seed,
            n_jobs=-1,
        )
        model.fit(X, y)
        self._model = model
        self.fit_info = {"n_train": self.n_train, "n_estimators": 200}
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if self._model is None:
            raise RuntimeError("RFSurrogate.predict called before fit")
        per_tree = np.stack(
            [est.predict(X) for est in self._model.estimators_], axis=0
        )
        mean = per_tree.mean(axis=0)
        std = per_tree.std(axis=0)
        # A forest can be perfectly confident; floor the std so acquisition
        # functions stay well defined.
        std = np.maximum(std, 1e-6)
        return mean, std


class FallbackSurrogate:
    """Tries the GP first, transparently degrades to the forest on failure.

    The active fallback is recorded in :attr:`kind` and reported through the
    API so a run never silently claims to be using a GP when it is not.
    """

    def __init__(self, config: SurrogateConfig | None = None, seed: int = 0) -> None:
        self.config = config or SurrogateConfig()
        self.seed = seed
        self._impl: Surrogate = (
            GPSurrogate(self.config)
            if self.config.kind == "gp"
            else RFSurrogate(self.config, seed)
        )
        self.kind = self._impl.kind
        self.fallback_reason: str | None = None

    @property
    def n_train(self) -> int:
        return getattr(self._impl, "n_train", 0)

    @property
    def fit_info(self) -> dict:
        info = dict(getattr(self._impl, "fit_info", {}))
        info["kind"] = self.kind
        if self.fallback_reason:
            info["fallback_reason"] = self.fallback_reason
        return info

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FallbackSurrogate":
        try:
            self._impl.fit(X, y)
            self.kind = self._impl.kind
            return self
        except Exception as exc:  # noqa: BLE001 - deliberate safety net
            if not (self.config.allow_fallback and self.config.kind == "gp"):
                raise
            logger.warning("GP fit failed (%s); falling back to RF ensemble", exc)
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            self._impl = RFSurrogate(self.config, self.seed)
            self._impl.fit(X, y)
            self.kind = self._impl.kind
            return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self._impl.predict(X)
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
            raise RuntimeError("Surrogate produced non-finite predictions")
        return mean, std


def build_surrogate(
    config: SurrogateConfig | None = None, seed: int = 0
) -> FallbackSurrogate:
    return FallbackSurrogate(config, seed)
