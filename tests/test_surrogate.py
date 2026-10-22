"""Surrogate shapes, finiteness, learning ability and fallback behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from drucion.config import SurrogateConfig
from drucion.models.surrogate import (
    GPSurrogate,
    RFSurrogate,
    build_surrogate,
)


def _toy(n=60, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.standard_normal(n)
    return X, y


ALL = [GPSurrogate, RFSurrogate, build_surrogate]


def _make(factory, seed=0):
    if factory is build_surrogate:
        return build_surrogate(SurrogateConfig(), seed=seed)
    if factory is RFSurrogate:
        return RFSurrogate(SurrogateConfig(), seed=seed)
    return GPSurrogate(SurrogateConfig())


@pytest.mark.parametrize("factory", ALL, ids=["gp", "rf", "fallback"])
class TestSurrogateContract:
    def test_predict_returns_matching_shapes(self, factory):
        X, y = _toy()
        sur = _make(factory).fit(X, y)
        mean, std = sur.predict(X[:13])
        assert mean.shape == (13,)
        assert std.shape == (13,)

    def test_predictions_are_finite(self, factory):
        X, y = _toy()
        sur = _make(factory).fit(X, y)
        mean, std = sur.predict(X)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))

    def test_uncertainty_is_non_negative(self, factory):
        X, y = _toy()
        sur = _make(factory).fit(X, y)
        _, std = sur.predict(X)
        assert np.all(std >= 0)

    def test_learns_a_learnable_signal(self, factory):
        """A surrogate that cannot beat a constant predictor is useless."""
        from scipy.stats import spearmanr

        X, y = _toy(n=120, seed=1)
        Xtr, ytr, Xte, yte = X[:80], y[:80], X[80:], y[80:]
        sur = _make(factory).fit(Xtr, ytr)
        mean, _ = sur.predict(Xte)
        assert spearmanr(mean, yte).statistic > 0.6

    def test_fit_returns_self(self, factory):
        X, y = _toy()
        sur = _make(factory)
        assert sur.fit(X, y) is sur

    def test_rejects_mismatched_lengths(self, factory):
        X, y = _toy()
        with pytest.raises(ValueError):
            _make(factory).fit(X, y[:-1])

    def test_rejects_too_few_points(self, factory):
        X, y = _toy()
        with pytest.raises(ValueError):
            _make(factory).fit(X[:1], y[:1])

    def test_rejects_non_finite_training_data(self, factory):
        X, y = _toy()
        X = X.copy()
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            _make(factory).fit(X, y)

    def test_rejects_one_dimensional_X(self, factory):
        X, y = _toy()
        with pytest.raises(ValueError):
            _make(factory).fit(X[:, 0], y)

    def test_reproducible_with_fixed_seed(self, factory):
        X, y = _toy()
        m1, s1 = _make(factory, seed=3).fit(X, y).predict(X[:10])
        m2, s2 = _make(factory, seed=3).fit(X, y).predict(X[:10])
        assert m1 == pytest.approx(m2)
        assert s1 == pytest.approx(s2)


class TestGPSpecifics:
    def test_uses_double_precision(self):
        X, y = _toy()
        gp = GPSurrogate().fit(X, y)
        assert gp._model.train_inputs[0].dtype.__str__() == "torch.float64"

    def test_handles_degenerate_constant_targets(self):
        X, _ = _toy()
        y = np.full(X.shape[0], 3.0)
        gp = GPSurrogate().fit(X, y)
        mean, std = gp.predict(X[:5])
        assert np.allclose(mean, 3.0)
        assert np.all(np.isfinite(std))
        assert gp.fit_info["degenerate_targets"] is True

    def test_uncertainty_is_lower_near_training_data(self):
        """The core property the Bayesian acquisitions rely on."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 2)) * 0.2  # tight cluster
        y = X[:, 0] + 0.01 * rng.standard_normal(40)
        gp = GPSurrogate().fit(X, y)
        _, std_near = gp.predict(np.zeros((1, 2)))
        _, std_far = gp.predict(np.full((1, 2), 8.0))
        assert std_far[0] > std_near[0]

    def test_subsamples_when_over_max_fit_points(self):
        cfg = SurrogateConfig(max_fit_points=50)
        X, y = _toy(n=200)
        gp = GPSurrogate(cfg).fit(X, y)
        assert gp.n_train == 50


class TestRFSpecifics:
    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            RFSurrogate().predict(np.zeros((2, 3)))

    def test_uncertainty_is_strictly_positive(self):
        X, y = _toy()
        _, std = RFSurrogate().fit(X, y).predict(X)
        assert np.all(std > 0)


class TestFallback:
    def test_reports_gp_when_gp_succeeds(self):
        X, y = _toy()
        sur = build_surrogate(SurrogateConfig(kind="gp")).fit(X, y)
        assert sur.kind == "gp"
        assert sur.fallback_reason is None
        assert sur.fit_info["kind"] == "gp"

    def test_falls_back_to_rf_when_gp_raises(self, monkeypatch):
        X, y = _toy()

        def boom(self, X, y):
            raise RuntimeError("simulated GP failure")

        monkeypatch.setattr(GPSurrogate, "fit", boom)
        sur = build_surrogate(SurrogateConfig(kind="gp", allow_fallback=True))
        sur.fit(X, y)
        assert sur.kind == "rf"
        assert "simulated GP failure" in sur.fallback_reason
        assert "fallback_reason" in sur.fit_info
        mean, std = sur.predict(X[:4])
        assert mean.shape == (4,)

    def test_reraises_when_fallback_disabled(self, monkeypatch):
        X, y = _toy()

        def boom(self, X, y):
            raise RuntimeError("simulated GP failure")

        monkeypatch.setattr(GPSurrogate, "fit", boom)
        sur = build_surrogate(SurrogateConfig(kind="gp", allow_fallback=False))
        with pytest.raises(RuntimeError, match="simulated"):
            sur.fit(X, y)

    def test_rejects_non_finite_predictions(self, monkeypatch):
        X, y = _toy()
        sur = build_surrogate(SurrogateConfig(kind="rf")).fit(X, y)

        def bad(self, X):
            return np.full(X.shape[0], np.nan), np.ones(X.shape[0])

        monkeypatch.setattr(RFSurrogate, "predict", bad)
        with pytest.raises(RuntimeError, match="non-finite"):
            sur.predict(X)

    def test_rf_kind_is_respected_without_fallback_path(self):
        X, y = _toy()
        sur = build_surrogate(SurrogateConfig(kind="rf")).fit(X, y)
        assert sur.kind == "rf"
