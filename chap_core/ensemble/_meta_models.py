"""Meta-models and CRPS utilities for stacking ensembles."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize, nnls

logger = logging.getLogger(__name__)


def _crps_score(obs: np.ndarray, forecast: np.ndarray) -> float:
    term1 = np.mean(np.abs(forecast - obs.reshape(-1, 1)), axis=1)
    m = forecast.shape[1]
    if m <= 1:
        return float(np.mean(term1))
    term2 = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            term2 += np.mean(np.abs(forecast[:, i] - forecast[:, j]))
    term2 /= m * (m - 1) / 2.0
    return float(np.mean(term1) - 0.5 * term2)


def crps_ensemble(observations: np.ndarray, forecasts: np.ndarray) -> float:
    """Legacy alias."""
    return _crps_score(observations, forecasts)


class NonNegativeMetaModel:
    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> NonNegativeMetaModel:
        coef, _ = nnls(X, y)  # type: ignore[misc]
        s = coef.sum()
        self.coef_ = coef / s if s > 0 else coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Meta-model not fitted")
        return np.dot(X, self.coef_)


class ProbabilisticMetaModel:
    def __init__(self, verbose: bool = False) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.verbose = verbose

    def fit(self, X_samples: list[np.ndarray], y: np.ndarray) -> ProbabilisticMetaModel:
        target_shape = X_samples[0].shape
        for i, s in enumerate(X_samples):
            if s.shape != target_shape:
                raise ValueError(f"Sample shape mismatch: X_samples[0]={target_shape}, X_samples[{i}]={s.shape}")

        def obj(w: np.ndarray) -> float:
            w_norm = w / (w.sum() + 1e-10)
            ens = sum(w_norm[i] * X_samples[i] for i in range(len(X_samples)))
            return _crps_score(y, ens)

        n = len(X_samples)
        w0 = np.ones(n) / n
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            constraints={"type": "ineq", "fun": lambda w: w},
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if self.verbose:
            logger.info("Probabilistic meta-model fit: CRPS=%.4f, success=%s", res.fun, res.success)
        coef = res.x
        coef = coef / (coef.sum() + 1e-10)
        self.coef_ = coef
        return self

    def predict(self, X_samples: list[np.ndarray]) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Meta-model not fitted")
        ens = sum(self.coef_[i] * X_samples[i] for i in range(len(X_samples)))
        return np.maximum(ens, 0.0)

