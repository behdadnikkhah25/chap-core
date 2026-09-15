"""Minimal, robust stacking ensemble for CHAP."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from chap_core.ensemble._meta_models import ProbabilisticMetaModel
from chap_core.ensemble._predictor import EnsemblePredictor
from chap_core.ensemble._sample_extractor import SampleExtractor as _SampleExtractor
from chap_core.ensemble.wrappers import BaseModelSpec, TemplateWithConfig
from chap_core.models.configured_model import ConfiguredModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


class EnsembleModel(ConfiguredModel):
    @property
    def model_information(self) -> ModelTemplateInformation | None:
        return None

    def __init__(
        self,
        base_templates: Sequence[Any] | None = None,
        inner_val_periods: int = 12,
        horizon: int = 3,
        target_col: str = "disease_cases",
        n_samples: int = 100,
        meta_model: ProbabilisticMetaModel | None = None,
    ) -> None:
        super().__init__()
        self.base_templates = list(base_templates or [])
        if not self.base_templates:
            raise ValueError("Need at least one base model")
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        self.inner_val_periods = inner_val_periods
        self.horizon = horizon
        self.target_col = target_col
        self.n_samples = n_samples
        self.meta_model: ProbabilisticMetaModel | None = meta_model
        self.weights: np.ndarray | None = None
        # One entry per train() call: with n_retrain > 1 the outer backtest fits the
        # meta-model more than once, and a single reported vector would silently
        # describe only the last round.
        self.fit_history: list[tuple[np.ndarray, np.ndarray]] = []

    def _base_names(self) -> list[str]:
        names: list[str] = []
        for tmpl in self.base_templates:
            name = getattr(tmpl, "name", None)
            if not name:
                repo = getattr(tmpl, "repo", None)
                if isinstance(repo, str) and repo:
                    name = repo.rstrip("/").split("/")[-1]
                else:
                    name = str(tmpl)
            names.append(name)
        return names

    def inner_validation_windows(self, train_data: DataSet) -> list[tuple[DataSet, DataSet]]:
        """Split the tail of ``train_data`` into (historic, future) windows of ``horizon`` periods.

        Base models are asked for exactly ``horizon`` steps in each window, matching the
        horizon they are used at during the outer backtest. Fitting the meta-weights on a
        single long window instead would rank the base models at the wrong horizon, since
        relative forecast skill is strongly horizon-dependent.
        """
        periods = list(train_data.period_range)
        if len(periods) < 2:
            raise ValueError("Need at least two time periods for training")
        split_idx = (
            len(periods) // 2 if len(periods) <= self.inner_val_periods else len(periods) - self.inner_val_periods
        )
        if split_idx <= 0 or split_idx >= len(periods):
            raise ValueError("Invalid inner validation split")

        # Only whole windows are used. A shorter trailing window would score the base
        # models at horizons 1..k and feed those rows into the same weight fit as the
        # full-horizon rows, reintroducing the horizon mismatch this split exists to
        # avoid, so the remainder is moved back into the inner training data instead.
        n_windows = (len(periods) - split_idx) // self.horizon
        if n_windows == 0:
            raise ValueError(
                f"Inner validation needs at least {self.horizon} periods to form one window of horizon "
                f"{self.horizon}, but only {len(periods) - split_idx} are held out. Increase "
                "inner_val_periods or lower the horizon."
            )
        remainder = len(periods) - split_idx - n_windows * self.horizon
        if remainder:
            logger.info(
                "Inner validation: moving %d leading validation period(s) into training so every window has horizon %d",
                remainder,
                self.horizon,
            )
            split_idx += remainder

        windows: list[tuple[DataSet, DataSet]] = []
        for start in range(split_idx, len(periods), self.horizon):
            stop = start + self.horizon
            historic = train_data.restrict_time_period(slice(None, periods[start - 1]))
            future = train_data.restrict_time_period(slice(periods[start], periods[stop - 1]))
            windows.append((historic, future))

        logger.info(
            "Inner validation: %d periods, train=%d, val=%d, %d window(s) of horizon %d",
            len(periods),
            split_idx,
            len(periods) - split_idx,
            len(windows),
            self.horizon,
        )
        return windows

    def train(self, train_data: DataSet, extra_args: Any = None) -> EnsemblePredictor:
        windows = self.inner_validation_windows(train_data)
        inner_train = windows[0][0]

        ests: list[Any] = []
        for tmpl in self.base_templates:
            est_cls = cast("type[Any]", tmpl.get_model(None))
            ests.append(est_cls())
        preds_inner = [e.train(inner_train) for e in ests]

        df_val = pd.concat([w[1].to_pandas() for w in windows], ignore_index=True)
        y_val = df_val[self.target_col].to_numpy()

        # The target must never reach the base models: ExternalModel writes future_data
        # verbatim to the CSV it hands the model, so leaving disease_cases in place would
        # let a base model read the very values the meta-weights are fitted against.
        masked_windows = [(historic, future.remove_field(self.target_col)) for historic, future in windows]

        meta_list: list[np.ndarray] = []
        for p in preds_inner:
            per_window = [
                _SampleExtractor.reshape_samples(
                    p.predict(historic, future),
                    future.to_pandas(),
                    self.n_samples,
                )
                for historic, future in masked_windows
            ]
            meta_list.append(np.concatenate(per_window, axis=0))

        nan_in_features = np.zeros(len(y_val), dtype=bool)
        per_base_nan = []
        for arr in meta_list:
            nan_rows = np.any(np.isnan(arr), axis=1)
            nan_in_features |= nan_rows
            per_base_nan.append(int(np.sum(nan_rows)))

        dropped = int(np.sum(nan_in_features | np.isnan(y_val)))
        if dropped:
            logger.warning("Dropping %d validation rows due to NaNs in targets/features", dropped)
            names = self._base_names()
            for name, cnt in zip(names, per_base_nan, strict=False):
                if cnt:
                    logger.warning("NaN count for base model %s: %d", name, cnt)

        mask = ~np.isnan(y_val) & ~nan_in_features
        if not np.any(mask):
            raise ValueError("No valid targets in validation")
        y_clean = y_val[mask]
        # A meta-model per train() call. The outer backtest calls train() once per retrain,
        # and re-fitting a cached instance in place would also mutate the meta-model of an
        # EnsemblePredictor handed out by an earlier call.
        X_clean_samples = [m[mask, :] for m in meta_list]
        meta_model = self.meta_model if self.meta_model is not None else ProbabilisticMetaModel(verbose=True)
        meta_model.fit(X_clean_samples, y_clean)

        coef_raw = cast("np.ndarray", meta_model.coef_)
        coef = np.maximum(np.asarray(coef_raw, float), 0.0)
        total = float(np.sum(coef))
        if total <= 0:
            # The probabilistic meta-model falls back to uniform weights rather than
            # returning an all-zero solution, so this should be unreachable.
            raise ValueError("Meta-model produced non-positive weights")
        weights = coef / total * 100.0
        self.weights = weights
        self.fit_history.append((weights, coef))

        names = self._base_names()
        logger.info("Meta-weights (percent): %s", weights)
        for name, w, c in zip(names, weights, coef, strict=True):
            logger.info("  %s: %.2f%% (coefficient %.6f)", name, w, c)

        full_ests: list[Any] = []
        for tmpl in self.base_templates:
            est_cls = cast("type[Any]", tmpl.get_model(None))
            full_ests.append(est_cls())
        full_predictors = [e.train(train_data) for e in full_ests]

        return EnsemblePredictor(
            predictors=full_predictors,
            meta=meta_model,
            n_samples=self.n_samples,
        )

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        raise NotImplementedError("Use train() to obtain EnsemblePredictor")


class EnsembleEstimator(EnsembleModel):
    """Legacy class name/API backed by the same core implementation."""

    def __init__(
        self,
        base_model_templates: list[Any] | None = None,
        base_model_specs: Sequence[BaseModelSpec] | None = None,
        target_column: str = "disease_cases",
        inner_val_periods: int = 12,
        horizon: int = 3,
        meta_model: ProbabilisticMetaModel | None = None,
        n_samples: int = 100,
    ) -> None:
        specs = list(base_model_specs or [])
        if base_model_templates is not None:
            specs.extend(BaseModelSpec(template=t, config=None) for t in base_model_templates)
        if not specs:
            raise ValueError("EnsembleEstimator requires at least one base model.")

        self._base_specs = specs
        super().__init__(
            base_templates=[TemplateWithConfig(s.template, s.config) for s in specs],
            inner_val_periods=inner_val_periods,
            horizon=horizon,
            target_col=target_column,
            n_samples=n_samples,
            meta_model=meta_model,
        )

    @classmethod
    def from_config(cls, spec: Any) -> EnsembleEstimator:
        base_specs = [
            BaseModelSpec(template=bm["template"], config=bm.get("config")) for bm in spec.config["base_models"]
        ]
        return cls(
            base_model_specs=base_specs,
            target_column=spec.config.get("target_column", "disease_cases"),
            inner_val_periods=spec.config.get("inner_val_periods", 12),
        )

    def train(self, train_data: DataSet, extra_args: Any = None) -> EnsemblePredictor:
        pred = super().train(train_data, extra_args)
        return pred


__all__ = [
    "BaseModelSpec",
    "EnsembleEstimator",
    "EnsembleModel",
    "ProbabilisticMetaModel",
]
