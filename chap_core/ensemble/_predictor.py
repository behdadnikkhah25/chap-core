"""Prediction-time logic for the stacking ensemble."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from chap_core.datatypes import Samples
from chap_core.ensemble._sample_extractor import SampleExtractor as _SampleExtractor
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

if TYPE_CHECKING:
    from chap_core.ensemble._meta_models import ProbabilisticMetaModel

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    def __init__(
        self,
        predictors: list[Any],
        meta: ProbabilisticMetaModel,
        n_samples: int,
    ) -> None:
        self._predictors = list(predictors)
        self._meta = meta
        self._n_samples = n_samples

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet[Samples]:
        df_future = future_data.to_pandas()

        base_samp = []
        for i, p in enumerate(self._predictors):
            samples = _SampleExtractor.reshape_samples(
                p.predict(historic_data, future_data), df_future, self._n_samples
            )
            missing = int(np.sum(np.any(np.isnan(samples), axis=1)))
            if missing:
                raise ValueError(
                    f"Missing base model predictions for {missing} of {samples.shape[0]} rows "
                    f"from base model at index {i}"
                )
            base_samp.append(samples)
        meta_prob = cast("ProbabilisticMetaModel", self._meta)
        ens_samp = meta_prob.predict(base_samp)
        return self._pack_samples(ens_samp, df_future, future_data)

    @staticmethod
    def _pack_samples(all_samples: np.ndarray, df_future: pd.DataFrame, future_data: DataSet) -> DataSet[Samples]:
        result: dict[Any, Samples] = {}
        n_samples = all_samples.shape[1]
        sample_cols = [f"sample_{i}" for i in range(n_samples)]
        for loc in sorted(future_data.locations()):
            mask = (df_future["location"] == loc).to_numpy()
            loc_idx = np.where(mask)[0]
            tp = future_data[loc].time_period
            if len(loc_idx) != len(tp):
                raise ValueError(f"Row/time_period mismatch for {loc}")
            df_samples = pd.DataFrame({"time_period": tp.topandas()})
            df_samples[sample_cols] = all_samples[loc_idx, :]
            result[loc] = Samples.from_pandas(df_samples)
        return DataSet(result)


__all__ = ["EnsemblePredictor"]
