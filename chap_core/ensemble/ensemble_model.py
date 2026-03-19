# chap_core/ensemble/ensemble_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from chap_core.datatypes import Samples, FullData
from chap_core.models.configured_model import ConfiguredModel
from chap_core.models.model_template import ModelTemplate
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chap_core.ensemble.classes.stackedEnsemble import StackedEnsemble


@dataclass
class BaseModelSpec:
    """
    Hvilke CHAP-modeller som skal inngå som basemodeller i ensemblet.

    - template: en ModelTemplate (for eksempel laget fra URL)
    - config:   en ModelConfiguration eller None (bruk default)
    """
    template: ModelTemplate
    config: Any | None = None


class EnsemblePredictor:
    """
    Predictor-delen av ensemblet. Denne returneres fra EnsembleEstimator.train()
    og brukes av backtest/evaluate_model.
    """

    def __init__(
        self,
        stacked: StackedEnsemble,
        base_predictors: list[ConfiguredModel],
        feature_columns: list[str],
        target_column: str = "disease_cases",
    ) -> None:
        self._stacked = stacked
        self._base_predictors = base_predictors
        self._feature_columns = feature_columns
        self._target_column = target_column

    def _dataset_to_X(self, data: DataSet) -> pd.DataFrame:
        """
        Konverter DataSet -> pandas.DataFrame med de samme feature-kolonnene
        som ble brukt ved trening.
        """
        df = data.to_pandas()
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"EnsemblePredictor: mangler kolonner i future-data: {missing}"
            )
        return df[self._feature_columns].copy()

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet[Samples]:
        """
        CHAP-konsistent predict:
        tar historic_data, future_data, og returnerer DataSet[Samples]
        for fremtidige perioder.

        Viktig: vi bruker time_period direkte fra future_data (CHAP PeriodRange),
        og bruker kun pandas til å koble prediksjoner til riktig lokasjon.
        """
        # 1) Bruk bare future_data til features (tabulært)
        X_future = self._dataset_to_X(future_data)

        # 2) Hent lokasjonsinfo fra future_data som DataFrame
        df_fut = future_data.to_pandas()
        locs = df_fut["location"].to_list()

        # 3) Kjør ensemblet: én prediksjonsverdi per rad
        y_pred = self._stacked.predict(X_future)
        assert len(y_pred) == len(df_fut)

        # 4) Bygg Samples per lokasjon – tidsaksen tas direkte fra future_data[loc]
        result_mapping: dict[Any, Samples] = {}
        for loc in sorted(set(locs)):
            mask = df_fut["location"] == loc
            preds_loc = y_pred[mask]

            # hent CHAP-perioden for denne lokasjonen fra future_data
            try:
                future_sample = future_data[loc]      # FullData for denne lokasjonen
                tp = future_sample.time_period       # PeriodRange (CHAP-type)
            except Exception as e:
                raise RuntimeError(
                    f"EnsemblePredictor: kunne ikke hente time_period for location {loc!r}"
                ) from e

            # sanity check: lengde på tidsakse = lengde på prediksjoner
            if len(preds_loc) != len(tp):
                raise ValueError(
                    f"EnsemblePredictor: lengde på prediksjoner ({len(preds_loc)}) "
                    f"stemmer ikke med lengde på time_period ({len(tp)}) for location {loc!r}"
                )

            # Viktig: første dimensjon = prediction_length, andre = n_samples
            samples_arr = np.asarray(preds_loc, dtype=float).reshape(-1, 1)

            samples = Samples(
                samples=samples_arr,
                time_period=tp,
            )
            result_mapping[loc] = samples

        return DataSet(result_mapping)


class EnsembleEstimator(ConfiguredModel):
    """
    En CHAP-kompatibel estimator som trener et stacked ensemble
    av flere CHAP-modeller, og returnerer en EnsemblePredictor.
    """

    def __init__(
        self,
        base_model_templates: list[ModelTemplate],
        meta_model: Any,
        n_folds: int = 5,
        use_time_series_split: bool = True,
        random_state: int = 42,
        feature_columns: list[str] | None = None,
        target_column: str = "disease_cases",
    ) -> None:
        self._base_specs: list[BaseModelSpec] = [
            BaseModelSpec(template=tpl, config=None) for tpl in base_model_templates
        ]
        self._meta_model = meta_model
        self._n_folds = n_folds
        self._use_tss = use_time_series_split
        self._random_state = random_state
        self._feature_columns = feature_columns
        self._target_column = target_column

        self._stacked: StackedEnsemble | None = None
        self._base_predictors: list[ConfiguredModel] | None = None

    @property
    def weights(self) -> np.ndarray | None:
        """
        Returnerer vektene (i prosent) lært av meta-modellen,
        én vekt per basemodell. None hvis modellen ikke er trent.
        """
        if self._stacked is None:
            return None
        return self._stacked.get_weights()

    def _dataset_to_Xy(self, data: DataSet) -> tuple[pd.DataFrame, pd.Series]:
        """
        Velg features og target fra DataSet.
        Første versjon: target = 'disease_cases', features = alle andre kolonner.
        """
        df = data.to_pandas()

        if self._target_column not in df.columns:
            raise ValueError(
                f"EnsembleEstimator: '{self._target_column}' finnes ikke i datasettet."
            )

        y = df[self._target_column].copy()

        if self._feature_columns is None:
            self._feature_columns = [c for c in df.columns if c != self._target_column]

        X = df[self._feature_columns].copy()
        return X, y

    def train(self, train_data: DataSet, extra_args=None) -> EnsemblePredictor:
        """
        Trener stacked-ensemblet.
        """
        X, y = self._dataset_to_Xy(train_data)

        # Fjern rader der target er NaN (meta-modellen tåler ikke NaN i y)
        if isinstance(y, pd.Series):
            mask = ~y.isna()
        else:
            mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        # 2) Lag basemodeller (ConfiguredModel)
        base_estimators: list[ConfiguredModel] = []
        for spec in self._base_specs:
            estimator_cls = spec.template.get_model(spec.config)
            estimator = estimator_cls()  # type: ignore[call-arg]
            base_estimators.append(estimator)

        # 3) Wrap basemodeller til noe StackedEnsemble kan bruke
        base_wrappers = [
            _ConfiguredModelWrapper(estimator, self._feature_columns, self._target_column)
            for estimator in base_estimators
        ]

        # 4) Tren StackedEnsemble på tabulær X, y
        stacked = StackedEnsemble(
            base_models=base_wrappers,
            meta_model=self._meta_model,
            n_folds=self._n_folds,
            use_time_series_split=self._use_tss,
            random_state=self._random_state,
        )
        stacked.train(X, y)

        # 5) Tren hver basemodell på hele train_data for senere bruk (CHAP-prediksjoner)
        base_predictors: list[ConfiguredModel] = []
        for estimator in base_estimators:
            predictor = estimator.train(train_data)
            base_predictors.append(predictor)

        self._stacked = stacked
        self._base_predictors = base_predictors

        return EnsemblePredictor(
            stacked=stacked,
            base_predictors=base_predictors,
            feature_columns=self._feature_columns,
            target_column=self._target_column,
        )

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet[Samples]:
        """
        Implementasjon kreves av ConfiguredModel, men all reell prediksjon
        gjøres av EnsemblePredictor. Denne metoden delegere dit.
        """
        if self._stacked is None or self._base_predictors is None:
            raise RuntimeError(
                "EnsembleEstimator: modellen er ikke trent. "
                "Kall .train() først for å få en EnsemblePredictor."
            )

        predictor = EnsemblePredictor(
            stacked=self._stacked,
            base_predictors=self._base_predictors,
            feature_columns=self._feature_columns,
            target_column=self._target_column,
        )
        return predictor.predict(historic_data, future_data)


class _ConfiguredModelWrapper:
    """
    Intern wrapper som gir et ConfiguredModel-etimator samme interface som
    base-modellene dine i StackedEnsemble: .fit(X_df, y) og .predict(X_df).

    VIKTIG: predict(X) må returnere én verdi per rad i X.
    """

    def __init__(
        self,
        estimator: ConfiguredModel,
        feature_columns: list[str],
        target_column: str = "disease_cases",
    ) -> None:
        self._estimator = estimator
        self._feature_columns = feature_columns
        self._target_column = target_column
        self._predictor: ConfiguredModel | None = None

    def _XY_to_dataset(
            self, X: pd.DataFrame, y: pd.Series | np.ndarray | None
    ) -> DataSet:
        """
        Bygg et DataSet fra X (+ ev. y).
        Vi antar at X har minst time_period, location og klimavariabler.
        """
        df = X.copy()
        import numpy as _np

        if y is not None:
            y_arr = _np.asarray(y)
            df[self._target_column] = y_arr
        else:
            if self._target_column not in df.columns:
                df[self._target_column] = _np.nan

        # Viktig: tillat hull i tidsserien når vi lager DataSet fra et fold-subsett
        return DataSet.from_pandas(df, FullData, fill_missing=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        dataset = self._XY_to_dataset(X, y)
        self._predictor = self._estimator.train(dataset)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returner én prediksjon per rad i X.

        Vi:
        - bygger et DataSet av X,
        - lar CHAP-modellen predikere per location,
        - tar én skalar per location (gjennomsnitt over samples og tider),
        - og mapper denne tilbake til hver rad i X etter location.
        """
        if self._predictor is None:
            raise RuntimeError("_ConfiguredModelWrapper: modellen er ikke trent.")

        # 1) bygg DataSet fra akkurat dette (fold-subsettet)
        dataset = self._XY_to_dataset(X, y=None)

        # 2) få CHAP-prediksjoner per location
        preds_ds = self._predictor.predict(dataset, dataset)

        # 3) én skalar per location fra CHAP-prediksjonene
        loc_to_value: dict[Any, float] = {}
        for loc, samples in preds_ds.items():
            arr = samples.samples  # (n_samples, prediction_length)
            loc_to_value[loc] = float(np.mean(arr))

        # 4) bygg verdier i samme lengde og rekkefølge som X
        df = X
        if "location" not in df.columns:
            raise ValueError(
                "_ConfiguredModelWrapper.predict: 'location' mangler i X"
            )

        vals: list[float] = []
        for loc in df["location"]:
            if loc not in loc_to_value:
                raise KeyError(
                    f"_ConfiguredModelWrapper.predict: location {loc!r} mangler i preds_ds"
                )
            vals.append(loc_to_value[loc])

        return np.array(vals, dtype=float)