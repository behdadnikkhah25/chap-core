from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import nnls

from chap_core.datatypes import FullData, Samples
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from chap_core.models.model_template import ModelTemplate


class NonNegativeMetaModel:
    """
    Meta-modell for ensemble som bruker non-negative least squares (NNLS).
    
    Dette sikrer at:
    1. Alle vekter er non-negative (>= 0)
    2. Ingen negative prediksjoner for count-data
    3. Vekter reflekterer faktisk viktighet av hver base-modell
    """

    def __init__(self) -> None:
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> NonNegativeMetaModel:
        """
        Finner non-negative koeffisienter som minimerer ||Xw - y||^2
        hvor w >= 0 for alle elementer.
        
        X shape: (n_samples, n_features) - base modell prediksjoner
        y shape: (n_samples,) - target verdier
        """
        # NNLS løser: min ||Ax - b||^2 subject to x >= 0
        coef, _ = nnls(X, y)
        s = coef.sum()
        if s > 0:
            coef = coef / s  # Normaliserer til vekter som summerer til 1
        self.coef_ = coef
        # Intercept blir 0 siden vi jobber direkte med prediksjoner
        self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predikterer som vektet kombinasjon av base modeller.
        """
        if self.coef_ is None:
            raise ValueError("Modellen må fittas først")

        # Vektet kombinasjon: sum(coef_i * X_i)
        return np.dot(X, self.coef_)


@dataclass
class BaseModelSpec:
    """
    Hvilke CHAP-modeller som skal inngå som basemodeller i ensemblet.

    - template: en ModelTemplate (for eksempel laget fra URL)
    - config:   en ModelConfiguration eller None (bruk default)
                (kan også være dict, men da ignoreres den og default brukes)
    """
    template: ModelTemplate
    config: Any | None = None


class EnsembleEstimator(ConfiguredModel):
    """
    Ensemble-estimator som bruker holdout-stacking på tidsseriehale.
    """

    def __init__(
        self,
        base_model_templates: list[ModelTemplate] | None = None,
        base_model_specs: Sequence[BaseModelSpec] | None = None,
        target_column: str = "disease_cases",
        inner_val_periods: int = 12,
        meta_model: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        specs: list[BaseModelSpec] = []

        if base_model_specs is not None:
            specs.extend(list(base_model_specs))

        if base_model_templates is not None:
            specs.extend(
                BaseModelSpec(template=tmpl, config=None)
                for tmpl in base_model_templates
            )

        if not specs:
            raise ValueError("EnsembleEstimator krever minst én base-modell.")

        self._base_specs = specs
        self._target_column = target_column
        self._inner_val_periods = inner_val_periods
        self._meta_model = meta_model or NonNegativeMetaModel()
        self._weights_: np.ndarray | None = None
        self._feature_columns: list[str] | None = None

        # NYTT: lagre lesbare navn på basemodellene (bruk template.name eller repo-slug)
        base_names: list[str] = []
        for spec in self._base_specs:
            tmpl = spec.template
            name = getattr(tmpl, "name", None)
            if not name:
                # fallback: hent siste del av repo-URLen hvis mulig
                repo = getattr(tmpl, "repo", None)
                if isinstance(repo, str) and repo:
                    name = repo.rstrip("/").split("/")[-1]
                else:
                    name = str(tmpl)
            base_names.append(name)
        self._base_model_names = base_names

    @classmethod
    def from_config(cls, spec: Any) -> EnsembleEstimator:
        base_specs: list[BaseModelSpec] = []
        for bm in spec.config["base_models"]:
            cfg = bm.get("config", None)
            base_specs.append(
                BaseModelSpec(
                    template=bm["template"],
                    config=cfg,
                )
            )
        target_col = spec.config.get("target_column", "disease_cases")
        inner_val_periods = spec.config.get("inner_val_periods", 12)
        return cls(
            base_model_specs=base_specs,
            target_column=target_col,
            inner_val_periods=inner_val_periods,
        )

    def _split_inner_train_val(self, data: DataSet) -> tuple[DataSet, DataSet]:
        df = data.to_pandas()
        all_periods = (
            df["time_period"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
        )

        if len(all_periods) <= self._inner_val_periods:
            split_idx = len(all_periods) // 2
        else:
            split_idx = len(all_periods) - self._inner_val_periods

        train_periods = set(all_periods[:split_idx])
        val_periods = set(all_periods[split_idx:])

        df_train = df[df["time_period"].astype(str).isin(train_periods)].copy()
        df_val = df[df["time_period"].astype(str).isin(val_periods)].copy()

        inner_train = DataSet.from_pandas(df_train, FullData, fill_missing=True)
        val_data = DataSet.from_pandas(df_val, FullData, fill_missing=True)
        return inner_train, val_data

    def _compute_weights_from_meta(self, X_meta: np.ndarray, y_val: np.ndarray) -> None:
        """
        Henter koeffisienter fra meta-modellen, normaliserer til prosentvekter
        og lagrer dem sammen med basemodell-navnene.

        I tillegg:
        - Skriver lesbar tekst til terminal
        - Skriver en CSV-fil 'ensemble_meta_report.csv' med vektingen
        """
        if not hasattr(self._meta_model, "coef_"):
            self._weights_ = None
            return

        coef = np.asarray(self._meta_model.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[0]

        # NNLS sikrer non-negative koeffisienter, men sikrer med absoluttverdier
        # Hvis noen koeff er negative pga numerisk instabilitet, ta absolutt verdi
        #coef = np.abs(coef)

        # Normaliserer koeffisienter direkte til vekter
        # Normaliserer slik at de summerer til 100% (prosenter)

        #coef_sum = coef.sum() + 1e-12

        #weights_percent = (coef / coef_sum) * 100.0
        weights_percent = coef * 100.0  # Direkte prosentvekter uten ytterligere normalisering
        self._weights_ = weights_percent

        # Koble vekter til modellnavn
        try:
            name_weight_pairs = list(zip(self._base_model_names, weights_percent, strict=True))
        except AttributeError:
            # Fallback hvis _base_model_names ikke finnes
            name_weight_pairs = [(f"model_{i}", w) for i, w in enumerate(weights_percent)]

        # 1) Lesbar print til terminal
        print(f"Vekter lært fra metamodell (i prosent): {self._weights_}")
        print(
            "Meta-vekter per modell: "
            + "; ".join(f"{name}: {w:.2f}%" for name, w in name_weight_pairs)
        )

        # 2) Skriv en "rapport"-CSV i stil med ensemble_report.csv:
        try:
            import csv
            from pathlib import Path

            report_path = Path("ensemble_meta_report.csv")

            # Header: Model + ett felt per basemodell
            header = ["Model"] + [name for name, _ in name_weight_pairs]
            # Rad: "ensemble_meta" + vektverdiene i samme rekkefølge
            row = ["ensemble_meta"] + [f"{float(w):.6f}" for _, w in name_weight_pairs]

            with report_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(row)

            print(f"Lagret ensemble-meta-rapport til {report_path.resolve()}")
        except Exception as e:
            print(f"Advarsel: klarte ikke å skrive ensemble_meta_report.csv: {e}")

    def train(self, train_data: DataSet, extra_args: Any = None) -> EnsemblePredictor:
        """
        Trener ensemblet på ett CHAP-train-vindu (per backtest-split).
        """
        # 1) Split train_data i inner_train og val
        inner_train, val_data = self._split_inner_train_val(train_data)

        # 2) Bygg basemodeller og tren dem på inner_train
        base_estimators: list[ConfiguredModel] = []
        for spec in self._base_specs:
            if spec.config is None or isinstance(spec.config, dict):
                estimator_cls = spec.template.get_model(None)  # type: ignore[arg-type]
            else:
                estimator_cls = spec.template.get_model(spec.config)  # type: ignore[arg-type]

            estimator = estimator_cls()  # type: ignore[call-arg]
            base_estimators.append(estimator)

        inner_predictors = [est.train(inner_train) for est in base_estimators]

        # 3) Bygg nivå-2 treningsdata (X_meta, y_val)
        df_val = val_data.to_pandas()
        y_val = df_val[self._target_column].to_numpy()
        key_cols = ["location", "time_period"]

        meta_features = []
        for predictor in inner_predictors:
            preds_ds = predictor.predict(inner_train, val_data)
            df_pred = self._samples_to_flat_dataframe(preds_ds)

            merged = df_val[key_cols].merge(
                df_pred[[*key_cols, "forecast"]],
                on=key_cols,
                how="left",
            )
            meta_features.append(merged["forecast"].to_numpy())

        X_meta = np.column_stack(meta_features)

        # 3b) Fjern rader der målvariabelen mangler (NaN) i val-delen
        mask = ~np.isnan(y_val)
        if not np.any(mask):
            raise ValueError(
                "Ingen gyldige (ikke-NaN) målverdier i valideringsdelen for stacking."
            )

        y_val_clean = y_val[mask]
        X_meta_clean = X_meta[mask, :]

        # 3c) Sjekk for NaN i meta-features fra basemodellene
        nan_in_X = np.any(np.isnan(X_meta_clean), axis=1)
        if np.any(nan_in_X):
            import logging
            logger = logging.getLogger(__name__)
            n_nan_rows = np.sum(nan_in_X)
            pct_nan = 100 * n_nan_rows / len(nan_in_X)
            logger.warning(
                f"Fjerner {n_nan_rows} rader ({pct_nan:.1f}%) med NaN i meta-features "
                f"fra basemodellprediksjoner"
            )
            valid_mask = ~nan_in_X
            X_meta_clean = X_meta_clean[valid_mask]
            y_val_clean = y_val_clean[valid_mask]

        if len(y_val_clean) == 0:
            raise ValueError(
                "Ingen gyldige rader igjen etter fjerning av NaN/manglende verdier. "
                "Sjekk at basemodellene produserer gyldige prediksjoner."
            )

        # 4) Tren meta-modellen og hent vekter
        self._meta_model.fit(X_meta_clean, y_val_clean)
        self._compute_weights_from_meta(X_meta_clean, y_val_clean)

        # 5) Tren basemodeller på full train_data for slutt-prediktor
        full_predictors = []
        for estimator in base_estimators:
            full_predictor = estimator.train(train_data)
            full_predictors.append(full_predictor)

        return EnsemblePredictor(
            target_column=self._target_column,
            base_predictors=full_predictors,
            meta_model=self._meta_model,
        )

    @staticmethod
    def _samples_to_flat_dataframe(preds_ds: Samples) -> pd.DataFrame:
        """
        Konverter Samples til ett forecast-tall per (location, time_period).

        Støtter flere formater fra ulike modeller:

        - Kolonne 'forecast' (standard CHAP).
        - Kolonne 'value' (noen modeller).
        - Flere kolonner 'sample_0', 'sample_1', ... (f.eks. chap_auto_ewars).
        - 'horizon_distance' kan mangle; da antar vi én horisont (0).
        """

        df = preds_ds.to_pandas()

        # 1) Finn/konstruer prediksjonskolonne
        pred_col: str

        if "forecast" in df.columns:
            pred_col = "forecast"
        elif "value" in df.columns:
            pred_col = "value"
        else:
            # Sjekk om vi har sample_*-kolonner (wide-format samples)
            sample_cols = [c for c in df.columns if c.startswith("sample_")]
            if sample_cols:
                # Ta gjennomsnitt over samples per rad
                df["forecast"] = df[sample_cols].mean(axis=1)
                pred_col = "forecast"
            else:
                raise ValueError(
                    "_samples_to_flat_dataframe: fant verken 'forecast', 'value' "
                    f"eller 'sample_*' i Samples.to_pandas() kolonner: {list(df.columns)}"
                )

        # 2) Håndter horizon_distance robust
        if "horizon_distance" in df.columns:
            df0 = df[df["horizon_distance"] == 0].copy()
        else:
            df0 = df.copy()

        # 3) Sørg for location og time_period
        missing = [c for c in ["location", "time_period"] if c not in df0.columns]
        if missing:
            raise ValueError(
                f"_samples_to_flat_dataframe: mangler kolonner {missing} i Samples.to_pandas(). "
                f"Kolonner: {list(df0.columns)}"
            )

        # 4) Returner standardisert DataFrame
        out = df0[["location", "time_period", pred_col]].copy()
        out = out.rename(columns={pred_col: "forecast"})
        return out

    @property
    def weights(self) -> np.ndarray | None:
        return self._weights_

    def predict(self, data: DataSet) -> Samples:
        raise NotImplementedError(
            "EnsembleEstimator brukes kun via train() som returnerer EnsemblePredictor."
        )


class EnsemblePredictor:
    """
    Bruker trenede basemodeller + meta-modell til å produsere ensemble-prediksjoner.
    """

    def __init__(
        self,
        target_column: str,
        base_predictors: Sequence[Any],
        meta_model: Any,
    ) -> None:
        self._target_column = target_column
        self._base_predictors = list(base_predictors)
        self._meta_model = meta_model

    def predict(
            self,
            historic_data: DataSet,
            future_data: DataSet,
    ) -> DataSet[Samples]:
        """
        CHAP-konsistent predict: returnerer DataSet[Samples],
        én Samples-tidsserie per location.
        """
        df_future = future_data.to_pandas()
        key_cols = ["location", "time_period"]

        # 1) Samle basemodellprediksjoner for alle rader i future_data
        meta_features = []
        for predictor in self._base_predictors:
            preds_ds = predictor.predict(historic_data, future_data)
            df_pred = EnsembleEstimator._samples_to_flat_dataframe(preds_ds)

            merged = df_future[key_cols].merge(
                df_pred[[*key_cols, "forecast"]],
                on=key_cols,
                how="left",
            )
            meta_features.append(merged["forecast"].to_numpy())

        X_meta_future = np.column_stack(meta_features)

        # 2) Meta-ensemble-forecast for hver rad
        y_pred = self._meta_model.predict(X_meta_future)

        # 3) Bygg Samples per location
        df_out = df_future.copy()
        df_out["forecast"] = y_pred

        result: dict[Any, Samples] = {}

        # sørg for stabil rekkefølge: sortér på location, time_period
        df_out = df_out.sort_values(["location", "time_period"])

        for loc in sorted(df_out["location"].unique()):
            mask = df_out["location"] == loc
            df_loc = df_out[mask].copy()

            # hent CHAP-time_period direkte fra future_data[loc]
            future_sample = future_data[loc]  # FullData for denne lokasjonen
            tp = future_sample.time_period

            preds_loc = df_loc["forecast"].to_numpy()
            if len(preds_loc) != len(tp):
                raise ValueError(
                    f"EnsemblePredictor: lengde på prediksjoner ({len(preds_loc)}) "
                    f"stemmer ikke med lengde på time_period ({len(tp)}) for location {loc!r}"
                )

            samples_arr = np.asarray(preds_loc, dtype=float).reshape(-1, 1)
            samples = Samples(samples=samples_arr, time_period=tp)
            result[loc] = samples

        return DataSet(result)
