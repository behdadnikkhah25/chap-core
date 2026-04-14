from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import nnls, minimize

from chap_core.datatypes import FullData, Samples
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from chap_core.models.model_template import ModelTemplate


def crps_ensemble(observations: np.ndarray, forecasts: np.ndarray) -> float:
    """
    Beregn CRPS (Continuous Ranked Probability Score) for ensemble-prediksjoner.

    Args:
        observations: Shape (n,) - faktiske verdier
        forecasts: Shape (n, m) - m samples per observasjon fra ensemble

    Returns:
        Gjennomsnittlig CRPS

    Teori: CRPS = integral av (CDF(x) - indicator(x >= obs))^2
    For samples: CRPS ≈ mean(|samples - obs|) - (1/(2*M)) * mean(|sample_i - sample_j|)
    """
    n_samples = forecasts.shape[1]

    # Term 1: Gjennomsnittlig avstand fra samples til observasjon
    term1 = np.mean(np.abs(forecasts - observations.reshape(-1, 1)), axis=1)

    # Term 2: Gjennomsnittlig parvise avstand mellom samples
    term2 = 0.0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            term2 += np.mean(np.abs(forecasts[:, i] - forecasts[:, j]))

    term2 = term2 / (n_samples * (n_samples - 1) / 2) if n_samples > 1 else 0.0

    crps = np.mean(term1) - 0.5 * term2
    return crps


class NonNegativeMetaModel:
    """
    Meta-modell for ensemble som bruker non-negative least squares (NNLS).

    Optimaliserer på MSE/RMSE (deterministisk stacking).
    """

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> NonNegativeMetaModel:
        """
        Finner non-negative koeffisienter som minimerer ||Xw - y||^2
        hvor w >= 0 for alle elementer.

        X shape: (n_samples, n_features) - base modell prediksjoner
        y shape: (n_samples,) - target verdier
        """
        coef, _ = nnls(X, y)
        s = coef.sum()
        if s > 0:
            coef = coef / s
        self.coef_ = coef
        self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predikterer som vektet kombinasjon av base modeller.
        """
        if self.coef_ is None:
            raise ValueError("Modellen må fittas først")
        return np.dot(X, self.coef_)


class ProbabilisticMetaModel:
    """
    Meta-modell for PROBABILISTISK stacking (Yao et al. 2018).

    Optimaliserer vekter ved å minimere CRPS på valideringssamplesene.
    Dette sikrer at den kombinerte fordelingen er best mulig kalibrert,
    ikke bare at midtpunktet er best.

    Teorien:
    - Vanlig stacking: min ||X*w - y||^2 (optimaliserer RMSE)
    - Probabilistisk: min CRPS(ensemble_samples(w), y) (optimaliserer fordeling)
    """

    def __init__(self, verbose: bool = False) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.verbose = verbose
        self._base_samples: list[np.ndarray] = []  # Lagre samples fra valideringen
        self._y_val: np.ndarray | None = None

    def fit(self, X_samples: list[np.ndarray], y: np.ndarray) -> ProbabilisticMetaModel:
        """
        Finner vekter som minimerer CRPS på valideringssamplesene.

        X_samples: Liste med n_features arrays, hver shape (n_samples, n_ensemble_samples)
                   - X_samples[i] = samples fra basemodell i
        y: Shape (n_samples,) - observerte verdier
        """
        self._base_samples = X_samples
        self._y_val = y

        n_features = len(X_samples)

        # Objektfunksjon: CRPS som funksjon av vekter w
        def crps_objective(w):
            """Beregn CRPS for gitt vekter."""
            # Normalisér vekter til å summere til 1
            w_norm = w / (w.sum() + 1e-10)

            # Kombiner samples: ensemble_samples = sum(w_i * X_samples_i)
            ensemble_samples = np.zeros_like(X_samples[0])
            for i, samples in enumerate(X_samples):
                ensemble_samples += w_norm[i] * samples

            # Beregn CRPS for denne kombinasjonen
            score = crps_ensemble(y, ensemble_samples)
            return score

        # Constraints: w >= 0
        constraints = {'type': 'ineq', 'fun': lambda w: w}  # w_i >= 0

        # Initial guess: uniform weights
        w0 = np.ones(n_features) / n_features

        # Optimiser
        result = minimize(
            crps_objective,
            w0,
            method='SLSQP',
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if self.verbose:
            print(f"Probabilistic meta-model fit: CRPS = {result.fun:.4f}, success = {result.success}")

        # Normalisér og lagre vekter
        coef = result.x
        coef = coef / (coef.sum() + 1e-10)
        self.coef_ = coef

        return self

    def predict(self, X_samples: list[np.ndarray]) -> np.ndarray:
        """
        Kombiner samples fra basemodeller med lærte vekter.

        Bruker LINEÆR KOMBINASJON:
        ensemble_samples = w1*S1 + w2*S2 + w3*S3

        Dette fungerer bedre enn Quantile-Stacking fordi:
        1. Bevarer korrelasjonsstrukturen mellom basemodeller
        2. Gir diverse samples (ikke repetitive blokker)
        3. Vekter optimaliseres direkte på CRPS

        X_samples: Liste med arrays shape (n_predictions, n_ensemble_samples)
        Returns: Kombinerte samples shape (n_predictions, n_ensemble_samples)
        """
        if self.coef_ is None:
            raise ValueError("Modellen må fittas først")

        # Lineær kombinasjon av alle modellers samples
        ensemble_samples = np.zeros_like(X_samples[0], dtype=float)
        for i, samples in enumerate(X_samples):
            ensemble_samples += self.coef_[i] * samples

        # Sikre ikke-negative verdier (for count-data)
        ensemble_samples = np.maximum(ensemble_samples, 0.0)

        return ensemble_samples


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

    Støtter både deterministisk og probabilistisk output:
    - Deterministisk: punkt-prediksjoner kombinert med vekter
    - Probabilistisk: samples generert rundt punkt-prediksjoner
      basert på observerte residualer fra validering
    """

    def __init__(
            self,
            base_model_templates: list[ModelTemplate] | None = None,
            base_model_specs: Sequence[BaseModelSpec] | None = None,
            target_column: str = "disease_cases",
            inner_val_periods: int = 12,
            meta_model: Any | None = None,
            probabilistic: bool = False,
            probabilistic_meta_model: bool = False,
            n_samples: int = 100,
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

        # Velg meta-modell: probabilistisk eller deterministisk
        if meta_model is None:
            if probabilistic_meta_model:
                meta_model = ProbabilisticMetaModel(verbose=True)
            else:
                meta_model = NonNegativeMetaModel()

        self._meta_model = meta_model
        self._weights_: np.ndarray | None = None
        self._feature_columns: list[str] | None = None
        self._probabilistic = probabilistic
        self._probabilistic_meta_model = probabilistic_meta_model
        self._n_samples = n_samples

        # For probabilistisk ensemble: lagre residualer fra basemodeller
        self._base_residuals: list[np.ndarray] = []

        # For probabilistisk meta-modell: lagre samples
        self._base_samples_val: list[np.ndarray] = []

        # Lagre lesbare navn på basemodellene
        base_names: list[str] = []
        for spec in self._base_specs:
            tmpl = spec.template
            name = getattr(tmpl, "name", None)
            if not name:
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
        # coef = np.abs(coef)

        # Normaliserer koeffisienter direkte til vekter
        # Normaliserer slik at de summerer til 100% (prosenter)

        # coef_sum = coef.sum() + 1e-12

        # weights_percent = (coef / coef_sum) * 100.0
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

        # 3) Bygg nivå-2 treningsdata
        df_val = val_data.to_pandas()
        y_val = df_val[self._target_column].to_numpy()
        key_cols = ["location", "time_period"]

        meta_features = []
        base_residuals_list: list[np.ndarray] = []
        base_samples_list: list[np.ndarray] = []

        for predictor in inner_predictors:
            preds_ds = predictor.predict(inner_train, val_data)

            # For probabilistisk meta-modell: lagre fulle samples
            if self._probabilistic_meta_model:
                df_pred = preds_ds.to_pandas()
                sample_cols = [c for c in df_pred.columns if c.startswith("sample_")]

                if sample_cols:
                    # Hent samples direkte
                    samples_mat = df_pred[sample_cols].to_numpy(dtype=float)
                else:
                    # Fallback: bruk punkt-prediksjoner kopiert til flere "samples"
                    df_pred_flat = self._samples_to_flat_dataframe(preds_ds)
                    merged = df_val[key_cols].merge(
                        df_pred_flat[[*key_cols, "forecast"]],
                        on=key_cols,
                        how="left",
                    )
                    point_preds = merged["forecast"].to_numpy()
                    # Lag "degenerert" fordeling: samme verdi repeated
                    samples_mat = np.tile(point_preds.reshape(-1, 1), (1, self._n_samples))

                # SIKRING: Sikre at alle samples-arrays har samme antall samples
                # Hvis modellen returnerer færre samples enn self._n_samples, resample
                n_samples_returned = samples_mat.shape[1]
                if n_samples_returned != self._n_samples:
                    if n_samples_returned == 1:
                        # Replikere punkt-prediksjoner til n_samples
                        samples_mat = np.tile(samples_mat, (1, self._n_samples))
                    else:
                        # Resample via bootstrap
                        indices = np.random.choice(n_samples_returned, size=self._n_samples, replace=True)
                        samples_mat = samples_mat[:, indices]

                base_samples_list.append(samples_mat)

            # For begge modeller: beregn gjennomsnitt for punkt-prediksjoner
            df_pred = self._samples_to_flat_dataframe(preds_ds)

            merged = df_val[key_cols].merge(
                df_pred[[*key_cols, "forecast"]],
                on=key_cols,
                how="left",
            )
            base_preds = merged["forecast"].to_numpy()
            meta_features.append(base_preds)

            # Beregn residualer for denne modellen
            residuals = y_val - base_preds
            base_residuals_list.append(residuals)

        X_meta = np.column_stack(meta_features)

        # 3b) Fjern rader der målvariabelen mangler
        mask = ~np.isnan(y_val)
        if not np.any(mask):
            raise ValueError(
                "Ingen gyldige (ikke-NaN) målverdier i valideringsdelen for stacking."
            )

        y_val_clean = y_val[mask]
        X_meta_clean = X_meta[mask, :]

        # Filtrer residualer samme måte
        base_residuals_clean = [res[mask] for res in base_residuals_list]

        # Filtrer samples samme måte
        base_samples_clean = [samp[mask, :] for samp in base_samples_list] if base_samples_list else []

        # 3c) Sjekk for NaN i meta-features
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
            base_residuals_clean = [res[valid_mask] for res in base_residuals_clean]
            base_samples_clean = [samp[valid_mask, :] for samp in base_samples_clean]

        if len(y_val_clean) == 0:
            raise ValueError(
                "Ingen gyldige rader igjen etter fjerning av NaN/manglende verdier. "
                "Sjekk at basemodellene produserer gyldige prediksjoner."
            )

        # 4) Tren meta-modellen
        if self._probabilistic_meta_model:
            # Probabilistisk: optimer CRPS på samples
            self._meta_model.fit(base_samples_clean, y_val_clean)
        else:
            # Deterministisk: optimer MSE på punkt-prediksjoner
            self._meta_model.fit(X_meta_clean, y_val_clean)

        self._compute_weights_from_meta(X_meta_clean, y_val_clean)

        # 5) Lagre residualer alltid
        self._base_residuals = base_residuals_clean
        self._base_samples_val = base_samples_clean

        # 6) Tren basemodeller på full train_data
        full_predictors = []
        for estimator in base_estimators:
            full_predictor = estimator.train(train_data)
            full_predictors.append(full_predictor)

        return EnsemblePredictor(
            target_column=self._target_column,
            base_predictors=full_predictors,
            meta_model=self._meta_model,
            probabilistic=self._probabilistic,
            probabilistic_meta_model=self._probabilistic_meta_model,
            n_samples=self._n_samples,
            base_residuals=self._base_residuals,
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

    Støtter to meta-modeller:
    - Deterministisk (NNLS): punkt-prediksjoner kombinert med vekter som minimerer MSE
    - Probabilistisk (CRPS): samples kombinert med vekter som minimerer CRPS

    Og to output-modi:
    - Deterministisk: punkt-prediksjoner (1 sample per periode)
    - Probabilistisk: samples rundt punkt-prediksjoner basert på residualer
    """

    def __init__(
            self,
            target_column: str,
            base_predictors: Sequence[Any],
            meta_model: Any,
            probabilistic: bool = False,
            probabilistic_meta_model: bool = False,
            n_samples: int = 100,
            base_residuals: list[np.ndarray] | None = None,
    ) -> None:
        self._target_column = target_column
        self._base_predictors = list(base_predictors)
        self._meta_model = meta_model
        self._probabilistic = probabilistic
        self._probabilistic_meta_model = probabilistic_meta_model
        self._n_samples = n_samples
        self._base_residuals = base_residuals or []

    def predict(
            self,
            historic_data: DataSet,
            future_data: DataSet,
    ) -> DataSet[Samples]:
        """
        CHAP-konsistent predict: returnerer DataSet[Samples].

        To modi:
        1. Probabilistisk Meta-Modell (ProbabilisticMetaModel):
           - Kombinerer samples direkte med Quantile-Stacking
           - RETURNER BARE de kombinerte samples (IKKE residualer!)
           - Gir optimal CRPS-kalibrering

        2. Deterministisk Meta-Modell (NonNegativeMetaModel):
           - Kombinerer punkt-prediksjoner
           - Generer samples rundt dem via bootstrap-residualer
           - Gir diverse samples basert på observert usikkerhet
        """
        df_future = future_data.to_pandas()
        key_cols = ["location", "time_period"]

        # MODUS 1: ProbabilisticMetaModel
        if self._probabilistic_meta_model:
            base_samples_future: list[np.ndarray] = []

            for predictor in self._base_predictors:
                preds_ds = predictor.predict(historic_data, future_data)
                df_pred = preds_ds.to_pandas()

                sample_cols = [c for c in df_pred.columns if c.startswith("sample_")]

                if sample_cols:
                    samples_mat = df_pred[sample_cols].to_numpy(dtype=float)
                else:
                    # Fallback: punkt-prediksjon replikert til n_samples
                    df_pred_flat = EnsembleEstimator._samples_to_flat_dataframe(preds_ds)
                    merged = df_future[key_cols].merge(
                        df_pred_flat[[*key_cols, "forecast"]],
                        on=key_cols,
                        how="left",
                    )
                    point_preds = merged["forecast"].to_numpy()
                    samples_mat = np.tile(point_preds.reshape(-1, 1), (1, self._n_samples))

                # Sikre konsistent antall samples
                n_samples_returned = samples_mat.shape[1]
                if n_samples_returned != self._n_samples:
                    if n_samples_returned == 1:
                        samples_mat = np.tile(samples_mat, (1, self._n_samples))
                    else:
                        indices = np.random.choice(
                            n_samples_returned, size=self._n_samples, replace=True
                        )
                        samples_mat = samples_mat[:, indices]

                base_samples_future.append(samples_mat)

            # ✅ QUANTILE-STACKING kombinerer samples med bevart fordeling
            ensemble_samples = self._meta_model.predict(base_samples_future)

            # ✅ RETURNÉR BARE kombinerte samples - IKKE residualer!
            # (Residualer ville ødela kalibreringen som meta-modellen optimaliserte)

            # Pakk tilbake til DataSet[Samples]
            result: dict[Any, Samples] = {}

            for loc in sorted(future_data.locations()):
                mask = df_future["location"] == loc
                loc_rows_idx = np.where(np.asarray(mask))[0]

                future_sample = future_data[loc]
                tp = future_sample.time_period

                if len(loc_rows_idx) != len(tp):
                    raise ValueError(
                        f"Antall rader for {loc} matcher ikke time_period"
                    )

                loc_samples = ensemble_samples[loc_rows_idx, :]
                samples_obj = Samples(samples=loc_samples, time_period=tp)
                result[loc] = samples_obj

            return DataSet(result)

        # MODUS 2: Deterministisk Meta-Modell + Bootstrap-Residualer
        else:
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

            # Hent vekter fra meta-modellen
            if not hasattr(self._meta_model, "coef_") or self._meta_model.coef_ is None:
                raise ValueError("Meta-modellen må være trent.")

            weights = np.asarray(self._meta_model.coef_, dtype=float)
            if weights.ndim == 2:
                weights = weights[0]

            weights = np.abs(weights)
            w_sum = weights.sum()
            if w_sum <= 0:
                raise ValueError("Meta-vekter har sum <= 0.")
            weights = weights / w_sum

            # Velg: hvis probabilistic eller har residualer, generer samples
            has_residuals = self._base_residuals and len(self._base_residuals) > 0

            if self._probabilistic or has_residuals:
                return self._predict_probabilistic(
                    X_meta_future, weights, df_future, future_data
                )
            else:
                # Deterministisk: bare punkt-prediksjoner
                y_pred = self._meta_model.predict(X_meta_future)
                df_out = df_future.copy()
                df_out["forecast"] = y_pred

                result: dict[Any, Samples] = {}
                df_out = df_out.sort_values(["location", "time_period"])

                for loc in sorted(df_out["location"].unique()):
                    mask = df_out["location"] == loc
                    df_loc = df_out[mask].copy()

                    future_sample = future_data[loc]
                    tp = future_sample.time_period

                    preds_loc = df_loc["forecast"].to_numpy()
                    if len(preds_loc) != len(tp):
                        raise ValueError(
                            f"Lengde på prediksjoner stemmer ikke for location {loc!r}"
                        )

                    samples_arr = np.asarray(preds_loc, dtype=float).reshape(-1, 1)
                    samples = Samples(samples=samples_arr, time_period=tp)
                    result[loc] = samples

                return DataSet(result)

    def _predict_probabilistic(
            self,
            X_meta_future: np.ndarray,
            weights: np.ndarray,
            df_future: Any,
            future_data: DataSet,
    ) -> DataSet[Samples]:
        """
        Generer probabilistiske samples rundt punkt-prediksjoner.

        For hver basemodell-prediksjon, tegn samples fra residual-fordelingen
        observert under trening, vekt dem, og kombinér.

        VIKTIG: Denne metoden håndterer både deterministiske og probabilistiske basemodeller.
        """
        n_rows = X_meta_future.shape[0]
        S = self._n_samples

        # Få punkt-prediksjoner fra meta-modellen (for sentrum av samples)
        y_pred_point = self._meta_model.predict(X_meta_future)

        # Hvis ingen residualer, returner deterministisk
        if not self._base_residuals or len(self._base_residuals) == 0:
            df_out = df_future.copy()
            df_out["forecast"] = y_pred_point
            result: dict[Any, Samples] = {}
            df_out = df_out.sort_values(["location", "time_period"])

            for loc in sorted(df_out["location"].unique()):
                mask = df_out["location"] == loc
                df_loc = df_out[mask].copy()
                future_sample = future_data[loc]
                tp = future_sample.time_period
                preds_loc = df_loc["forecast"].to_numpy()
                samples_arr = np.asarray(preds_loc, dtype=float).reshape(-1, 1)
                samples = Samples(samples=samples_arr, time_period=tp)
                result[loc] = samples

            return DataSet(result)

        # Generer samples for hver basemodell basert på residualer
        # Sikre minimum varians: hvis residual-std er for liten, bruk punkter
        ensemble_samples = np.zeros((n_rows, S), dtype=float)

        for model_idx in range(len(self._base_residuals)):
            residuals = self._base_residuals[model_idx]

            # Fjern NaN residualer
            residuals_clean = residuals[~np.isnan(residuals)]

            if len(residuals_clean) == 0:
                # Hvis ingen residualer for denne modellen, bruk punkt-prediksjon
                for row_idx in range(n_rows):
                    model_base_pred = X_meta_future[row_idx, model_idx]
                    ensemble_samples[row_idx, :] += weights[model_idx] * model_base_pred
                continue

            # Beregn empirisk standardavvik av residualer
            residual_std = np.std(residuals_clean)

            # For hver prediksjon: tegn samples fra residual-fordelingen
            for row_idx in range(n_rows):
                # Tegn S samples fra observerte residualer (med replacement)
                sampled_residuals = np.random.choice(
                    residuals_clean, size=S, replace=True
                )
                # Legg til residualer til punkt-prediksjonen
                model_base_pred = X_meta_future[row_idx, model_idx]
                model_samples = model_base_pred + sampled_residuals

                # Sikre at prediksjoner ikke blir negative (for count-data)
                model_samples = np.maximum(model_samples, 0.0)

                # Vekt med modell-vekt
                ensemble_samples[row_idx, :] += weights[model_idx] * model_samples

        # Pakk tilbake til DataSet[Samples]
        df_out = df_future.copy()
        result: dict[Any, Samples] = {}

        # Bygg en rad per location/time_period med samples
        for loc in sorted(future_data.locations()):
            mask = df_future["location"] == loc
            future_sample = future_data[loc]
            tp = future_sample.time_period

            # Hent samples for denne lokasjonen
            loc_rows_idx = np.where(np.asarray(mask))[0]
            if len(loc_rows_idx) != len(tp):
                raise ValueError(
                    f"Antall rader for {loc} matcher ikke time_period"
                )

            loc_samples = ensemble_samples[loc_rows_idx, :]
            samples_obj = Samples(samples=loc_samples, time_period=tp)
            result[loc] = samples_obj

        return DataSet(result)