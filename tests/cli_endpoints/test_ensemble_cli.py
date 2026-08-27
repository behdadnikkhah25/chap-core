import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import yaml

from chap_core.api_types import BacktestParams, RunConfig
from chap_core.cli_endpoints import ensemble as ensemble_cli
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def test_evaluate_ensemble_smoke(weekly_full_data, tmp_path, monkeypatch):
    def fake_load_dataset(**_kwargs):
        return weekly_full_data

    monkeypatch.setattr(ensemble_cli, "_load_dataset", fake_load_dataset)

    class _DummyTemplate:
        def __init__(self, name: str, value: float):
            self.name = name
            self._value = value
            self.entered = False
            self.exited = False
            self.model_template_config = SimpleNamespace(
                required_covariates=[],
                allow_free_additional_continuous_covariates=True,
            )

        def __enter__(self):
            # The CLI must enter the template: for chapkit models this is what starts
            # the backing service, and skipping it left get_model raising at runtime.
            self.entered = True
            return self

        def __exit__(self, *_exc):
            self.exited = True
            return False

        def get_model(self, _config):
            assert self.entered, "get_model called before the template was entered"
            return lambda: _ConstantEstimator(self._value, 1)

    class _ConstantPredictor:
        def __init__(self, value: float, n_samples: int):
            self._value = value
            self._n_samples = n_samples

        def predict(self, _historic_data, future_data):
            result = {}
            for loc in future_data.locations():
                tp = future_data[loc].time_period
                vals = np.full(len(tp), self._value, dtype=float)
                df_samples = pd.DataFrame({"time_period": tp.topandas()})
                for i in range(self._n_samples):
                    df_samples[f"sample_{i}"] = vals
                result[loc] = Samples.from_pandas(df_samples)
            return DataSet(result)

    class _ConstantEstimator:
        def __init__(self, value: float, n_samples: int):
            from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation

            self._value = value
            self._n_samples = n_samples
            self.model_information = ModelTemplateInformation(min_prediction_length=1, max_prediction_length=1)

        def train(self, _train_data):
            return _ConstantPredictor(self._value, self._n_samples)

    created: list[_DummyTemplate] = []

    def fake_from_directory_or_github_url(cls, name, **_kwargs):
        value = 2.0 if "b" in name else 1.0
        template = _DummyTemplate(name, value)
        created.append(template)
        return template

    from chap_core.models.model_template import ModelTemplate

    monkeypatch.setattr(
        ModelTemplate,
        "from_directory_or_github_url",
        classmethod(fake_from_directory_or_github_url),
    )

    report_path = tmp_path / "ensemble_report.csv"
    results = ensemble_cli.evaluate_ensemble(
        base_model_names="model_a,model_b",
        ensemble_method="deterministic",
        dataset_name=None,
        dataset_country=None,
        dataset_csv=None,
        polygons_json=None,
        polygons_id_field="id",
        report_filename=report_path,
        output_file=None,
        backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
        run_config=RunConfig(),
        model_configuration_yaml=None,
        inner_val_periods=4,
        data_source_mapping=None,
        historical_context_years=1,
    )

    assert results
    assert report_path.with_suffix(".csv").exists()
    assert created, "no templates were loaded"
    assert all(t.entered and t.exited for t in created)


def test_evaluate_ensemble_rejects_invalid_method_before_loading_dataset(monkeypatch):
    def fail_if_called(**_kwargs):
        raise AssertionError("dataset loading should not happen for an invalid ensemble method")

    monkeypatch.setattr(ensemble_cli, "_load_dataset", fail_if_called)

    with pytest.raises(ValueError, match="ensemble_method must be"):
        ensemble_cli.evaluate_ensemble(
            base_model_names="model_a",
            ensemble_method="not-a-method",
            dataset_name="dummy",
            dataset_country=None,
            dataset_csv=None,
            polygons_json=None,
            polygons_id_field="id",
            report_filename=Path("ensemble_report.csv"),
            output_file=None,
            backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
            run_config=RunConfig(),
            model_configuration_yaml=None,
            inner_val_periods=4,
            n_samples=100,
            data_source_mapping=None,
            historical_context_years=1,
        )


def test_evaluate_ensemble_help_marks_command_experimental():
    """The experimental status must reach CLI users, not just the source."""
    help_text = ensemble_cli.evaluate_ensemble.__doc__
    assert help_text is not None
    assert help_text.lstrip().startswith("EXPERIMENTAL:")


def test_evaluate_ensemble_wraps_base_model_with_short_max_prediction_length(weekly_full_data, tmp_path, monkeypatch):
    def fake_load_dataset(**_kwargs):
        return weekly_full_data

    monkeypatch.setattr(ensemble_cli, "_load_dataset", fake_load_dataset)

    class _DummyTemplate:
        def __init__(self):
            self.model_template_config = SimpleNamespace(
                required_covariates=[],
                allow_free_additional_continuous_covariates=True,
            )

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def get_model(self, _config):
            class _Estimator:
                def __init__(self):
                    from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation

                    self.model_information = ModelTemplateInformation(min_prediction_length=1, max_prediction_length=1)

                def train(self, _train_data):
                    raise AssertionError("train should not be called in this test")

            return lambda: _Estimator()

    from chap_core.models.model_template import ModelTemplate

    monkeypatch.setattr(
        ModelTemplate,
        "from_directory_or_github_url",
        classmethod(lambda cls, _name, **_kwargs: _DummyTemplate()),
    )

    class _StopHere(RuntimeError):
        pass

    captured_kwargs = {}

    def _ensemble_ctor(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        raise _StopHere

    with (
        patch("chap_core.ensemble.ensemble_model.EnsembleModel", side_effect=_ensemble_ctor),
        patch("chap_core.cli_endpoints.ensemble.ExtendedPredictor") as ext_mock,
    ):
        with pytest.raises(_StopHere):
            ensemble_cli.evaluate_ensemble(
                base_model_names="model_a",
                ensemble_method="deterministic",
                dataset_name=None,
                dataset_country=None,
                dataset_csv=None,
                polygons_json=None,
                polygons_id_field="id",
                report_filename=tmp_path / "ensemble_report.csv",
                output_file=None,
                backtest_params=BacktestParams(n_periods=3, n_splits=1, stride=1),
                run_config=RunConfig(),
                model_configuration_yaml=None,
                inner_val_periods=4,
                n_samples=7,
                data_source_mapping=None,
                historical_context_years=1,
            )

    assert ext_mock.call_count == 1
    assert captured_kwargs["n_samples"] == 7


def test_write_meta_report_uses_report_stem(tmp_path):
    report_path = tmp_path / "run_01.csv"

    ensemble_cli._write_meta_report(report_path, ["a", "b"], [0.25, 0.75])

    assert (tmp_path / "run_01_meta.csv").exists()
    assert not (tmp_path / "ensemble_meta_report.csv").exists()


@pytest.mark.parametrize(
    "fixture_name",
    [
        "chap_ewars_monthly_config.yaml",
        "inla_baseline_config.yaml",
        "rwanda_sarimax_config.yaml",
    ],
)
def test_ensemble_config_fixtures_parse(fixture_name):
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "ensemble_config" / fixture_name
    config = ModelConfiguration.model_validate(yaml.safe_load(fixture_path.read_text(encoding="utf-8")))

    assert config is not None
