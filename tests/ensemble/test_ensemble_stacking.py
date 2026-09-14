import numpy as np

from chap_core.ensemble.ensemble_model import EnsembleModel


def test_probabilistic_predict_samples_count(weekly_full_data, constant_template_factory):
    templates = [
        constant_template_factory(3.0, 2, "model_a"),
        constant_template_factory(6.0, 2, "model_b"),
    ]
    n_samples = 6
    model = EnsembleModel(base_templates=templates, n_samples=n_samples)

    predictor = model.train(weekly_full_data)
    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        df_samples = preds[loc].to_pandas()
        sample_cols = [c for c in df_samples.columns if c.startswith("sample_")]
        assert len(sample_cols) == n_samples
        assert len(df_samples) == len(weekly_full_data[loc].time_period)


def test_probabilistic_ensemble_outputs_sorted_samples(weekly_full_data, constant_template_factory):
    templates = [
        constant_template_factory(3.0, 2, "model_a"),
        constant_template_factory(6.0, 2, "model_b"),
    ]
    model = EnsembleModel(base_templates=templates, n_samples=6)

    predictor = model.train(weekly_full_data)
    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        samples = (
            preds[loc].to_pandas()[[c for c in preds[loc].to_pandas().columns if c.startswith("sample_")]].to_numpy()
        )
        assert np.all(np.diff(samples, axis=1) >= -1e-8)
