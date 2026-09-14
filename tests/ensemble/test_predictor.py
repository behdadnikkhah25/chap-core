import numpy as np

from chap_core.ensemble._meta_models import ProbabilisticMetaModel
from chap_core.ensemble._predictor import EnsemblePredictor


class _FixedMetaProbabilistic(ProbabilisticMetaModel):
    def __init__(self, coef):
        super().__init__()
        self.coef_ = np.asarray(coef, float)

    def predict(self, X_samples):
        coef = self.coef_
        assert coef is not None
        ens = sum(coef[i] * X_samples[i] for i in range(len(X_samples)))
        return np.maximum(ens, 0.0)


def test_predictor_probabilistic_samples(weekly_full_data, constant_predictor_factory):
    predictors = [constant_predictor_factory(1.0, 2), constant_predictor_factory(3.0, 2)]
    meta = _FixedMetaProbabilistic([0.5, 0.5])

    predictor = EnsemblePredictor(
        predictors=predictors,
        meta=meta,
        n_samples=4,
    )

    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == 4
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)
