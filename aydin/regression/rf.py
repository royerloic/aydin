import numpy

from aydin.regression.gbm import GBMRegressor


class RandomForrestRegressor(GBMRegressor):
    """
    Random Forrest Regressor (uses the LGBM library).

    """

    def __init__(
        self,
        num_leaves=1024,
        n_estimators=2048,
        max_bin=512,
        learning_rate=0.001,
        loss='l1',
        patience=5,
        verbosity=100,
    ):

        super().__init__(
            num_leaves, n_estimators, max_bin, learning_rate, loss, patience, verbosity
        )

    def _get_params(self, num_samples, dtype=numpy.float32):
        params = super()._get_params(num_samples, dtype)
        params["boosting_type"] = "rf"
        return params
