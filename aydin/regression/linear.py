from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso


class LinearRegressor:
    """
    Linear Regressor.

    Note: Fast but overall poor performance -- as expected.

    """

    linear: LinearRegression

    def __init__(self, mode='linear'):
        """
        Constructs a linear regressor.

        """
        if mode == 'lasso':
            self.linear = Lasso(alpha=0.1)
        elif mode == 'huber':
            self.linear = HuberRegressor()
        elif mode == 'linear':
            self.linear = LinearRegression()

    def progressive(self):
        return False

    def reset(self):
        pass

    def fit(
        self,
        x_train,
        y_train,
        x_valid=None,
        y_valid=None,
        is_batch=False,
        regressor_callback=None,
    ):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        """
        if is_batch:
            raise NotImplemented("Batch training not het implemented!")
        self.linear = self.linear.fit(x_train, y_train)

    def predict(self, x, model_to_use=None):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        """
        return (
            self.linear.predict(x) if model_to_use is None else model_to_use.predict(x)
        )
