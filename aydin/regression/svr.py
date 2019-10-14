from sklearn.svm import SVR, LinearSVR


class SupportVectorRegressor:
    """
    Support Vector Regressor.

    Note: Way too slow when non-linear, nearly useless...
    When using linear much faster, but does not perform better than straight linear regression...


    """

    svr: SVR

    def __init__(self, linear=True):
        """
        Constructs a linear regressor.

        """
        if linear:
            self.svr = LinearSVR()  # gamma='scale')
        else:
            self.svr = SVR(gamma='scale')

    def progressive(self):
        return False

    def reset(self):
        pass

    def fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        """
        self.svr = self.svr.fit(x_train, y_train)

    def predict(self, x, model_to_use=None):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        """
        return self.svr.predict(x) if model_to_use is None else model_to_use.predict(x)
