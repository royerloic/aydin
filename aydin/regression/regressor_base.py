import collections
from abc import ABC, abstractmethod


RegressorCallbackTuple = collections.namedtuple(
    "RegressorCallbackEnv",
    [
        "model",
        "params",
        "iteration",
        "begin_iteration",
        "end_iteration",
        "evaluation_result_list",
    ],
)


class RegressorBase(ABC):
    """
        Image Translator base class

    """

    def __init__(self):
        """

        """

        self.callbacks = []
        self.monitoring_datasets = []

        self.num_features = None

    @abstractmethod
    def reset(self):
        """
        resets the regressor to a blank state.

        :param x_train: x training values
        :type x_train:
        :param y_train: y training values
        :type y_train:
        :param x_test:
        :type x_test: x test values
        :param y_test:
        :type y_test: y test values
        """
        self.num_features = None

    @abstractmethod
    def fit_batch(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the validation dataset: (x_valid, y_valid).

        This method can be called multiple times with different batches.
        To reset the regressor call reset()


        :param x_train: x training values
        :type x_train:
        :param y_train: y training values
        :type y_train:
        :param x_valid:  x validation values
        :type x_valid:
        :param y_valid:  y validation values
        :type y_valid:
        :param monitoring_variables:
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, x_train, y_train, x_valid, y_valid, regressor_callback=None):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).


        :param monitoring_variables:
        :param x_train: x training values
        :type x_train:
        :param y_train: y training values
        :type y_train:
        :param x_valid:  x validation values
        :type x_valid:
        :param y_valid:  y validation values
        :type y_valid:
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x, model_to_use=None):
        """
        Predicts y given x by applying the learned function f: y=f(x)

        :param model_to_use:
        :param x: x values
        :type x:
        :return: inferred y values
        :rtype:
        """
        raise NotImplementedError()
