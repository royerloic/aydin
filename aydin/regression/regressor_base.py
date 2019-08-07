import collections
import os
from abc import ABC, abstractmethod
from os.path import join

import jsonpickle

from aydin.util.json import encode_indent
from aydin.util.log.logging import lprint


class RegressorBase(ABC):
    """
        Image Translator base class

    """

    def __init__(self):
        """

        """

    def save(self, path: str):
        """
        Saves an 'all-batteries-included' regressor at a given path (folder).
        :param path: path to save to
        """
        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        lprint("Saving regressor to: {path}")
        with open(join(path, "regressor.json"), "w") as json_file:
            json_file.write(frozen)

        return frozen

    @staticmethod
    def load(path: str):
        """
        Returns an 'all-batteries-included' regressor from a given path (folder).
        :param model_path: path to load from.
        """

        lprint("Loading regressor from: {path}")
        with open(join(path, "regressor.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)
        thawed._load_internals(path)

        return thawed

    @abstractmethod
    def _load_internals(self, path: str):
        raise NotImplementedError()

    @property
    @abstractmethod
    def progressive(self) -> bool:
        """
        A regressor is progressive if it supports training through multiple epochs.
        If that is teh case, the properties: max_epochs and patience should be defined.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
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
        raise NotImplementedError()

    @abstractmethod
    def fit(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        is_batch=False,
        regressor_callback=None,
    ):
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
