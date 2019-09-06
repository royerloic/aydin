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
        self._stop_fit = False

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
        """
        Loading internal data of the regressor  -- typically model parameters of 3rd party libs
        :param path:
        :type path:
        """
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
        Resets the regressor to a blank state.

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


        :param x_train: x training values
        :param y_train: y training values
        :param x_valid:  x validation values
        :param y_valid:  y validation values
        :param is_batch: if true does batch training
        :param regressor_callback: regressor callback

        """
        self._stop_fit = False

    def stop_fit(self):
        """
        Stops training (can be called by another thread)
        """
        self._stop_fit = True

    @abstractmethod
    def predict(self, x, model_to_use=None):
        """
        Predicts y given x by applying the learned function f: y=f(x)

        :param x: x values
        :param model_to_use:
        :return: inferred y values
        """
        raise NotImplementedError()
