import os
from enum import Enum

from pitl.plaidml.plaidml_provider import PlaidMLProvider
from pitl.plaidml.weightnorm import AdamWithWeightnorm

provider = PlaidMLProvider()


from pitl.regression.nn.models import yinyang, feed_forward
from pitl.regression.regressor_base import RegressorBase


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers, Model


class NNRegressor(RegressorBase):
    """
    Regressor that uses CNN.

    """

    nnreg: Model

    def __init__(self, max_epochs=1024, learning_rate=0.01, depth=16, loss='l1'):
        """
        Constructs a LightGBM regressor.

        :param num_leaves:
        :type num_leaves:
        :param net_width:
        :type net_width:
        :param learning_rate:
        :type learning_rate:
        :param eval_metric:
        :type eval_metric:
        :param early_stopping_rounds:
        :type early_stopping_rounds:
        """

        self.debug_log = True

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.depth = depth

        loss = 'mae' if loss == 'l1' else loss
        loss = 'mse' if loss == 'l2' else loss
        self.loss = loss

        self.nnreg = None

        self.EStop = EarlyStopping(
            monitor='val_loss', min_delta=0.0001, patience=4, verbose=1, mode='auto'
        )

        self.ReduceLR = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            verbose=1,
            patience=4,
            mode='auto',
            min_lr=1e-9,
        )

    def reset(self):

        del self.nnreg
        self.nnreg = None

    def fit_batch(self, x_train, y_train, x_valid=None, y_valid=None):

        self._fit(x_train, y_train, x_valid, y_valid, batch=True)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        """

        self.nnreg = None
        self._fit(x_train, y_train, x_valid, y_valid, batch=False)

    def _fit(self, x_train, y_train, x_valid, y_valid, batch=False):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        """

        nb_training_entries = x_train.shape[0]
        feature_dim = x_train.shape[-1]

        if self.nnreg is None:

            model = feed_forward(feature_dim, self.depth)
            self.x_shape = (-1, feature_dim)
            self.y_shape = (-1, 1)

            if self.debug_log:
                print(f"Number of parameters in model: {model.count_params()}")
                print(f"Summary: \n {model.summary()}")

            opt = optimizers.Adam(lr=self.learning_rate, decay=0.00001)
            model.compile(optimizer=opt, loss=self.loss)
            self.nnreg = model

        x_train = x_train.reshape(*self.x_shape)
        y_train = y_train.reshape(*self.y_shape)
        x_valid = x_valid.reshape(*self.x_shape)
        y_valid = y_valid.reshape(*self.y_shape)

        self.nnreg.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=10 if batch else self.max_epochs,
            batch_size=min(1024, nb_training_entries),
            callbacks=[self.EStop, self.ReduceLR],
        )

    def predict(self, x):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        :param x:
        :type x:
        :return:
        :rtype:
        """
        x = x.reshape(self.x_shape)
        return self.nnreg.predict(x)
