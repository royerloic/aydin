from pitl.plaidml.plaidml_provider import PlaidMLProvider
from pitl.regression.nn.callback import NNCallback

provider = PlaidMLProvider()


from pitl.regression.nn.models import feed_forward, yinyang2, back_feed
from pitl.regression.regressor_base import RegressorBase


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers, Model


class NNRegressor(RegressorBase):
    """
    Regressor that uses CNN.

    """

    nnreg: Model

    def __init__(
        self,
        max_epochs=1024,
        learning_rate=0.02,
        early_stopping_rounds=5,
        depth=16,
        loss='l1',
    ):
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

        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=early_stopping_rounds,
            verbose=1,
            mode='auto',
            restore_best_weights=True,
        )

        self.reduce_learning_rate = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            verbose=1,
            patience=max(1, early_stopping_rounds // 2),
            mode='auto',
            min_lr=1e-9,
        )

        self.keras_callback = NNCallback()

    def reset(self):

        del self.nnreg
        self.nnreg = None

    def fit_batch(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):

        self._fit(
            x_train,
            y_train,
            x_valid,
            y_valid,
            batch=True,
            regressor_callback=regressor_callback,
        )

    def fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        """

        self.nnreg = None
        self._fit(
            x_train,
            y_train,
            x_valid,
            y_valid,
            batch=False,
            regressor_callback=regressor_callback,
        )

    def _fit(
        self, x_train, y_train, x_valid, y_valid, batch=False, regressor_callback=None
    ):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        """

        # TODO: parameter should be number_of_batches instead of batch, so the code here knows roughly what is happening above.

        nb_training_entries = x_train.shape[0]
        feature_dim = x_train.shape[-1]

        if self.nnreg is None:

            model = feed_forward(feature_dim, depth=self.depth)
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

        # The bigger the batch size the faster training in terms of time per epoch,
        # but small batches are also better for convergence (inherent batch noise).
        # We make sure that we have at least about 1000 items per batch for small images,
        # which is a good minimum. For larger datasets we get bigger batches which is fine.
        batch_size = max(1, x_train.shape[0] // 256)

        if self.debug_log:
            print(f"Batch size: {batch_size}")
            print(f"Starting training...")

        self.keras_callback.regressor_callback = regressor_callback

        self.nnreg.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=10 if batch else self.max_epochs,
            batch_size=min(batch_size, nb_training_entries),
            callbacks=[
                self.early_stopping,
                self.reduce_learning_rate,
                self.keras_callback,
            ],
        )

    def predict(self, x, model_to_use=None):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        :param model_to_use:
        :param x:
        :type x:
        :return:
        :rtype:
        """

        # TODO: batch size should also be limited based on VRAM size...
        batch_size = max(1, x.shape[0] // 256)

        if self.debug_log:
            print(f"Batch size: {batch_size}")

        if self.debug_log:
            print(f"Predicting. features shape = {x.shape}")

        x = x.reshape(self.x_shape)
        return (
            self.nnreg.predict(x, batch_size=batch_size)
            if model_to_use is None
            else model_to_use.predict(x, batch_size=batch_size)
        )
