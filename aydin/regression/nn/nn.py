from os.path import join

import gc
import numpy
import random

from aydin.io.folders import get_temp_folder
from aydin.plaidml.plaidml_provider import PlaidMLProvider
from aydin.regression.nn.callback import NNCallback

# NOTE: This line should stay exactly here!
provider = PlaidMLProvider()


from aydin.regression.nn.models import feed_forward, yinyang2, back_feed
from aydin.regression.regressor_base import RegressorBase


from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import optimizers, Model


class NNRegressor(RegressorBase):
    """
    Regressor that uses CNN.

    """

    model: Model

    def __init__(
        self, max_epochs=1024, learning_rate=0.001, patience=10, depth=16, loss='l1'
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
        :param patience:
        :type patience:
        """

        self.debug_log = True

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.depth = depth

        loss = 'mae' if loss == 'l1' else loss
        loss = 'mse' if loss == 'l2' else loss
        self.loss = loss

        self.model = None

        self.reset()

    def progressive(self):
        return False

    def reset(self):

        del self.model
        self.model = None

        self.model_file_path = join(
            get_temp_folder(),
            f"aydin_nn_keras_model_file_{random.randint(0, 1e16)}.hdf5",
        )
        self.checkpoint = ModelCheckpoint(
            self.model_file_path, monitor='val_loss', verbose=1, save_best_only=True
        )

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

        """

        # First we make sure that the arrays are of a type supported:
        def assert_type(array):
            assert (
                (array.dtype == numpy.float64)
                or (array.dtype == numpy.float32)
                or (array.dtype == numpy.uint16)
                or (array.dtype == numpy.uint8)
            )

        assert_type(x_train)
        assert_type(y_train)
        assert_type(x_valid)
        assert_type(y_valid)

        # Types have to be consistent between train and valid sets:
        assert x_train.dtype is x_valid.dtype
        assert y_train.dtype is y_valid.dtype

        # In case the y dtype does not match the x dtype, we rescale and cast y:
        if numpy.issubdtype(x_train.dtype, numpy.integer) and numpy.issubdtype(
            y_train.dtype, numpy.floating
        ):

            # We remember the original type of y:
            self.original_y_dtype = y_train.dtype

            if x_train.dtype == numpy.uint8:
                y_train *= 255
                y_train = y_train.astype(numpy.uint8)
                y_valid *= 255
                y_valid = y_valid.astype(numpy.uint8)
                self.original_y_scale = 1 / 255.0

            elif x_train.dtype == numpy.uint16:
                y_train *= 255 * 255
                y_train = y_train.astype(numpy.uint16)
                y_valid *= 255 * 255
                y_valid = y_valid.astype(numpy.uint16)
                self.original_y_scale = 1 / (255.0 * 255.0)
        else:
            self.original_y_dtype = None

        # Get the number of entries and features from the array shape:
        nb_training_entries = x_train.shape[0]
        feature_dim = x_train.shape[-1]

        # Shapes of both x and y arrays:
        x_shape = (-1, feature_dim)
        y_shape = (-1, 1)

        # Initialise model if not done yet:
        if self.model is None:

            self.model = feed_forward(feature_dim, depth=self.depth)

            if self.debug_log:
                print(f"Number of parameters in model: {self.model.count_params()}")
                # print(f"Summary: \n {self.model.summary()}")

            opt = optimizers.Adam(lr=self.learning_rate, decay=0.00001)
            self.model.compile(optimizer=opt, loss=self.loss)

        # Reshape arrays:
        x_train = x_train.reshape(x_shape)
        y_train = y_train.reshape(y_shape)
        x_valid = x_valid.reshape(x_shape)
        y_valid = y_valid.reshape(y_shape)

        # The bigger the batch size the faster training in terms of time per epoch,
        # but small batches are also better for convergence (inherent batch noise).
        # We make sure that we have at least about 1000 items per batch for small images,
        # which is a good minimum. For larger datasets we get bigger batches which is fine.
        batch_size = max(1, x_train.shape[0] // 256)

        if self.debug_log:
            print(f"Batch size: {batch_size}")
            print(f"Starting training...")

        # Effective number of epochs:
        effective_number_of_epochs = 2 if is_batch else self.max_epochs

        # Here is the list of callbacks:
        callbacks = []

        # Set upstream callback:
        self.keras_callback = NNCallback()
        self.keras_callback.regressor_callback = regressor_callback
        callbacks.append(self.keras_callback)

        # Set standard callbacks:
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.000001 if is_batch else 0.0001,
            patience=2 if is_batch else self.patience,
            verbose=1,
            mode='auto',
            restore_best_weights=True,
        )
        callbacks.append(self.early_stopping)

        self.reduce_learning_rate = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            verbose=1,
            patience=1 if is_batch else max(1, self.patience // 2),
            mode='auto',
            min_lr=1e-9,
        )
        callbacks.append(self.reduce_learning_rate)

        callbacks.append(self.checkpoint)

        # x_train = x_train.astype(numpy.float64)
        # y_train = y_train.astype(numpy.float64)
        # x_valid = x_valid.astype(numpy.float64)
        # y_valid = y_valid.astype(numpy.float64)

        # Training happens here:
        train_history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=effective_number_of_epochs,
            batch_size=min(batch_size, nb_training_entries),
            shuffle=True,
            verbose=2,  # 0 if is_batch else 1,
            callbacks=callbacks,
        )

        # Reload the best weights:
        self.model.load_weights(self.model_file_path)

        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss'][0]

        gc.collect()

        return val_loss

    def predict(self, x, model_to_use=None):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        :param model_to_use:
        :param x:
        :type x:
        :return:
        :rtype:
        """

        # Number of features:
        number_of_features = x.shape[-1]

        # How much memory is available in GPU:
        max_gpu_mem_in_bytes = provider.device_max_mem

        # We limit ourselves to using only a quarter of GPU memory:
        max_number_of_floats = (max_gpu_mem_in_bytes // 4) // 4

        # Max size of batch:
        max_gpu_batch_size = max_number_of_floats / number_of_features

        # Batch size taking all this into account:
        batch_size = max(1, min(max_gpu_batch_size, x.shape[0] // 256))

        if self.debug_log:
            print(f"Batch size: {batch_size}")

        if self.debug_log:
            print(f"Predicting. features shape = {x.shape}")

        yp = (
            self.model.predict(x, batch_size=batch_size)
            if model_to_use is None
            else model_to_use.predict(x, batch_size=batch_size)
        )

        # We cast back  yp to teh correct type and range:
        if not self.original_y_dtype is None:
            yp = yp.astype(self.original_y_dtype)
            yp *= self.original_y_scale

        return yp
