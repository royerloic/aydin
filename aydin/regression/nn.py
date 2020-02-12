import gc
import os

import psutil
import numpy
import random
from os.path import join, exists

# env variable setting is very important for XLA, must happen before all tensorflow imports!:
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
from tensorflow_core import optimizers
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.saving.model_config import model_from_json


from aydin.io.folders import get_temp_folder
from aydin.util.log.log import lsection, lprint


from aydin.regression.nn_utils.callbacks import (
    NNCallback,
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from aydin.regression.nn_utils.models import feed_forward
from aydin.regression.regressor_base import RegressorBase

# from keras import optimizers, Model
# from keras.engine.saving import model_from_json


class NNRegressor(RegressorBase):
    """
    Regressor that uses standard perceptron-like neural networks.

    """

    # Device max mem:
    device_max_mem = psutil.virtual_memory().total

    model: Model

    def __init__(
        self, max_epochs=1024, learning_rate=0.001, patience=10, depth=6, loss='l1'
    ):
        """
        Constructs a LightGBM regressor.
        :param max_epochs:
        :param learning_rate:
        :param patience:
        :param depth:
        :param loss:

        """

        super().__init__()

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.depth = depth

        loss = 'mae' if loss == 'l1' else loss
        loss = 'mse' if loss == 'l2' else loss
        self.loss = loss

        self.model = None

        self.reset()

    def save(self, path: str):
        super().save(path)
        if not self.model is None:
            # serialize model to JSON:
            keras_model_file = join(path, 'keras_model.txt')
            model_json = self.model.to_json()
            with open(keras_model_file, "w") as json_file:
                json_file.write(model_json)

            # serialize weights to HDF5:
            keras_model_file = join(path, 'keras_weights.txt')
            self.model.save_weights(keras_model_file)

        return model_json

    def _load_internals(self, path: str):
        # load JSON and create model:
        keras_model_file = join(path, 'keras_model.txt')
        with open(keras_model_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
        # load weights into new model:
        keras_model_file = join(path, 'keras_weights.txt')
        self.model.load_weights(keras_model_file)

    ## We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['early_stopping']
        del state['reduce_learning_rate']
        del state['checkpoint']
        del state['model']
        del state['keras_callback']
        return state

    def reset(self):

        del self.model
        self.model = None
        self.checkpoint = None

    def fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        """

        super().fit(
            x_train, y_train, x_valid, y_valid, regressor_callback=regressor_callback
        )

        with lsection(f"NN Regressor fitting:"):

            # First we make sure that the arrays are of a type supported:
            def assert_type(array):
                assert (
                    (array.dtype == numpy.float64)
                    or (array.dtype == numpy.float32)
                    or (array.dtype == numpy.uint16)
                    or (array.dtype == numpy.uint8)
                )

            # Do we have a validation dataset?
            has_valid_dataset = not x_valid is None and not y_valid is None

            assert_type(x_train)
            assert_type(y_train)
            if has_valid_dataset:
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
                    if has_valid_dataset:
                        y_valid *= 255
                        y_valid = y_valid.astype(numpy.uint8)
                    self.original_y_scale = 1 / 255.0

                elif x_train.dtype == numpy.uint16:
                    y_train *= 255 * 255
                    y_train = y_train.astype(numpy.uint16)
                    if has_valid_dataset:
                        y_valid *= 255 * 255
                        y_valid = y_valid.astype(numpy.uint16)
                    self.original_y_scale = 1 / (255.0 * 255.0)
            else:
                self.original_y_dtype = None

            # Get the number of entries and features from the array shape:
            nb_data_points = x_train.shape[0]
            num_features = x_train.shape[-1]

            lprint(f"Number of data points : {nb_data_points}")
            if has_valid_dataset:
                lprint(f"Number of validation data points: {x_valid.shape[0]}")
            lprint(f"Number of features per data point: {num_features}")

            # Shapes of both x and y arrays:
            x_shape = (-1, num_features)
            y_shape = (-1, 1)

            # Learning rate and decay:
            learning_rate = self.learning_rate
            learning_rate_decay = 0.1 * self.learning_rate
            lprint(f"Learning rate: {learning_rate}")
            lprint(f"Learning rate decay: {learning_rate_decay}")

            # Weight decay and noise:
            weight_decay = 0.01 * self.learning_rate
            noise = 0.1 * self.learning_rate
            lprint(f"Weight decay: {weight_decay}")
            lprint(f"Added noise: {noise}")

            # Initialise model if not done yet:
            if self.model is None:
                self.model = feed_forward(
                    num_features,
                    depth=self.depth,
                    weight_decay=weight_decay,
                    noise=noise,
                )
                opt = optimizers.Adam(lr=learning_rate, decay=learning_rate_decay)
                self.model.compile(optimizer=opt, loss=self.loss)

            lprint(f"Number of parameters in model: {self.model.count_params()}")

            # Reshape arrays:
            x_train = x_train.reshape(x_shape)
            y_train = y_train.reshape(y_shape)

            if not x_valid is None and not y_valid is None:
                x_valid = x_valid.reshape(x_shape)
                y_valid = y_valid.reshape(y_shape)

            batch_size = 1024
            lprint(f"Keras batch size for training: {batch_size}")

            # Effective number of epochs:
            effective_number_of_epochs = self.max_epochs
            lprint(f"Effective max number of epochs: {effective_number_of_epochs}")

            # Early stopping patience:
            early_stopping_patience = self.patience
            lprint(f"Early stopping patience: {early_stopping_patience}")

            # Effective LR patience:
            effective_lr_patience = max(1, self.patience // 2)
            lprint(f"Effective LR patience: {effective_lr_patience}")

            # Here is the list of callbacks:
            callbacks = []

            # Set upstream callback:
            self.keras_callback = NNCallback(regressor_callback)

            # Early stopping callback:
            self.early_stopping = EarlyStopping(
                self,
                monitor='val_loss',
                min_delta=min(0.000001, 0.1 * self.learning_rate),
                patience=early_stopping_patience,
                mode='auto',
                restore_best_weights=True,
            )

            # Reduce LR on plateau:
            self.reduce_learning_rate = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                verbose=1,
                patience=effective_lr_patience,
                mode='auto',
                min_lr=0.0001 * self.learning_rate,
            )

            if self.checkpoint is None:
                self.model_file_path = join(
                    get_temp_folder(),
                    f"aydin_nn_keras_model_file_{random.randint(0, 1e16)}.hdf5",
                )
                self.checkpoint = ModelCheckpoint(
                    self.model_file_path,
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                )

            # Add callbacks to the list:
            callbacks.append(self.keras_callback)
            callbacks.append(self.early_stopping)
            callbacks.append(self.reduce_learning_rate)
            callbacks.append(self.checkpoint)

            # x_train = x_train.astype(numpy.float64)
            # y_train = y_train.astype(numpy.float64)
            # x_valid = x_valid.astype(numpy.float64)
            # y_valid = y_valid.astype(numpy.float64)

            # Training happens here:
            with lsection("NN regressor fitting now:"):
                train_history = self.model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_valid, y_valid)
                    if (not x_valid is None and not y_valid is None)
                    else None,
                    epochs=effective_number_of_epochs,
                    batch_size=min(batch_size, nb_data_points),
                    shuffle=True,
                    verbose=0,  # 0 if is_batch else 1,
                    callbacks=callbacks,
                )
                lprint(f"NN regressor fitting done.")

            del x_train
            del y_train
            gc.collect()

            # Reload the best weights:
            if exists(self.model_file_path):
                lprint(f"Loading best model to date.")
                self.model.load_weights(self.model_file_path)

            # loss_history = train_history.history['loss']
            # lprint(f"Loss history after training: {loss_history}")

            if 'val_loss' in train_history.history:
                val_loss = train_history.history['val_loss'][0]
                return val_loss
            else:
                return None

    def predict(self, x, model_to_use=None):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        :param model_to_use:
        :param x:
        :type x:
        :return:
        :rtype:
        """

        with lsection(f"NN Regressor prediction:"):

            lprint(f"Number of data points             : {x.shape[0]}")
            lprint(f"Number of features per data points: {x.shape[-1]}")

            # Number of features:
            num_of_features = x.shape[-1]

            # We check that we get the right number of features.
            # If not, most likely the batch_dims are set wrong...
            assert num_of_features == x.shape[-1]

            # Batch size such that we have n=128 batches:
            batch_size = max(1, x.shape[0] // 128)

            lprint(f"Batch size: {batch_size}")
            lprint(f"Predicting. features shape = {x.shape}")

            lprint(f"NN regressor predicting now...")
            yp = (
                self.model.predict(x, batch_size=batch_size)
                if model_to_use is None
                else model_to_use.predict(x, batch_size=batch_size)
            )
            lprint(f"NN regressor predicting done!")

            # We cast back yp to the correct type and range:
            if not self.original_y_dtype is None:
                yp = yp.astype(self.original_y_dtype)
                yp *= self.original_y_scale

            return yp
