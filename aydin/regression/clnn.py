import gc
import math
import random
from os.path import join

import jsonpickle
import numpy

from aydin.clnn.modules.linear.dense import Dense
from aydin.clnn.modules.losses.l1 import L1Loss
from aydin.clnn.modules.losses.l2 import L2Loss
from aydin.clnn.modules.nonlin.relu import ReLU
from aydin.clnn.modules.special.noise import Noise
from aydin.clnn.modules.special.resize import Resize
from aydin.clnn.optimizers.adam import ADAM
from aydin.clnn.optimizers.sgd import SGD
from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor
from aydin.regression.regressor_base import RegressorBase
from aydin.util.json import encode_indent
from aydin.util.log.log import lsection, lprint


class CLNNRegressor(RegressorBase):
    """
    Regressor that uses standard perceptron-like neural networks implemented in OpenCL.

    """

    def __init__(
        self,
        max_epochs=1024,
        learning_rate=0.001,
        patience=10,
        depth=6,
        layer_width=None,
        loss='l1',
        tensor_class=CLTensor,
        activation_class=ReLU,
        optimiser_class=ADAM,
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
        self.layer_width = layer_width
        self.tensor_class = tensor_class
        self.activation_class = activation_class
        self.optimiser_class = optimiser_class

        if isinstance(self.activation_class, ReLU):
            self.activation_class.residual_gradient = 0.1 * learning_rate

        loss = L1Loss if loss == 'l1' else loss
        loss = L2Loss if loss == 'l2' else loss
        self.loss_function_class = loss

        self.active_layers_depth = 3
        self.dense_layers = None
        self.layers = None
        self.modules = None
        self.losses = None
        self.best_layer_index = None

        self.reset()

    def save(self, path: str):
        super().save(path)

        map_to_save = {
            'dense': [l.to_tensor_class(NPTensor) for l in self.dense_layers],
            'layers': [l.to_tensor_class(NPTensor) for l in self.layers],
            'modules': [l.to_tensor_class(NPTensor) for l in self.modules],
            'losses': [l.to_tensor_class(NPTensor) for l in self.losses],
        }

        frozen = encode_indent(map_to_save)

        lprint(f"Saving clnn model to: {path}")
        with open(join(path, "clnn_model.json"), "w") as json_file:
            json_file.write(frozen)

        return None

    def _load_internals(self, path: str):
        # load JSON and create model:

        lprint(f"Loading clnn model from: {path}")
        with open(join(path, "clnn_model.json"), "r") as json_file:
            frozen = json_file.read()

        thawed_map = jsonpickle.decode(frozen)

        self.dense_layers = [
            l.to_tensor_class(self.tensor_class) for l in thawed_map['dense']
        ]
        self.layers = [
            l.to_tensor_class(self.tensor_class) for l in thawed_map['layers']
        ]
        self.modules = [
            l.to_tensor_class(self.tensor_class) for l in thawed_map['modules']
        ]
        self.losses = [
            l.to_tensor_class(self.tensor_class) for l in thawed_map['losses']
        ]

        pass

    ## We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['dense_layers']
        del state['layers']
        del state['modules']
        del state['losses']
        return state

    def reset(self):
        del self.dense_layers
        del self.layers
        del self.modules
        del self.losses
        self.dense_layers = None
        self.layers = None
        self.modules = None
        self.losses = None

    def init_model(self, num_features, input_noise_level):

        # Addresses a bug (all zero output) that appears when running multiple denoising in same process
        if self.tensor_class == CLTensor:
            CLTensor.opencl_provider = None

        # Initialise model if not done yet:
        layer_width = self.layer_width
        if layer_width is None:
            layer_width = num_features
        self.dense_layers = []
        self.layers = []
        self.modules = []
        self.losses = []
        current_layer = Noise(None, noise_level=input_noise_level)
        for i in range(self.depth):
            dense = Dense(
                current_layer,
                nb_inputs=num_features if i == 0 else layer_width,
                nb_outputs=layer_width,
            )
            dense._learning_rate = 1.0
            dense.ensure_parameters_allocated(self.tensor_class(0.0))
            if i > 0:
                # We bias the network to do the smart thing: reuse the previous estimate as-is first before trying to make it better...
                weights = dense.weights.nparray
                weights[:, 0] = 0.0
                weights[0, 0] = 1.0
                dense.weights.nparray = weights
            self.dense_layers.append(dense)
            act = self.activation_class(dense)
            self.layers.append(act)
            current_layer = act

            crop = Resize(act, shape=(1,))
            self.modules.append(crop)

            loss = self.loss_function_class(crop)
            self.losses.append(loss)
        lprint(f"Architecture: {loss.tree()}")
        lprint(f"Number of parameters in model: {self.modules[-1].count_parameters()}")

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

        with lsection(f"CLNN Regressor fitting:"):

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
            input_noise_level = 0.1 * self.learning_rate
            lprint(f"Weight decay: {weight_decay}")
            lprint(f"Added noise: {input_noise_level}")

            # initialise model
            self.init_model(num_features, input_noise_level)

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
            lr_patience = max(1, self.patience // 2)
            lprint(f"Learning-rate patience: {lr_patience}")

            # Network resize patience:
            network_grow_patience = max(1, int(self.patience * 0.90))
            network_shrink_patience = max(1, int(self.patience * 0.10))
            lprint(f"Network grow patience: {network_grow_patience}")
            lprint(f"Network shrink patience: {network_shrink_patience}")

            # Optimiser
            optimiser = self.optimiser_class(
                self.losses[-1],
                learning_rate=self.learning_rate,
                parameter_decay=weight_decay,
            )
            optimiser.reset()

            # Mini batch size and num:
            mini_batch_size = 1024
            num_mini_batches = max(1, nb_data_points // mini_batch_size)
            lprint(f"Mini-batch size: {mini_batch_size}")
            lprint(f"Number of mini-batches: {num_mini_batches}")

            x_train_t = self.tensor_class(x_train)
            y_train_t = self.tensor_class(y_train)

            x_valid_t = self.tensor_class(x_valid)
            y_valid_t = self.tensor_class(y_valid)

            x_batch = self.tensor_class.instanciate(
                shape=(mini_batch_size, num_features), dtype=numpy.float32
            )
            y_batch = self.tensor_class.instanciate(
                shape=(mini_batch_size, 1), dtype=numpy.float32
            )

            best_val_loss_values = None
            best_val_loss_ever = math.inf
            best_state = [None] * self.depth

            best_layer_index = self.depth - 2
            self.best_layer_index = best_layer_index
            no_improvement_counter = 0
            update_layer_counter = 0
            iteration = 0

            # Training happens here:
            with lsection(
                f"CLNN regressor fitting now for at most {effective_number_of_epochs} epochs:"
            ):
                for e in range(effective_number_of_epochs):

                    if self._stop_fit:
                        lprint(f"Training interupted!")
                        break

                    with lsection(
                        f"Epoch {e}: CLNN Regressor fitting {num_mini_batches} mini-batches of size {mini_batch_size}:"
                    ):

                        for i in range(num_mini_batches):
                            seed = random.randint(0, 2 ** 30)
                            x_batch.sample(x_train_t, seed)
                            y_batch.sample(y_train_t, seed)

                            # forward pass:
                            for loss in self.losses[: self.active_layers_depth]:
                                loss(x_batch, y_batch)

                            # zero gradients:
                            self.losses[self.active_layers_depth - 1].zero_gradients()

                            # backward pass:
                            for loss in self.losses[: self.active_layers_depth]:
                                loss.backprop()

                            optimiser.module = self.losses[self.active_layers_depth - 1]
                            optimiser.step()

                        with lsection(f"Sanity check..."):
                            optimiser.module.check_for_zeros_parameters()
                            optimiser.module.check_for_nans_and_infs()

                        with lsection(f"Validating..."):
                            if has_valid_dataset:
                                # How many layers are  currently used:
                                lprint(
                                    f"Active layers: {[i for i in range(self.active_layers_depth)]}"
                                )
                                # we compute the val losses:
                                val_loss_results = []
                                for loss in self.losses[: self.active_layers_depth]:
                                    val_loss_result = loss(x_valid_t, y_valid_t)
                                    val_loss_results.append(val_loss_result)
                                val_loss_values = [
                                    val_loss_result.mean_all()
                                    for val_loss_result in val_loss_results
                                ]
                                lprint(
                                    f"Current losses     : {' '.join([str(x) for x in val_loss_values])}"
                                )
                                # In case we don't have a 'best val loss' values yet:
                                if not best_val_loss_values:
                                    best_val_loss_values = val_loss_values.copy()
                                while len(best_val_loss_values) < len(val_loss_values):
                                    best_val_loss_values.append(math.inf)
                                lprint(
                                    f"Overall best losses: {' '.join([str(x) for x in best_val_loss_values])}"
                                )

                                val_loss = min(val_loss_values)
                                lprint(f"Current min val loss: {val_loss}")

                                best_layer_index = val_loss_values.index(
                                    min(val_loss_values)
                                )
                                lprint(f"Current best layer : {best_layer_index}")

                                loss_comparison = [
                                    x < y
                                    for x, y in zip(
                                        val_loss_values, best_val_loss_values
                                    )
                                ]
                                if any(loss_comparison):
                                    lprint(
                                        f"loss improvement: {loss_comparison}, saving state"
                                    )

                                    # print(f"len(val_loss_values)={len(val_loss_values)}, len(best_val_loss_values)={len(best_val_loss_values)}, self.active_layers_depth={self.active_layers_depth} ")
                                    for j in range(self.active_layers_depth):
                                        if val_loss_values[j] < best_val_loss_values[j]:
                                            best_val_loss_values[j] = val_loss_values[j]
                                            best_state[j] = self.dense_layers[j].state

                                best_val_loss_across_layers = min(best_val_loss_values)
                                lprint(
                                    f"All-layer Best loss: {best_val_loss_across_layers}"
                                )
                                self.best_layer_index = best_val_loss_values.index(
                                    best_val_loss_across_layers
                                )
                                lprint(f"All-layer best layer: {self.best_layer_index}")

                                if best_val_loss_across_layers < best_val_loss_ever:
                                    lprint(f"Overall validation loss has improved!")
                                    best_val_loss_ever = best_val_loss_across_layers
                                    no_improvement_counter = 0
                                else:
                                    lprint(
                                        f"Overall validation loss has _not_ improved! (counter={no_improvement_counter+1})"
                                    )
                                    no_improvement_counter += 1

                                # make sure we end an epoch with the best possible state:
                                lprint(f"Reloading best state so far for all layers.")
                                for i in reversed(range(self.active_layers_depth)):
                                    if best_state[i]:
                                        self.dense_layers[i].state = best_state[i]

                        with lsection(
                            f"Learning loop management early stopping, learning rate patience and decay, active layers etc..."
                        ):
                            if no_improvement_counter > early_stopping_patience:
                                lprint(
                                    f"Early stopping! (patience = {early_stopping_patience}, counter={no_improvement_counter})"
                                )
                                break

                            if no_improvement_counter > lr_patience:
                                optimiser.learning_rate *= math.pow(
                                    0.5, 1.0 / (early_stopping_patience - lr_patience)
                                )
                                lprint(
                                    f"Stepping down learning rate because no improvement for too long! (patience = {lr_patience}, counter={no_improvement_counter})"
                                )

                            optimiser.learning_rate *= (
                                1.0 - learning_rate_decay
                            ) ** num_mini_batches
                            lprint(f"New learning rate: {optimiser.learning_rate}")

                            last_layer_index = self.active_layers_depth - 1
                            before_last_layer_index = last_layer_index - 1
                            if best_layer_index == last_layer_index:
                                if (
                                    update_layer_counter > network_grow_patience
                                    and self.active_layers_depth < self.depth
                                ):
                                    self.active_layers_depth += 1
                                    update_layer_counter = 0
                                    lprint(
                                        f"Last-layers validation performance is saturated -> number of layers increased (nb active layers = {self.active_layers_depth})"
                                    )
                                else:
                                    update_layer_counter += 1
                                    lprint(
                                        f"Last-layers validation performance is saturated, network growing eventually needed (count_down={network_grow_patience-update_layer_counter})"
                                    )
                            elif best_layer_index == before_last_layer_index:
                                update_layer_counter = 0
                                lprint(
                                    f"Validation performance across layers is optimal no need to shrink or grow network"
                                )
                            elif best_layer_index < before_last_layer_index:
                                if (
                                    update_layer_counter > network_shrink_patience
                                    and self.active_layers_depth > 0
                                ):
                                    self.active_layers_depth -= 1
                                    update_layer_counter = 0
                                    lprint(
                                        f"Last-layers validation performance is weak -> number of layers decreased (nb active layers = {self.active_layers_depth})"
                                    )
                                else:
                                    update_layer_counter += 1
                                    lprint(
                                        f"Last-layers validation performance is weak, network shrinking eventually needed (count_down={network_shrink_patience-update_layer_counter})"
                                    )

                        if regressor_callback:
                            module = self.modules[self.best_layer_index]
                            module_np = module.to_tensor_class(NPTensor)
                            model = lambda x: module_np(x)
                            regressor_callback(iteration, best_val_loss_ever, model)

                        # Increment iterations:
                        iteration += 1

                lprint(f"CLNN regressor fitting done.")

            del x_train
            del y_train
            gc.collect()

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

        with lsection(f"CLNN Regressor prediction:"):

            lprint(f"Number of data points             : {x.shape[0]}")
            lprint(f"Number of features per data points: {x.shape[-1]}")

            # Number of features:
            num_of_features = x.shape[-1]

            # We check that we get the right number of features.
            # If not, most likely the batch_dims are set wrong...
            assert num_of_features == x.shape[-1]

            x_t = self.tensor_class(x)

            lprint(f"Predicting. features shape = {x.shape}")

            lprint(f"CLNN regressor predicting now...")
            y_t = (
                self.modules[self.best_layer_index](x_t)
                if model_to_use is None
                else model_to_use(x_t)
            )
            lprint(f"CLNN regressor predicting done!")

            y_np = y_t.nparray

            # We cast back yp to the correct type and range:
            if not self.original_y_dtype is None:
                y_np = y_np.astype(self.original_y_dtype)
                y_np *= self.original_y_scale

            return y_np
