import math
import random
import time
from os.path import join

import numpy

from aydin.it.exceptions.base import ArrayShapeDoesNotMatchError
from aydin.providers.opencl.opencl_provider import OpenCLProvider

provider = OpenCLProvider()

import tensorflow as tf  # noqa: E402
from aydin.io.folders import get_temp_folder  # noqa: E402
from aydin.it.cnn.models.unet_2d import UNet2DModel  # noqa: E402
from aydin.it.cnn.models.unet_3d import Unet3DModel  # noqa: E402
from aydin.it.it_base import ImageTranslatorBase  # noqa: E402
from aydin.util.log.log import lsection, lprint  # noqa: E402

from aydin.it.cnn.layers.maskout import Maskout  # noqa: E402
from aydin.it.cnn.util.mask_generator import maskedgen, randmaskgen  # noqa: E402
from aydin.it.cnn.layers.util import rot90, split  # noqa: E402
from aydin.regression.nn_utils.callbacks import ModelCheckpoint  # noqa: E402
from aydin.it.cnn.util.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    CNNCallback,
)  # noqa: E402

from aydin.it.cnn.util.data_util import (
    random_sample_patches,
    sim_model_size,
)  # noqa: E402
from aydin.it.cnn.util.validation_generator import (
    val_img_generator,
    val_data_generator,
)  # noqa: E402


class ImageTranslatorCNN(ImageTranslatorBase):
    """
        aydin CNN Image Translator

        Using CNN (Unet and Co)

    """

    verbose = 0

    def __init__(
        self,
        normaliser_type: str = 'percentile',
        monitor=None,
        training_architecture=None,
        model_architecture='unet',
        batch_size=32,
        num_layer=5,
        initial_units=8,
        batch_norm=None,
        activation='ReLU',
        patch_size=None,
        total_num_patches=None,
        adoption_rate=0.5,
        mask_shape: tuple = (9, 9),
        random_mask_ratio=0.1,
        EStop_patience=None,
        ReduceLR_patience=None,
        min_delta=0,
        mask_p=0.1,
        max_epochs=1024,
        patience=4,
        learn_rate=0.01,
        reduce_lr_factor=0.3,
        replace_by='zero',  # 'zero', 'radom', 'median'
        weight_decay=0,
        use_residual=False,
        pooling_mode='max',
        interpolation='nearest',
    ):
        """
        :param monitor: an object for monitoring training (only for GUI)
        :param training_architecture: 'shiftconv' or 'checkerbox' or 'randommasking' or 'checkran' architecture
        :param batch_size: batch size for training
        :param num_layer: number of layers
        :param initial_units: number of filters in the 1st layer of CNN
        :param batch_norm; type of batch normalization (e.g. batch, instance)
        :param patch_size: Size for patch sample e.g. 64 or (64, 64)
        :param total_num_patches: total number of patches for training
        :param adoption_rate: % of random patches will be used for training, the rest will be discarded
        :param mask_shape: mask shape for masking architecture; has to be the same size as the spatial dimension.
        :param random_mask_ratio: probability of masked pixels in random masking approach
        :param EStop_patience: patience for early stopping
        :param ReduceLR_patience: patience for reduced learn rate
        :param min_delta: 1e-6; minimum delta to determine if loss is dropping
        :param mask_p: possibility of masked pixels in ramdom masking approach
        :param max_epochs: maximum epoches
        :param patience: patience for EarlyStop or ReducedLR to be triggered
        :param learn_rate: initial learn rate
        :param reduce_lr_factor: reduce learn rate by factor of this value when it reaches plateau
        :param replace_by: <Only for randommasking architecture> replace masked pixels by 0 or random or median values.
        :param weight_decay: regularization facotr for L1 regularizers
        :param use_residual: use Add layers insead of Concatenate layers at merging layers in Unet.
        :param pooling_mode: 'max' for max pooling, 'ave' for average pooling
        :param interpolation: 'nearest' or 'bilinear' for Upsampling2D

        """
        super().__init__(normaliser_type, monitor)
        self.model = None  # a CNN model
        self.infmodel = None  # inference model
        self.supervised = True
        self.training_architecture = training_architecture
        self.model_architecture = (model_architecture,)
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.initial_units = initial_units
        self.batch_norm = batch_norm
        self.activation_fun = activation
        self.patch_size = patch_size
        self.total_num_patches = total_num_patches
        self.adoption_rate = adoption_rate
        self.mask_shape = mask_shape
        self.random_mask_ratio = random_mask_ratio
        self.EStop_patience = EStop_patience
        self.ReduceLR_patience = ReduceLR_patience
        self.min_delta = min_delta
        self.p = mask_p
        self.max_epochs = max_epochs
        self.patience = patience
        self.learn_rate = learn_rate
        self.reduce_lr_factor = reduce_lr_factor
        self.replace_by = replace_by
        self.weight_decay = weight_decay
        self.use_residual = use_residual
        self.pooling_mode = pooling_mode
        self.interpolation = interpolation

        self.batch_dim_upto3 = None
        self.axes_permutation = None
        self.image_shape_upto3 = None
        self.callback_period = None
        self.checkpoint = None
        self.input_dim = None
        self.stop_fitting = False
        self.rot_batch_size = None
        self.img_val = None
        self.val_marker = None
        self._batch_processing = (
            False  # whether validation data is chosen from pixels or patches
        )

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).
        :param path: path to save to
        """
        with lsection(f"Saving 'CNN' image translator to {path}"):
            frozen = super().save(path)
            frozen += self.save_cnn(path) + '\n'
        return frozen

    def save_cnn(self, path: str):
        model_json = None
        if self.model is not None:
            # serialize model to JSON:
            keras_model_file = join(path, 'keras_model.txt')
            model_json = self.model.to_json()
            with open(keras_model_file, "w") as json_file:
                json_file.write(model_json)

            # serialize weights to HDF5:
            keras_model_file = join(path, 'keras_weights.txt')
            self.model.save_weights(keras_model_file)
        return model_json

    def __getstate__(self):
        state = self.__dict__.copy()
        # exclude fields below that should/cannot be saved properly:
        del state['early_stopping']
        del state['reduce_learning_rate']
        del state['checkpoint']
        del state['model']
        return state

    def _load_internals(self, path: str):
        with lsection(f"Loading 'classic' image translator from {path}"):
            # load JSON and create model:
            keras_model_file = join(path, 'keras_model.txt')
            with open(keras_model_file, 'r') as json_file:
                loaded_model_json = json_file.read()
                self.model = tf.keras.models.model_from_json(
                    loaded_model_json,
                    custom_objects={'Maskout': Maskout, 'split': split, 'rot90': rot90},
                )
            # load weights into new model:
            keras_model_file = join(path, 'keras_weights.txt')
            self.model.load_weights(keras_model_file)

    def get_receptive_field_radius(self, num_layer, shiftconv=False):
        if 'dncnn' in self.model_architecture:
            if shiftconv:
                rf = 10 if num_layer <= 2 else 2 + num_layer * 4
            else:
                rf = 5 if num_layer <= 2 else 1 + num_layer * 2
            return int(rf // 2)
        else:
            if shiftconv:
                rf = 7 if num_layer == 0 else 36 * 2 ** (num_layer - 1) - 6
            else:
                rf = 3 if num_layer == 0 else 18 * 2 ** (num_layer - 1) - 4

        return int(rf // 2)

    def stop_training(self):
        self.stop_fitting = True

    def dim_order_forward(self, image, batch_dim_bool):
        image_dimension = len(image.shape)
        # permutate dimensions in image to consolidate batch dimensions at the front:
        batch_dim_axes = [i for i in range(0, image_dimension) if batch_dim_bool[i]]
        non_batch_dim_axes = numpy.array(
            [i for i in range(0, image_dimension) if not batch_dim_bool[i]]
        )
        # make sure the smallest size dimension comes to the first in non_batch_dim
        non_batch_img_size = [image.shape[i] for i in non_batch_dim_axes]
        non_batch_dim_axes = list(non_batch_dim_axes[numpy.argsort(non_batch_img_size)])

        self.axes_permutation = batch_dim_axes + non_batch_dim_axes
        image = numpy.transpose(image, axes=self.axes_permutation)
        nb_batch_dim = sum([(1 if i else 0) for i in batch_dim_bool])
        nb_non_batch_dim = image_dimension - nb_batch_dim
        batch_dim_permuted = batch_dim_bool[self.axes_permutation]
        lprint(
            f'Axis permutation for batch-dim consolidation during feature gen: {self.axes_permutation}'
        )
        lprint(
            f'Number of batch dim: {nb_batch_dim}, number of non batch dim: {nb_non_batch_dim}'
        )
        if nb_non_batch_dim == 2:
            lprint('The input image will be processed with 2D CNN.')
        elif nb_non_batch_dim == 3:
            lprint('The input image will be processed with 3D CNN.')
        elif nb_non_batch_dim > 3:
            lprint(
                'The input image will be processed with 3D CNN. If this is not intended, please check batch_dims.'
            )

        # Limit non-batch dims < 3
        self.batch_dim_upto3 = [
            True if i < len(image.shape) - 3 else False for i in range(len(image.shape))
        ]
        self.batch_dim_upto3 = numpy.logical_or(
            self.batch_dim_upto3, batch_dim_permuted
        )

        # Denoised images will be first reshaped to self.image_sahpe_upto3
        self.image_shape_upto3 = image.shape

        # Collapse batch dims
        batch_dim_size = numpy.prod(
            numpy.array(image.shape)[numpy.array(self.batch_dim_upto3)]
        )
        batch_dim_size = 1 if batch_dim_size == 0 else batch_dim_size
        spatial_dim_size = list(
            numpy.array(self.image_shape_upto3)[
                numpy.invert(numpy.array(self.batch_dim_upto3))
            ]
        )
        target_shape = [batch_dim_size] + spatial_dim_size + [1]
        image = image.reshape(target_shape)
        batch_dim_out = [True if i == 0 else False for i in range(len(image.shape))]
        return image, batch_dim_out

    def dim_order_backward(self, image):
        axes_reverse_perm = [0] * len(self.axes_permutation)  # generate a blank list
        # Reverse the order of permutation
        for i, j in enumerate(self.axes_permutation):
            axes_reverse_perm[j] = i

        # Reshape image
        batch_dim_sizes = tuple(
            numpy.array(self.image_shape_upto3)[self.batch_dim_upto3]
        )
        image = image.reshape(batch_dim_sizes + image.shape[1:-1])
        return numpy.transpose(image, axes=axes_reverse_perm)

    def batch2tile_size(self, batch_size, shiftconv=True, floattype=32):
        tile_size = 1024
        for i in range(1024):
            t = [i for _ in range(len(self.input_dim[:-1]))]
            model_size = sim_model_size(t, shiftconv, floattype)
            if model_size * batch_size >= (provider.device.global_mem_size * 0.8):
                tile_size = i - 1
                break
        return tile_size

    def train(
        self,
        input_image,
        target_image,
        batch_dims=None,
        train_valid_ratio=0.1,
        callback_period=3,
    ):
        """

        :param input_image: dimension (batch, H, W, C)
        :param target_image: dimension (batch, H, W, C)
        :param batch_dims:
        :param train_valid_ratio:
        :param callback_period:
        :return:
        """
        # Check for supervised & batch_input
        if input_image is target_image:
            self.supervised = False

        # set default batch_dim value:
        if batch_dims:
            if len(batch_dims) != len(input_image.shape):
                raise ArrayShapeDoesNotMatchError(
                    'The length of batch_dims and input_image dimensions are different.'
                )
        else:
            batch_dims = (False,) * len(input_image.shape)
        # batch_dims is the very first (original to the raw input) dimension indicator.

        # Check if there're dims with size 1.
        batch_dims_size1 = numpy.array(input_image.shape) == 1
        batch_dims_comb = numpy.logical_or(batch_dims, batch_dims_size1)

        # Reshape the input image
        input_image, batch_dims = self.dim_order_forward(input_image, batch_dims_comb)

        self.callback_period = callback_period

        self.checkpoint = None
        self.input_dim = input_image.shape[1:]

        # Compute tile size from batch size
        if self.patch_size is None:
            self.patch_size = self.batch2tile_size(
                self.batch_size,
                shiftconv=True if 'shiftconv' in self.training_architecture else False,
            )
            self.patch_size = min(
                self.patch_size,
                self.get_receptive_field_radius(
                    self.num_layer,
                    shiftconv=True
                    if 'shiftconv' in self.training_architecture
                    else False,
                ),
            )
            if self.patch_size < 2 ** self.num_layer:
                raise ValueError('Number of layers is too large for given patch size.')
            self.patch_size = self.patch_size - self.patch_size % 2 ** self.num_layer
        else:
            if 'shiftconv' in self.training_architecture:
                if numpy.unique(self.patch_size).size != 1:
                    raise ValueError(
                        'shiftconv only accepts patch_size with same length in all dimensions.'
                    )
        lprint(f'Patch size: {self.patch_size}')

        # Check if the patch_size is appropriate for the model
        if type(self.patch_size) == int:
            self.patch_size = [self.patch_size for _ in self.input_dim[:-1]]

        patch_size = numpy.array(self.patch_size)

        if (patch_size.max() / (2 ** self.num_layer) <= 0).any():
            raise ValueError(
                f'Tile size is too small. The largest dimension of tile size has to be >= {2 ** self.num_layer}.'
            )
        if (patch_size[-2:] % 2 ** self.num_layer != 0).any():
            raise ValueError(
                f'Tile sizes on XY plane have to be multiple of 2^{self.num_layer}'
            )

        # Determine total number of patches
        if self.total_num_patches is None:
            self.total_num_patches = max(
                1024, input_image.size / numpy.prod(self.patch_size)
            )  # compare with the input image size
            self.total_num_patches = min(
                self.total_num_patches, 10240
            )  # upper limit of num of patches
            self.total_num_patches = (
                self.total_num_patches
                - (self.total_num_patches % self.batch_size)
                + self.batch_size
            )
        else:
            if self.total_num_patches < self.batch_size:
                raise ValueError('total_num_patches has to be larger than batch_size.')
            self.total_num_patches = (
                self.total_num_patches
                - (self.total_num_patches % self.batch_size)
                + self.batch_size
            )

        self.rot_batch_size = self.batch_size
        lprint("Max mem: ", provider.device.global_mem_size)
        lprint(f"Keras batch size for training: {self.batch_size}")

        # Decide whether to use validation pixels or patches
        if 1024 > input_image.size / numpy.prod(self.patch_size):
            with lsection(
                f'Validation data will be created by monitoring {train_valid_ratio} of the pixels in the input data.'
            ):
                img_train, img_val, val_marker = val_img_generator(
                    input_image, p=train_valid_ratio
                )
        else:
            with lsection(
                f'Validation data will be created by monitoring {train_valid_ratio} of the patches/images in the input data.'
            ):
                self._batch_processing = True

        # Tile input and target image
        if self.patch_size is not None:
            with lsection(f'Random patch sampling...'):
                lprint(f'Total number of patches: {self.total_num_patches}')
                input_patch_idx = random_sample_patches(
                    input_image,
                    self.patch_size,
                    self.total_num_patches,
                    self.adoption_rate,
                )

                img_train_patch = []

                if self._batch_processing:
                    for i in input_patch_idx:
                        img_train_patch.append(input_image[i])
                    img_train = numpy.vstack(img_train_patch)
                else:
                    img_val_patch = []
                    marker_patch = []
                    for i in input_patch_idx:
                        img_train_patch.append(img_train[i])
                        img_val_patch.append(img_val[i])
                        marker_patch.append(val_marker[i])
                    img_train = numpy.vstack(img_train_patch)
                    img_val = numpy.vstack(img_val_patch)
                    val_marker = numpy.vstack(marker_patch)
                    self.img_val = img_val
                    self.val_marker = val_marker

                if self.supervised:
                    target_patch = []
                    for i in input_patch_idx:
                        target_patch.append(target_image[i])
                    target_image = numpy.vstack(target_patch)
                else:
                    target_image = img_train

        # Check if it is self-supervised
        if self.supervised:
            lprint('Model will be created for supervised learning.')
            lprint(
                'Shift convolution will be turned off automatically because supervised learning was selected.'
            )
        elif 'shiftconv' == self.training_architecture and self.supervised is False:
            # TODO: Hirofumi what is going on the conditional below :D  <-- check input dim is compatible w/ shiftconv
            if (
                numpy.mod(
                    img_train.shape[1:][:-1],
                    numpy.repeat(2 ** self.num_layer, len(img_train.shape[1:][:-1])),
                )
                != 0
            ).any():
                raise ValueError(
                    'Each dimension of the input image has to be a multiple of 2^num_layer for shiftconv.'
                )
            lprint(
                'Model will be generated for self-supervised learning with shift convolution scheme.'
            )
            if numpy.diff(img_train.shape[1:][:2]) != 0:
                raise ValueError(
                    'Make sure the input image shape is cubic as shiftconv mode involves rotation.'
                )
            if (
                numpy.mod(
                    img_train.shape[1:][:-1],
                    numpy.repeat(
                        2 ** (self.num_layer - 1), len(img_train.shape[1:][:-1])
                    ),
                )
                != 0
            ).any():
                raise ValueError(
                    'Each dimension of the input image has to be a multiple of '
                    '2^(num_layer-1) as shiftconv mode involvs pixel shift. '
                )
        else:
            lprint(
                'Model will be generated for self-supervised with moving-blind spot scheme.'
            )

        if len(self.patch_size) == 2:
            self.model = UNet2DModel(
                img_train.shape[1:],
                rot_batch_size=self.rot_batch_size,
                num_lyr=self.num_layer,
                initial_unit=self.initial_units,
                normalization=self.batch_norm,
                activation=self.activation_fun,
                supervised=self.supervised,
                shiftconv=True
                if 'shiftconv' == self.training_architecture
                and self.supervised is False
                else False,
                weight_decay=self.weight_decay,
                learning_rate=self.learn_rate,
                residual=self.use_residual,
                pooling_mode=self.pooling_mode,
                interpolation=self.interpolation,
            )
        elif len(self.patch_size) == 3:
            self.model = Unet3DModel(
                img_train.shape[1:],
                initial_unit=self.initial_units,
                rot_batch_size=self.rot_batch_size,  # img_train.shape[0],
                num_lyr=self.num_layer,
                normalization=self.batch_norm,
                activation=self.activation_fun,
                supervised=self.supervised,
                shiftconv=True
                if 'shiftconv' == self.training_architecture
                and self.supervised is False
                else False,
                weight_decay=self.weight_decay,
                learning_rate=self.learn_rate,
                residual=self.use_residual,
                pooling_mode=self.pooling_mode,
            )
        with lsection(f'CNN model summary:'):
            lprint(f'Number of parameters in the model: {self.model.count_params()}')
            lprint(f'Number of layers: {self.num_layer}')
            lprint(f'Batch normalization: {self.batch_norm}')
            lprint(f'Train scheme: {self.training_architecture}')
            lprint(f'Training input size: {img_train.shape[1:]}')

        super().train(
            img_train,
            target_image,
            batch_dims=batch_dims,
            callback_period=callback_period,
        )

    def retrain(self, input_image, target_image, training_architecture=None):
        self.training_architecture = training_architecture

        self.train(
            input_image, target_image,
        )

    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        train_valid_ratio=0.1,
        callback_period=3,
    ):

        with lsection(
            f"Training image translator from image of shape {input_image.shape} to image of shape {target_image.shape}:"
        ):

            # Resize for monitoring images:
            if self.monitor is not None and self.monitor.monitoring_images is not None:
                # Normalise monitoring images:
                normalised_monitoring_images = [
                    self.input_normaliser.normalise(monitoring_image)
                    for monitoring_image in self.monitor.monitoring_images
                ]
                monitoring_images_patches = [
                    numpy.expand_dims(numpy.expand_dims(monitoring_image, 0), -1)
                    for monitoring_image in normalised_monitoring_images
                ]
            else:
                monitoring_images_patches = None

            # We keep these features handy...
            self.monitoring_datasets = monitoring_images_patches

            if self.EStop_patience is None:
                self.EStop_patience = self.patience * 2
            if self.ReduceLR_patience is None:
                self.ReduceLR_patience = self.patience

            # Callback for monitoring:
            def monitor_callback(iteration, val_loss):

                if val_loss is None:
                    return

                current_time_sec = time.time()

                if (
                    current_time_sec
                    > self.last_callback_time_sec + self.callback_period
                ):

                    if self.monitoring_datasets and self.monitor:

                        predicted_monitoring_datasets = [
                            self.translate(
                                x_m, tile_size=[64 for _ in self.input_dim[:-1]]
                            )
                            for x_m in self.monitoring_datasets
                        ]
                        inferred_images = [
                            y_m.reshape(self.patch_size)
                            for (image, y_m) in zip(
                                self.monitor.monitoring_images,
                                predicted_monitoring_datasets,
                            )
                        ]

                        denormalised_inferred_images = [
                            self.target_normaliser.denormalise(inferred_image)
                            for inferred_image in inferred_images
                        ]

                        self.monitor.variables = (
                            iteration,
                            val_loss,
                            denormalised_inferred_images,
                        )
                    elif self.monitor:
                        self.monitor.variables = (iteration, val_loss, None)

                    self.last_callback_time_sec = current_time_sec

            # Early stopping patience:
            lprint(f"Early stopping patience: {self.EStop_patience}")

            # Effective LR patience:
            lprint(f"Effective LR patience: {self.ReduceLR_patience}")
            lprint(f'Batch size: {self.batch_size}')

            # Here is the list of callbacks:
            callbacks = []

            # Set upstream callback:
            self.keras_callback = CNNCallback(monitor_callback)

            # Early stopping callback:
            self.early_stopping = EarlyStopping(
                self,
                monitor='loss' if self.supervised else 'val_loss',
                min_delta=self.min_delta,
                patience=self.EStop_patience,
                mode='auto',
                restore_best_weights=True,
            )
            self.early_stopping1 = EarlyStopping(
                self,
                monitor='loss' if self.supervised else 'val_loss',
                min_delta=self.min_delta,
                patience=self.EStop_patience,
                mode='auto',
                restore_best_weights=True,
            )

            # Reduce LR on plateau:
            self.reduce_learning_rate = ReduceLROnPlateau(
                monitor='loss' if self.supervised else 'val_loss',
                min_delta=self.min_delta,
                factor=0.1,
                verbose=1,
                patience=self.ReduceLR_patience,
                mode='auto',
                min_lr=1e-8,
            )
            self.reduce_learning_rate1 = ReduceLROnPlateau(
                monitor='loss' if self.supervised else 'val_loss',
                min_delta=self.min_delta,
                factor=self.reduce_lr_factor,
                verbose=1,
                patience=self.ReduceLR_patience,
                mode='auto',
                min_lr=self.learn_rate * self.reduce_lr_factor ** 2,
            )

            if self.checkpoint is None:
                self.model_file_path = join(
                    get_temp_folder(),
                    f"aydin_cnn_keras_model_file_{random.randint(0, 1e16)}.hdf5",
                )
                lprint(f"Model will be saved at: {self.model_file_path}")
                self.checkpoint = ModelCheckpoint(
                    self.model_file_path,
                    monitor='loss' if self.supervised else 'val_loss',
                    verbose=1,
                    save_best_only=True,
                )
                # Add callbacks to the list:
                callbacks.append(self.keras_callback)
                callbacks.append(self.checkpoint)
                callbacks.append(self.early_stopping)
                callbacks.append(self.reduce_learning_rate)

            if self._batch_processing:
                tv_ratio = train_valid_ratio
            else:
                tv_ratio = 0

            lprint("Training now...")
            if self.supervised:
                self.model.fit(
                    input_image,
                    target_image,
                    batch_size=self.batch_size,
                    epochs=self.max_epochs,
                    callbacks=callbacks,
                    verbose=self.verbose,
                )
            elif 'shiftconv' in self.training_architecture:
                steps_per_epoch = numpy.ceil(
                    self.total_num_patches / self.batch_size
                ).astype(int)
                validation_data = [input_image, self.img_val]

                if self._batch_processing:
                    val_split = self.total_num_patches * train_valid_ratio
                    val_split = (
                        val_split - (val_split % self.batch_size) + self.batch_size
                    ) / self.total_num_patches
                    steps_per_epoch = numpy.ceil(
                        self.total_num_patches * val_split
                    ).astype(int)
                    validation_data = [
                        input_image[int(self.total_num_patches * val_split) :],
                        input_image[int(self.total_num_patches * val_split) :],
                    ]

                self.model.fit(
                    input_image,
                    input_image,
                    epochs=self.max_epochs,
                    callbacks=callbacks,
                    verbose=self.verbose,
                    steps_per_epoch=steps_per_epoch,
                    batch_size=self.batch_size,
                    validation_data=validation_data,
                )
            elif 'checkerbox' in self.training_architecture:
                self.model.fit(
                    maskedgen(
                        self.patch_size,
                        self.mask_shape,
                        input_image,
                        self.batch_size,
                        train_valid_ratio=tv_ratio,
                        replace_by=self.replace_by,
                        subset='training',
                    ),
                    epochs=self.max_epochs,
                    steps_per_epoch=numpy.prod(self.mask_shape)
                    * numpy.ceil(
                        input_image.shape[0] * (1 - tv_ratio) / self.batch_size
                    ).astype(int),
                    verbose=self.verbose,
                    callbacks=callbacks,
                    validation_data=maskedgen(
                        self.patch_size,
                        self.mask_shape,
                        input_image,
                        self.batch_size,
                        train_valid_ratio=train_valid_ratio,
                        subset='validation',
                        replace_by=self.replace_by,
                    )
                    if self._batch_processing
                    else val_data_generator(
                        input_image,
                        self.img_val,
                        self.val_marker,
                        self.batch_size,
                        train_valid_ratio=train_valid_ratio,
                    ),
                    validation_steps=numpy.floor(
                        input_image.shape[0] * train_valid_ratio / self.batch_size
                    ).astype(int),
                )
            elif 'random' in self.training_architecture:
                self.model.fit(
                    randmaskgen(
                        input_image,
                        self.batch_size,
                        p=self.random_mask_ratio,
                        train_valid_ratio=tv_ratio,
                        replace_by=self.replace_by,
                        subset='training',
                    ),
                    epochs=self.max_epochs,
                    steps_per_epoch=numpy.ceil(
                        self.total_num_patches * (1 - tv_ratio) / self.batch_size
                    ).astype(int),
                    verbose=self.verbose,
                    callbacks=callbacks,
                    validation_data=randmaskgen(
                        input_image,
                        self.batch_size,
                        p=self.random_mask_ratio,
                        train_valid_ratio=train_valid_ratio,
                        subset='validation',
                        replace_by=self.replace_by,
                    )
                    if self._batch_processing
                    else val_data_generator(
                        input_image,
                        self.img_val,
                        self.val_marker,
                        self.batch_size,
                        train_valid_ratio=train_valid_ratio,
                    ),
                    validation_steps=numpy.floor(
                        self.total_num_patches * train_valid_ratio / self.batch_size
                    ).astype(int),
                )
            elif 'checkran' in self.training_architecture:
                # train with checkerbox first
                lprint('Starting with checkerbox masking.')
                self.history = self.model.fit(
                    maskedgen(
                        self.patch_size,
                        self.mask_shape,
                        input_image,
                        self.batch_size,
                        train_valid_ratio=tv_ratio,
                        subset='training',
                        replace_by=self.replace_by,
                    ),
                    epochs=self.max_epochs,
                    steps_per_epoch=numpy.prod(self.mask_shape)
                    * numpy.ceil(
                        input_image.shape[0] * (1 - tv_ratio) / self.batch_size
                    ).astype(int),
                    verbose=self.verbose,
                    callbacks=callbacks[:2]
                    + [self.early_stopping1, self.reduce_learning_rate1],
                    validation_data=maskedgen(
                        self.patch_size,
                        self.mask_shape,
                        input_image,
                        self.batch_size,
                        train_valid_ratio=train_valid_ratio,
                        replace_by=self.replace_by,
                        subset='validation',
                    )
                    if self._batch_processing
                    else val_data_generator(
                        input_image,
                        self.img_val,
                        self.val_marker,
                        self.batch_size,
                        train_valid_ratio=train_valid_ratio,
                    ),
                    validation_steps=numpy.ceil(
                        input_image.shape[0] * train_valid_ratio / self.batch_size
                    ).astype(int),
                )

                # Then switch to random masking
                lprint('Switched to random masking.')
                self.model.fit(
                    randmaskgen(
                        input_image,
                        self.batch_size,
                        p=self.random_mask_ratio,
                        train_valid_ratio=tv_ratio,
                        replace_by=self.replace_by,
                        subset='training',
                    ),
                    epochs=self.max_epochs,
                    steps_per_epoch=numpy.ceil(
                        self.total_num_patches * (1 - tv_ratio) / self.batch_size
                    ).astype(int),
                    verbose=self.verbose,
                    callbacks=callbacks,
                    initial_epoch=self.history.epoch[-1] + 1,
                    validation_data=randmaskgen(
                        input_image,
                        self.batch_size,
                        p=self.random_mask_ratio,
                        train_valid_ratio=train_valid_ratio,
                        subset='validation',
                        replace_by=self.replace_by,
                    )
                    if self._batch_processing
                    else val_data_generator(
                        input_image,
                        self.img_val,
                        self.val_marker,
                        self.batch_size,
                        train_valid_ratio=train_valid_ratio,
                    ),
                    validation_steps=numpy.ceil(
                        self.total_num_patches * train_valid_ratio / self.batch_size
                    ).astype(int),
                )

    def translate(
        self,
        input_image,
        translated_image=None,
        batch_dims=None,
        tile_size=None,
        max_margin=32,
    ):
        if batch_dims:
            if len(batch_dims) != len(input_image.shape):
                raise ValueError(
                    'The length of batch_dims and that of input image dimensions are different.'
                )
        else:
            batch_dims = (False,) * len(input_image.shape)
        # batch_dims is the very first (original to the raw input) dimension indicator.

        # Check if there're dims with size 1.
        batch_dims_size1 = numpy.array(input_image.shape) == 1
        batch_dims_comb = numpy.logical_or(batch_dims, batch_dims_size1)

        # Reshape the input image
        input_image, batch_dims = self.dim_order_forward(input_image, batch_dims_comb)
        non_batch_dims = numpy.array(input_image.shape)[numpy.invert(batch_dims)]

        if numpy.array(tile_size == numpy.unique(non_batch_dims[:-1])).all():
            tile_size = None

        result_image = numpy.zeros(input_image.shape, dtype=numpy.float32)
        for batch_index in range(input_image.shape[0]):
            result_image[batch_index] = super().translate(
                input_image[batch_index : batch_index + 1],
                translated_image,
                batch_dims=batch_dims,
                tile_size=tile_size,
            )

        # Convert the dimensions back to the original
        output_image = self.dim_order_backward(result_image)

        return output_image

    def _translate(self, input_image, batch_dim=None):
        reshaped_for_cube = False
        reshaped_for_model = False
        input_shape = numpy.array(input_image.shape[1:-1])
        if abs(numpy.diff(input_shape)).min() != 0:
            reshaped_for_cube = True
            input_shape_max = numpy.ones(input_shape.shape) * input_shape.max()
            pad_square = (input_shape_max - input_shape) / 2
            pad_width1 = (
                [[0, 0]]
                + [
                    [
                        numpy.ceil(pad_square[i]).astype(int),
                        numpy.floor(pad_square[i]).astype(int),
                    ]
                    for i in range(len(pad_square))
                ]
                + [[0, 0]]
            )
            input_image = numpy.pad(input_image, pad_width1, 'edge')
            input_shape = numpy.array(input_image.shape[1:-1])

        if not (input_shape % 2 ** self.num_layer == 0).all():
            reshaped_for_model = True
            pad_width0 = (
                2 ** self.num_layer
                - (input_shape % 2 ** self.num_layer)
                # + pad_square
            ) / 2
            pad_width2 = (
                [[0, 0]]
                + [
                    [
                        numpy.ceil(pad_width0[i]).astype(int),
                        numpy.floor(pad_width0[i]).astype(int),
                    ]
                    for i in range(len(pad_width0))
                ]
                + [[0, 0]]
            )
            input_image = numpy.pad(input_image, pad_width2, 'edge')

        # Change the batch_size in split layer or input dimensions accordingly
        if self.infmodel is None:
            if len(input_image.shape[1:-1]) == 2:
                self.infmodel = UNet2DModel(
                    [None, None, input_image.shape[-1]],
                    rot_batch_size=1,  # input_image.shape[0],
                    num_lyr=self.num_layer,
                    initial_unit=self.initial_units,
                    normalization=self.batch_norm,
                    activation=self.activation_fun,
                    supervised=True
                    if 'random' in self.training_architecture
                    or 'check' in self.training_architecture
                    else self.supervised,
                    shiftconv=True
                    if 'shiftconv' in self.training_architecture
                    else False,
                )
            elif len(input_image.shape[1:-1]) == 3:
                self.infmodel = Unet3DModel(
                    [None, None, None, input_image.shape[-1]],
                    rot_batch_size=1,  # input_image.shape[0],
                    num_lyr=self.num_layer,
                    initial_unit=self.initial_units,
                    normalization=self.batch_norm,
                    activation=self.activation_fun,
                    supervised=True
                    if 'random' in self.training_architecture
                    or 'check' in self.training_architecture
                    else self.supervised,
                    shiftconv=True
                    if 'shiftconv' in self.training_architecture
                    else False,
                    original_zdim=self.input_dim[1],
                )
            self.infmodel.set_weights(self.model.get_weights())

        if 'shiftconv' in self.training_architecture:
            output_image = self.infmodel.predict(
                input_image, batch_size=1, verbose=self.verbose
            )
        else:
            try:
                output_image = self.infmodel.predict(
                    input_image, batch_size=self.batch_size, verbose=self.verbose
                )
            except Exception:
                output_image = self.infmodel.predict(
                    input_image, batch_size=1, verbose=self.verbose
                )

        if reshaped_for_model:
            if len(input_shape) == 2:
                output_image = output_image[
                    0:1,
                    pad_width2[1][0] : -pad_width2[1][1] or None,
                    pad_width2[2][0] : -pad_width2[2][1] or None,
                    :,
                ]
            else:
                output_image = output_image[
                    0:1,
                    pad_width2[1][0] : -pad_width2[1][1] or None,
                    pad_width2[2][0] : -pad_width2[2][1] or None,
                    pad_width2[3][0] : -pad_width2[3][1] or None,
                    :,
                ]
        if reshaped_for_cube:
            if len(input_shape) == 2:
                output_image = output_image[
                    0:1,
                    pad_width1[1][0] : -pad_width1[1][1] or None,
                    pad_width1[2][0] : -pad_width1[2][1] or None,
                    :,
                ]
            else:
                output_image = output_image[
                    0:1,
                    pad_width1[1][0] : -pad_width1[1][1] or None,
                    pad_width1[2][0] : -pad_width1[2][1] or None,
                    pad_width1[3][0] : -pad_width1[3][1] or None,
                    :,
                ]

        return output_image

    # TODO: modify these functions to generate tiles with the same size of tile_size
    def _get_tilling_strategy(self, batch_dims, tile_size, shape):

        # We will store the batch strategy as a list of integers representing the number of chunks per dimension:
        with lsection(f"Determine tilling strategy:"):

            lprint(f"shape                   = {shape}")
            lprint(f"batch_dims              = {batch_dims}")
            lprint(f"tile_size               = {tile_size}")

            # This is the tile strategy, essentially how to split each dimension...
            tilling_strategy = tuple(
                (1 if is_batch else max(1, int(math.ceil(dim / tile_size))))
                for dim, is_batch in zip(shape, batch_dims)
            )
            lprint(f"Tilling strategy is: {tilling_strategy}")

            return tilling_strategy

    def _get_margins(self, shape, tilling_strategy, max_margin):

        # Receptive field:
        receptive_field_radius = self.get_receptive_field_radius(len(shape))

        # We compute the margin from the receptive field but limit it to 33% of the tile size:
        margins = (receptive_field_radius,) * len(shape)

        # We only need margins if we split a dimension:
        margins = tuple(
            (0 if split == 1 else margin)
            for margin, split in zip(margins, tilling_strategy)
        )
        return margins
