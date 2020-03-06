import time
import math
from os.path import join

import numpy
import random
from aydin.providers.opencl.opencl_provider import OpenCLProvider

provider = OpenCLProvider()

import tensorflow as tf  # noqa: E402
from aydin.io.folders import get_temp_folder  # noqa: E402
from aydin.it.cnn.models.unet_2d import unet_2d_model  # noqa: E402
from aydin.it.cnn.models.unet_3d import unet_3d_model  # noqa: E402
from aydin.it.it_base import ImageTranslatorBase  # noqa: E402
from aydin.util.log.log import lsection, lprint  # noqa: E402
from aydin.it.cnn.layers import split  # noqa: E402
from aydin.it.cnn.layers.masking import Maskout, maskedgen, randmaskgen  # noqa: E402
from aydin.it.cnn.layers.layers import rot90  # noqa: E402
from aydin.regression.nn_utils.callbacks import ModelCheckpoint  # noqa: E402
from aydin.it.cnn.util.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    CNNCallback,
)  # noqa: E402
from aydin.it.cnn.util.receptive_field import receptive_field_model  # noqa: E402

# from tensorflow.keras.models import model_from_json
from aydin.it.cnn.util.data_util import (
    random_sample_patches,
    sim_model_size,
)  # noqa: E402
from aydin.it.cnn.util.val_generator import (
    val_img_generator,
    val_data_generator,
)  # noqa: E402

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


class ImageTranslatorCNN(ImageTranslatorBase):
    """
        aydin CNN Image Translator

        Using CNN (Unet and Co)

    """

    def __init__(
        self,
        normaliser_type: str = 'percentile',
        monitor=None,
        training_architecture=None,  # 'shiftconv' or 'random' or 'checkerbox'
        batch_size=32,
        num_layer=5,
        batch_norm=None,
        tile_size=None,
        total_num_patches=None,
        adoption_rate=0.5,
        mask_shape=(9, 9),
        EStop_patience=None,
        ReduceLR_patience=None,
        min_delta=0,
        mask_p=0.1,
        verbose=0,
        max_epochs=1024,
        patience=4,
    ):
        """
        :param monitor: an object for monitoring training (only for GUI)
        :param training_architecture: 'shiftconv' or 'checkerbox' or 'randommasking' architecture
        :param batch_size: batch size for training
        :param num_layer: number of layers
        :param batch_norm; type of batch normalization (e.g. batch, instance)
        :param tile_size: tile size for patch sample e.g. 64 or (64, 64)
        :param total_num_patches: total number of patches for training
        :param adoption_rate: % of random patches will be used for training, the rest will be discarded
        :param mask_shape: mask shape for masking architecture
        :param EStop_patience: patience for early stopping
        :param ReduceLR_patience: patience for reduced learn rate
        :param min_delta: 1e-6; minimum delta to determine if loss is dropping
        :param mask_p: possibility of masked pixels in ramdom masking approach
        :param verbose: print out training status from keras
        :param max_epochs: maximum epoches
        :param patience: patience for EarlyStop or ReducedLR to be triggered
        """
        super().__init__(normaliser_type, monitor)
        self.model = None  # a CNN model
        self.infmodel = None  # inference model
        self.supervised = True
        self.training_architecture = training_architecture
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.batch_norm = batch_norm
        self.tile_size = tile_size
        self.total_num_patches = total_num_patches
        self.adoption_rate = adoption_rate
        self.mask_shape = mask_shape
        self.EStop_patience = EStop_patience
        self.ReduceLR_patience = ReduceLR_patience
        self.min_delta = min_delta
        self.p = mask_p
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.patience = patience

        self.axes_permutation = None
        self.image_shape_upto3 = None
        self.callback_period = None
        self.checkpoint = None
        self.input_dim = None
        self.stop_fitting = False
        self.rot_batch_size = None
        self.img_val = None
        self.val_marker = None
        self.batch_processing = (
            False  # whether validation data is chosen from pixels or slices
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
        # super().save(path)
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

    # TODO: We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        # exclude fields below that should/cannot be saved properly:
        del state['early_stopping']
        del state['reduce_learning_rate']
        del state['checkpoint']
        del state['model']
        # del state['input_normaliser']
        # del state['target_normaliser']
        return state

    # TODO: SPECIAL please fix the naming of the file for the weights here and in nn.

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
        if shiftconv:
            rf = [6, 22, 54, 110, 222, 446, 894, 1790, 3582, 7166]
        else:
            rf = [3, 10, 22, 46, 94, 190, 382, 766, 1534, 3070]
        if self.model is None:
            r = rf[num_layer]
        else:
            r = receptive_field_model(self.model)
        # r = numpy.sqrt(r)  # effective receptive field
        return int(r // 2)

    def limit_epochs(self, max_epochs: int) -> int:
        return max_epochs

    def stop_training(self):
        self.stop_fitting = True

    def dim_order_forward(self, image, batch_dim_bool):
        image_dimension = len(image.shape)
        # permutate dimensions in image to consolidate batch dimensions at the front:
        batch_dim_axes = [i for i in range(0, image_dimension) if batch_dim_bool[i]]
        non_batch_dim_axes = [
            i for i in range(0, image_dimension) if not batch_dim_bool[i]
        ]
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
        batch_dim_upto3 = [
            True if i < len(image.shape) - 3 else False for i in range(len(image.shape))
        ]
        batch_dim_upto3 = numpy.logical_or(batch_dim_upto3, batch_dim_permuted)
        self.image_shape_upto3 = image.shape
        # Denoised images will be first reshaped to self.image_sahpe_upto3

        # Collapse batch dims
        batch_dim_size = numpy.prod(
            numpy.array(image.shape)[numpy.array(batch_dim_upto3)]
        )
        batch_dim_size = 1 if batch_dim_size == 0 else batch_dim_size
        target_shape = (
            [batch_dim_size]
            + list(numpy.array(self.image_shape_upto3)[~numpy.array(batch_dim_upto3)])
            + [1]
        )
        image = image.reshape(target_shape)
        batch_dim_out = [True if i == 0 else False for i in range(len(image.shape))]
        return image, batch_dim_out

    def dim_order_backward(self, image):
        # Reshape image
        image = image.reshape(self.image_shape_upto3)
        return numpy.transpose(image, axes=self.axes_permutation)

    def batch2tile_size(self, batch_size, shiftconv=True, floattype=32):
        tile_size = 1024
        for i in range(1024):
            t = [i for _ in range(len(self.input_dim[:-1]))]
            model_size = sim_model_size(t, shiftconv, floattype)
            if model_size * batch_size >= (provider.device.global_mem_size * 0.8):
                tile_size = i - 1
                break
        return tile_size

    def datagen(
        self, image_train, image_val, batch_size, train_valid_ratio, subset='training'
    ):
        datagen_train = ImageDataGenerator(validation_split=train_valid_ratio).flow(
            image_train, batch_size=batch_size, subset='training'
        )
        datagen_val = ImageDataGenerator(validation_split=train_valid_ratio).flow(
            image_val, batch_size=batch_size, subset='validation'
        )

        while True:
            if 'train' in subset:
                train_batch = datagen_train.next()
                yield train_batch, train_batch
            else:
                train_batch = datagen_train.next()
                val_batch = datagen_val.next()
                yield train_batch, val_batch

    def train(
        self,
        input_image,  # dimension (batch, H, W, C)
        target_image,  # dimension (batch, H, W, C)
        batch_dims=None,
        train_valid_ratio=0.1,
        callback_period=3,
    ):
        # Check for supervised & batch_input
        if input_image is target_image:
            self.supervised = False

        # set default batch_dim value:
        image_dimension = len(input_image.shape)
        if batch_dims:
            assert len(batch_dims) == image_dimension, (
                'The size of batch_dims and that of input image dimensions are different. '
                'Please make sure them are same.'
            )
        else:
            batch_dims = (False,) * image_dimension
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
        if self.tile_size is None:
            self.tile_size = self.batch2tile_size(
                self.batch_size,
                shiftconv=True if 'shiftconv' in self.training_architecture else False,
            )
            self.tile_size = min(
                self.tile_size,
                self.get_receptive_field_radius(
                    self.num_layer,
                    shiftconv=True
                    if 'shiftconv' in self.training_architecture
                    else False,
                ),
            )
            assert (
                self.tile_size >= 2 ** self.num_layer
            ), 'Number of layers is too large.'
            self.tile_size = self.tile_size - self.tile_size % 2 ** self.num_layer
        else:
            assert numpy.unique(self.tile_size).size == 1, (
                'Tiles with different length in each dim are currently not accepted.'
                'Please make tile_size with the same length in all dimensions. '
            )
        lprint(f'Patch size: {self.tile_size}')

        # Check if the tile_size is appropriate for the model
        if type(self.tile_size) == int:
            self.tile_size = [self.tile_size for _ in self.input_dim[:-1]]

        tile_size = numpy.array(self.tile_size)
        assert (
            tile_size / 2 ** self.num_layer > 0
        ).any(), f'Sampling patch size is too small. Patch size has to be >= {2 ** self.num_layer}. Change number of layers of patch size.'
        assert (
            tile_size % 2 ** self.num_layer == 0
        ).any(), f'Sampling patch size has to be multiple of 2^{self.num_layer}'

        # Determine total number of patches
        if self.total_num_patches is None:
            self.total_num_patches = max(
                1024, input_image.size / numpy.prod(self.tile_size)
            )  # compare with the input image size
            self.total_num_patches = min(
                self.total_num_patches, 10240
            )  # upper limit of num of patches
            self.total_num_patches = numpy.lcm(self.total_num_patches, self.batch_size)
        else:
            assert (
                self.total_num_patches >= self.batch_size
            ), 'total_num_patches has to be larger than batch_size. Please make a larger value for total_num_patches.'
            self.total_num_patches = (
                self.total_num_patches
                - (self.total_num_patches % self.batch_size)
                + self.batch_size
            )

        self.rot_batch_size = self.batch_size
        lprint("Max mem: ", provider.device.global_mem_size)
        lprint(f"Keras batch size for training: {self.batch_size}")

        # Determine how to create validation data
        if 1024 > input_image.size / numpy.prod(self.tile_size):
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
                self.batch_processing = True

        # Tile input and target image
        if self.tile_size is not None:
            with lsection(f'Random patch sampling...'):
                lprint(f'Total number of patches: {self.total_num_patches}')
                input_patch_idx = random_sample_patches(
                    input_image,
                    self.tile_size,
                    self.total_num_patches,
                    self.adoption_rate,
                )

                if self.batch_processing:
                    img_train_patch = []
                    for i in input_patch_idx:
                        img_train_patch.append(img_train[i])
                    img_train = numpy.vstack(img_train_patch)
                else:
                    img_train_patch = []
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

        if len(self.tile_size) == 2:
            self.model = unet_2d_model(
                img_train.shape[1:],
                rot_batch_size=self.rot_batch_size,  # img_train.shape[0],
                num_lyr=self.num_layer,
                normalization=self.batch_norm,
                supervised=self.supervised,
                shiftconv=True if 'shiftconv' in self.training_architecture else False,
            )
        elif len(self.tile_size) == 3:
            self.model = unet_3d_model(
                img_train.shape[1:],
                rot_batch_size=self.rot_batch_size,  # img_train.shape[0],
                num_lyr=self.num_layer,
                normalization=self.batch_norm,
                supervised=self.supervised,
                shiftconv=True if 'shiftconv' in self.training_architecture else False,
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
                            y_m.reshape(self.tile_size)
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
            early_stopping_patience = self.EStop_patience
            lprint(f"Early stopping patience: {early_stopping_patience}")

            # Effective LR patience:
            effective_lr_patience = self.ReduceLR_patience
            lprint(f"Effective LR patience: {effective_lr_patience}")
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
                patience=early_stopping_patience,
                mode='auto',
                restore_best_weights=True,
            )

            # Reduce LR on plateau:
            self.reduce_learning_rate = ReduceLROnPlateau(
                monitor='loss' if self.supervised else 'val_loss',
                min_delta=self.min_delta,
                factor=0.1,
                verbose=1,
                patience=effective_lr_patience,
                mode='auto',
                min_lr=1e-8,
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
                callbacks.append(self.early_stopping)
                callbacks.append(self.reduce_learning_rate)
                callbacks.append(self.checkpoint)

            if self.batch_processing:
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
                if self.batch_processing:
                    val_split = self.total_num_patches * train_valid_ratio
                    val_split = (
                        val_split - (val_split % self.batch_size) + self.batch_size
                    ) / self.total_num_patches
                    self.model.fit(
                        input_image,
                        input_image,
                        batch_size=self.batch_size,
                        epochs=self.max_epochs,
                        callbacks=callbacks,
                        verbose=self.verbose,
                        validation_split=val_split,
                    )
                else:
                    self.model.fit(
                        self.datagen(
                            input_image,
                            input_image,
                            batch_size=self.batch_size,
                            train_valid_ratio=tv_ratio,
                            subset='training',
                        ),
                        epochs=self.max_epochs,
                        callbacks=callbacks,
                        verbose=self.verbose,
                        steps_per_epoch=numpy.ceil(
                            self.total_num_patches * (1 - tv_ratio) / self.batch_size
                        ).astype(int),
                        validation_data=self.datagen(
                            input_image,
                            self.img_val,
                            batch_size=self.batch_size,
                            train_valid_ratio=0,
                            subset='training',
                        ),
                        validation_steps=numpy.ceil(
                            self.total_num_patches * train_valid_ratio / self.batch_size
                        ).astype(int),
                    )
            elif 'checkerbox' in self.training_architecture:
                self.model.fit(
                    maskedgen(
                        self.tile_size,
                        self.mask_shape,
                        input_image,
                        self.batch_size,
                        train_valid_ratio=tv_ratio,
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
                        self.tile_size,
                        self.mask_shape,
                        input_image,
                        self.batch_size,
                        train_valid_ratio=train_valid_ratio,
                        subset='validation',
                    )
                    if self.batch_processing
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
                        p=train_valid_ratio,
                        train_valid_ratio=tv_ratio,
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
                        p=train_valid_ratio,
                        train_valid_ratio=train_valid_ratio,
                        subset='validation',
                    )
                    if self.batch_processing
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

    def translate(
        self, input_image, translated_image=None, batch_dims=None, tile_size=None
    ):
        if batch_dims:
            assert len(batch_dims) == len(input_image.shape), (
                'The size of batch_dims and that of input image dimensions are different. '
                'Please make sure them are same.'
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

        if (tile_size == numpy.unique(non_batch_dims[:-1])).all():
            tile_size = None
        elif tile_size is None:
            tile_size = self.tile_size

        return super().translate(
            input_image, translated_image, batch_dims=batch_dims, tile_size=tile_size,
        )

    def _translate(self, input_image, batch_dim=None):
        input_shape = numpy.array(input_image.shape[1:-1])
        if abs(numpy.diff(input_shape)).min() != 0:
            input_shape_max = numpy.ones(input_shape.shape) * input_shape.max()
            pad_square = (input_shape_max - input_shape) / 2
            pad_width = (
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
            input_image = numpy.pad(input_image, pad_width, 'constant')
            input_shape = numpy.array(input_image.shape[1:-1])

        if not (input_shape.max() % 2 ** self.num_layer == 0).all():
            pad_width0 = (
                2 ** self.num_layer
                - (input_shape % 2 ** self.num_layer)
                # + pad_square
            ) / 2
            pad_width = (
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
            input_image = numpy.pad(input_image, pad_width, 'constant')

        # Change the batch_size in split layer or input dimensions accordingly
        if self.infmodel is None:
            if len(input_image.shape[1:-1]) == 2:
                self.infmodel = unet_2d_model(
                    [None, None, input_image.shape[-1]],
                    rot_batch_size=input_image.shape[0],
                    num_lyr=self.num_layer,
                    normalization=self.batch_norm,
                    supervised=True
                    if 'random' in self.training_architecture
                    or 'check' in self.training_architecture
                    else self.supervised,
                    shiftconv=True
                    if 'shiftconv' in self.training_architecture
                    else False,
                )
            elif len(input_image.shape[1:-1]) == 3:
                self.infmodel = unet_3d_model(
                    [None, None, None, input_image.shape[-1]],
                    rot_batch_size=input_image.shape[0],
                    num_lyr=self.num_layer,
                    normalization=self.batch_norm,
                    supervised=True
                    if 'random' in self.training_architecture
                    or 'check' in self.training_architecture
                    else self.supervised,
                    shiftconv=True
                    if 'shiftconv' in self.training_architecture
                    else False,
                )
            self.infmodel.set_weights(self.model.get_weights())

        if 'shiftconv' in self.training_architecture:
            output_image = self.infmodel.predict(
                input_image, batch_size=input_image.shape[0], verbose=self.verbose
            )
        else:
            output_image = self.infmodel.predict(
                input_image, batch_size=self.batch_size, verbose=self.verbose
            )

        if not (input_shape % 2 ** self.num_layer == 0).all():
            if len(input_shape) == 2:
                output_image = output_image[
                    0:1,
                    pad_width[1][0] : -pad_width[1][1] or None,
                    pad_width[2][0] : -pad_width[2][1] or None,
                    :,
                ]
            else:
                output_image = output_image[
                    0:1,
                    pad_width[1][0] : -pad_width[1][1] or None,
                    pad_width[2][0] : -pad_width[2][1] or None,
                    pad_width[3][0] : -pad_width[3][1] or None,
                    :,
                ]
        # Convert the dimensions back to the original
        output_image = self.dim_order_backward(output_image)
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
