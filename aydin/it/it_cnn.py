import psutil
import time, math
from os.path import join

import numpy
import random

from aydin.it.cnn.cnn_util.memorycheck import MemoryCheckCNN
from aydin.io.folders import get_temp_folder

from aydin.it.cnn.unet import unet_model
from aydin.it.it_base import ImageTranslatorBase
from aydin.util.log.log import lsection, lprint
from aydin.it.cnn.layers import Maskout, split, rot90, maskedgen
from aydin.regression.nn_utils.callbacks import ModelCheckpoint
from aydin.it.cnn.cnn_util.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    CNNCallback,
)
from aydin.it.cnn.cnn_util.receptive_field import receptive_field_model
from keras.engine.saving import model_from_json
from aydin.it.cnn.cnn_util.data_util import random_sample_patches, sim_model_size


class ImageTranslatorCNN(ImageTranslatorBase):
    """
        aydin CNN Image Translator

        Using CNN (Unet and Co)

    """

    # Device max mem:
    device_max_mem = psutil.virtual_memory().total

    def __init__(
        self,
        normaliser_type: str = 'percentile',
        analyse_correlation: bool = False,
        monitor=None,
    ):

        super().__init__(normaliser_type, analyse_correlation, monitor)
        self.model = None
        self.infmodel = None
        self.supervised = None
        self.shiftconv = None
        self.input_dim = None
        self.max_epochs = None
        self.patience = None
        self.checkpoint = None
        self.stop_fitting = False
        self.rand_smpl_size = None
        self.batch_size = None
        self.num_lyr = None
        self.tile_size = None
        self.train_img_size = None
        self.batch_norm = None
        self.rot_batch_size = None

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

    ##TODO: We exclude certain fields from saving:
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
                self.model = model_from_json(
                    loaded_model_json,
                    custom_objects={'Maskout': Maskout, 'split': split, 'rot90': rot90},
                )
            # load weights into new model:
            keras_model_file = join(path, 'keras_weights.txt')
            self.model.load_weights(keras_model_file)

    def get_receptive_field_radius(self, nb_dim):
        r = receptive_field_model(self.model)
        r = numpy.sqrt(r)  # effective receptive field
        return int(r // 2)

    def is_enough_memory(self, array):
        return MemoryCheckCNN(self.model.count_params()).is_enough_memory(array)

    def limit_epochs(self, max_epochs: int) -> int:
        return max_epochs

    def stop_training(self):
        self.stop_fitting = True

    # TODO: check the order of dimensions e.g. (B, H, W, C)
    def dim_order(self):
        pass

    def train(
        self,
        input_image,  # dimension (batch, H, W, C)
        target_image,  # dimension (batch, H, W, C)
        shiftconv=True,
        train_data_ratio=1,  # TODO: remove this argument from base and add it to the rest of it_xxx. !!WAIT!!
        batch_dims=None,
        batch_size=1,
        batch_shuffle=False,
        monitoring_images=None,
        num_lyr=5,
        callback_period=3,
        patience=4,
        patience_epsilon=0.000001,
        max_epochs=1024,
        batch_norm=None,
        tile_size=None,  # tile size for patch sample e.g. 64 or (64, 64)
        tile_batch_scale=1,  # batch size of tiling = tile_batch_scale * batch_size; only for tiling
        adoption_rate=0.5,  # the ratio of randomly sampled patches will be used. e.g., 100 patch images are randomly
        # sampled and <adoption_rate> of them will be used for training and the rest will be discarded.
        min_num_tile=64,
    ):
        self.batch_dims = batch_dims
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.callback_period = callback_period
        self.patience = patience
        self.checkpoint = None
        self.input_dim = input_image.shape[1:]
        self.num_lyr = num_lyr
        self.supervised = True
        self.batch_norm = batch_norm
        self.shiftconv = shiftconv
        if input_image is target_image:
            self.supervised = False

        # Check whether input dimensions fit CNN
        d = numpy.array(input_image.shape[1:-1])  # input dimension
        d1 = 2 ** self.num_lyr  # reduction rate

        if tile_size is None:
            if not (d / d1 == d // d1).all():
                tile_size = [64 for _ in self.input_dim[:-1]]
            # Check if model can fit the memory size
            if self.device_max_mem * 0.8 <= sim_model_size(self.input_dim[:-1]):
                tile_size = [64 for _ in self.input_dim[:-1]]
                assert self.device_max_mem * 0.8 <= sim_model_size(
                    tile_size
                ), 'Memory is too low to fit a CNN model.'

        else:
            assert numpy.unique(tile_size).size == 1, (
                'Tiles with different length in each dim are currently not accepted.'
                'Please make tile_size with the same length in all dimensions. '
            )

            # Random patch sampling
            if type(tile_size) == int:
                tile_size = [tile_size for _ in self.input_dim[:-1]]

            self.rand_smpl_size = numpy.array(tile_size)
            assert (
                self.rand_smpl_size / 2 ** self.num_lyr > 0
            ).any(), f'Sampling patch size is too small. Patch size has to be >= {2 ** self.num_lyr}'
            assert (
                self.rand_smpl_size % 2 ** self.num_lyr == 0
            ).any(), f'Sampling path size has to be multiple of 2^{self.num_lyr}'
        self.tile_size = tile_size
        self.train_img_size = tile_size if tile_size else input_image.shape[1:-1]

        # Determine batch size according to GPU memory
        multiple = max(
            int(
                ((provider.device_max_mem * 0.8) / sim_model_size(self.train_img_size))
                // min_num_tile
            ),
            1,
        )
        if self.tile_size:
            self.batch_size = min_num_tile * multiple
        else:
            self.batch_size = math.gcd(
                input_image.shape[0],
                math.floor(
                    (self.device_max_mem * 0.8) / sim_model_size(self.train_img_size)
                ),
            )
        self.rot_batch_size = self.batch_size
        lprint("Max mem: ", self.device_max_mem)
        lprint(f"Keras batch size for training: {self.batch_size}")

        # Tile input and target image
        if tile_size is not None:
            input_patch_idx = random_sample_patches(
                input_image,
                tile_size,
                self.batch_size * tile_batch_scale,
                adoption_rate,
            )

            img_patch = []
            for i in input_patch_idx:
                img_patch.append(
                    input_image[
                        i[0], i[1] : i[1] + tile_size[0], i[2] : i[2] + tile_size[1], :
                    ]
                )
            input_image = numpy.stack(img_patch)
            self.batch_dims = (True,) + (False,) * len(input_image.shape[1:])

            if self.supervised:
                target_patch = []
                for i in input_patch_idx:
                    target_patch.append(
                        target_image[
                            i[0],
                            i[1] : i[1] + tile_size[0],
                            i[2] : i[2] + tile_size[1],
                            :,
                        ]
                    )
                target_image = numpy.stack(target_patch)
            else:
                target_image = input_image
        self.train_img_size = input_image.shape[1:-1]

        self.model = unet_model(
            input_image.shape[1:],
            rot_batch_size=self.rot_batch_size,  # input_image.shape[0],
            num_lyr=self.num_lyr,
            normalization=batch_norm,
            supervised=self.supervised,
            shiftconv=shiftconv,
        )

        super().train(
            input_image,
            target_image,
            batch_dims=self.batch_dims,
            train_data_ratio=train_data_ratio,  # TODO: to be removed as this will not be used in it_cnn   !!WAIT!!
            callback_period=callback_period,
            max_epochs=max_epochs,
            patience=patience,
            patience_epsilon=patience_epsilon,
        )

    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        train_data_ratio=1,
        max_voxels_for_training=math.inf,
        train_valid_ratio=0.1,
        callback_period=3,
        mask_shape=(9, 9),  # mask shape for masking approach
        is_batch=False,
        monitoring_images=None,
        EStop_patience=None,
        ReduceLR_patience=None,
        min_delta=0,  # 1e-6,
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

            self.is_batch = is_batch
            if EStop_patience is None:
                EStop_patience = self.patience * 2
            if ReduceLR_patience is None:
                ReduceLR_patience = self.patience

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
                                x_m,
                                batch_size=self.batch_size,
                                tile_size=[64 for _ in self.input_dim[:-1]],
                            )
                            for x_m in self.monitoring_datasets
                        ]
                        inferred_images = [
                            y_m.reshape(self.train_img_size)
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

            # Effective number of epochs:
            # effective_number_of_epochs = 2 if is_batch else self.max_epochs
            # lprint(f"Effective max number of epochs: {effective_number_of_epochs}")

            # Early stopping patience:
            early_stopping_patience = EStop_patience  # if is_batch else self.patience
            lprint(f"Early stopping patience: {early_stopping_patience}")

            # Effective LR patience:
            effective_lr_patience = (
                ReduceLR_patience  # if is_batch else max(1, self.patience // 2)
            )
            lprint(f"Effective LR patience: {effective_lr_patience}")

            # Here is the list of callbacks:
            callbacks = []

            # Set upstream callback:
            self.keras_callback = CNNCallback(monitor_callback)

            # Early stopping callback:
            self.early_stopping = EarlyStopping(
                self,
                monitor='loss',
                min_delta=min_delta,  # 0.000001 if is_batch else 0.0001,
                patience=early_stopping_patience,
                mode='auto',
                restore_best_weights=True,
            )

            # Reduce LR on plateau:
            self.reduce_learning_rate = ReduceLROnPlateau(
                monitor='loss',
                min_delta=min_delta,
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
                    self.model_file_path, monitor='loss', verbose=1, save_best_only=True
                )

                # Add callbacks to the list:
                callbacks.append(self.keras_callback)
                callbacks.append(self.early_stopping)
                callbacks.append(self.reduce_learning_rate)
                callbacks.append(self.checkpoint)

            lprint("Training now...")
            if is_batch:
                # TODO: YOU DON"T HAVE TO WORRY ABOUT THIS (for now)
                pass
            else:
                if self.supervised:
                    history = self.model.fit(
                        input_image,
                        target_image,
                        batch_size=self.batch_size,
                        epochs=self.max_epochs,
                        callbacks=callbacks,
                        verbose=0,
                    )
                elif self.shiftconv:
                    history = self.model.fit(
                        input_image,
                        input_image,
                        batch_size=self.batch_size,
                        epochs=self.max_epochs,
                        callbacks=callbacks,
                        verbose=0,
                    )
                else:
                    history = self.model.fit_generator(
                        maskedgen(
                            self.train_img_size,
                            mask_shape,
                            input_image,
                            self.batch_size,
                        ),
                        epochs=self.max_epochs,
                        steps_per_epoch=numpy.prod(mask_shape)
                        * numpy.ceil(input_image.shape[0] / self.batch_size).astype(
                            int
                        ),
                        verbose=1,
                        callbacks=callbacks,
                    )

    def translate(
        self, input_image, translated_image=None, batch_dims=None, tile_size=None
    ):
        return super().translate(
            input_image,
            translated_image,
            batch_dims=self.batch_dims,
            tile_size=self.tile_size
            if self.tile_size is None
            else numpy.unique(self.tile_size),
        )

    def _translate(self, input_image, batch_dim=None):
        input_shape = input_image.shape
        if input_shape[1:-1] != self.tile_size and self.tile_size is not None:
            pad_width0 = (
                numpy.array(self.tile_size) - numpy.array(input_image.shape[1:-1])
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

        if not self.shiftconv and not self.supervised:
            output_image = self.model.predict(
                [input_image, numpy.ones(input_image.shape)],
                batch_size=self.batch_size,
                verbose=1,
            )
        else:
            # Change the batchsize in split layer accordingly
            if self.infmodel is None:
                self.infmodel = unet_model(
                    input_image.shape[1:],
                    rot_batch_size=input_image.shape[0],
                    num_lyr=self.num_lyr,
                    normalization=self.batch_norm,
                    supervised=self.supervised,
                    shiftconv=self.shiftconv,
                )
                self.infmodel.set_weights(self.model.get_weights())
            output_image = self.infmodel.predict(
                input_image, batch_size=input_image.shape[0], verbose=1
            )

        if input_shape[1:-1] != self.tile_size and self.tile_size is not None:
            if len(self.tile_size) == 2:
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
        return output_image

    ## TODO: modify these functions to generate tiles with the same size of tile_size
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

    def _get_margins(self, shape, tilling_strategy):

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
