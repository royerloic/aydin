import time
from os.path import join

import numpy
import random

from aydin.it.cnn.memorycheck import MemoryCheckCNN
from aydin.io.folders import get_temp_folder
from aydin.providers.plaidml.plaidml_provider import PlaidMLProvider

provider = (
    PlaidMLProvider()
)  # NOTE: This line should stay exactly here! All keras calls must be _AFTER_ the line below:
from aydin.it.cnn.unet import unet_model
from aydin.it.it_base import ImageTranslatorBase

from aydin.util.log.logging import lsection, lprint

from aydin.regression.nn_utils.callbacks import ModelCheckpoint  # , EarlyStopping
from aydin.it.cnn.callbacks import NNCallback, EarlyStopping, ReduceLROnPlateau

# from keras.callbacks import ReduceLROnPlateau  # , EarlyStopping


class ImageTranslatorCNN(ImageTranslatorBase):
    """
        aydin CNN Image Translator

        Using CNN (Unet and Co)

    """

    def __init__(
        self,
        input_dim: tuple,
        supervised: bool = False,
        shiftconv: bool = True,
        normaliser_type: str = 'percentile',
        analyse_correlation: bool = False,
        monitor=None,
        # set model to None...
    ):

        super().__init__(normaliser_type, analyse_correlation, monitor)
        self.model = None
        self.supervised = None
        self.shiftconv = None
        self.input_dim = None
        self.max_epochs = None
        self.patience = None
        self.checkpoint = None
        self.stop_training = False

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).
        :param path: path to save to
        """
        with lsection(f"Saving CNN image translator to {path}"):
            frozen = super().save(path)

            # TODO: We don't have feature gen or regressors here:
            frozen += self.feature_generator.save(path) + '\n'
            frozen += self.regressor.save(path) + '\n'

            # TODO: save keras model here, check how it is done in nn
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

        return frozen

    ##TODO: We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        # exclude fields below that should/cannot be saved properly:
        del state['early_stopping']
        del state['reduce_learning_rate']
        del state['checkpoint']
        del state['model']
        del state['keras_callback']
        return state

    # TODO: SPECIAL please fix the naming of the file for the weights here and in nn.

    # TODO: implement this method:
    def _load_internals(self, path: str):
        pass
        # # load JSON and create model:
        # keras_model_file = join(path, 'keras_model.txt')
        # with open(keras_model_file, 'r') as json_file:
        #     loaded_model_json = json_file.read()
        #     self.model = model_from_json(loaded_model_json)
        # # load weights into new model:
        # keras_model_file = join(path, 'keras_weights.txt')
        # self.model.load_weights(keras_model_file)

    # TODO: get an estimate, upper bound is ok:
    def get_receptive_field_radius(self, nb_dim):
        pass

    def is_enough_memory(self, array):
        return MemoryCheckCNN(self.model.count_params()).is_enough_memory(array)

    def limit_epochs(self, max_epochs: int) -> int:
        return max_epochs

    def _load_internals(self, path: str):
        pass

    # TODO: implement:
    def stop_training(self):
        self.model.stop_training = True
        return self.model.stop_training

    def train(
        self,
        input_image,  # dimension (batch, H, W, C)
        target_image,  # dimension (batch, H, W, C)
        supervised=False,
        shiftconv=True,
        train_test_ratio=0.1,  # TODO: remove this argument from base and add it to the rest of it_xxx. !!WAIT!!
        batch_dims=None,
        batch_size=None,
        batch_shuffle=False,
        monitoring_images=None,
        callback_period=3,
        patience=3,
        patience_epsilon=0.000001,
        max_epochs=1024,
    ):
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint = None
        self.input_dim = input_image.shape[1:]
        self.model = unet_model(
            input_image.shape[1:],
            rot_batch_size=1,
            num_lyr=5,
            normalization='batch',
            supervised=supervised,
            shiftconv=shiftconv,
        )
        self.supervised = supervised
        self.shiftconv = shiftconv

        # TODO: THIS IS WHERE YOU HAVE TO RESET YOUR KERAS MODEL, which means you create the model here.

        # TODO: Indeed, instanciate here model given the correct dimension obtained from the input/target image

        return super().train(
            input_image,
            target_image,
            train_test_ratio,  # TODO: to be removed as this will not be used in it_cnn   !!WAIT!!
            batch_dims,
            batch_size,
            batch_shuffle,
            monitoring_images,
            callback_period,
            patience,
            patience_epsilon,
        )

    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        mask_shape=(3, 3),  # mask shape for masking approach
        is_batch=False,
        monitoring_images=None,
        callback_period=3,
        EStop_patience=8,  # TODO: THIS METHOD MUST STRICTKLY ADHERE TO THE CONTRACT DEFINED IN THE BASE CLASS
        ReduceLR_patience=4,  # TODO Set both patience to the 'master' patience ( self.patience )
        min_delta=1e-6,
    ):

        self.is_batch = is_batch

        def masker(batch_vol, i, mask_shape):
            i = i % numpy.prod(mask_shape)
            mask = numpy.zeros(numpy.prod(mask_shape), dtype=bool)
            mask[i] = True
            mask = mask.reshape(mask_shape)
            rep = numpy.ceil(
                numpy.asarray(batch_vol) / numpy.asarray(mask_shape)
            ).astype(int)
            mask = numpy.tile(mask, tuple(rep))
            mask = mask[: batch_vol[0], : batch_vol[1]]
            return mask

        def maskedgen(batch_vol, mask_shape, image, batch_size):
            while True:
                for j in range(numpy.ceil(image.shape[0] / batch_size).astype(int)):
                    image_batch = image[j * batch_size : (j + 1) * batch_size, ...]
                    for i in range(numpy.prod(mask_shape).astype(int)):
                        mask = masker(batch_vol, i, mask_shape)
                        masknega = numpy.broadcast_to(
                            numpy.expand_dims(numpy.expand_dims(mask, 0), -1),
                            image_batch.shape,
                        )
                        train_img = (
                            numpy.broadcast_to(
                                numpy.expand_dims(numpy.expand_dims(~mask, 0), -1),
                                image_batch.shape,
                            )
                            * image_batch
                        )
                        target_img = masknega * image_batch
                        yield {
                            'input': train_img,
                            'input_msk': masknega.astype(numpy.float32),
                        }, target_img

        # Regressor callback:
        def regressor_callback(iteration, val_loss, model):

            current_time_sec = time.time()

            # Correct for dtype range:
            # if self.feature_generator.dtype == numpy.uint8:
            #     val_loss /= 255
            # elif self.feature_generator.dtype == numpy.uint16:
            #     val_loss /= 255 * 255

            if current_time_sec > self.last_callback_time_sec + self.callback_period:
                # TODO: predict image for display on the end of every epoch
                if self.monitoring_datasets and self.monitor:
                    predicted_monitoring_datasets = [
                        self.regressor.predict(x_m, model_to_use=model)
                        for x_m in self.monitoring_datasets
                    ]
                    inferred_images = [
                        y_m.reshape(image.shape)
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
            else:
                pass
            # print(f"Iteration={iteration} metric value: {eval_metric_value} ")

        # The bigger the batch size the faster training in terms of time per epoch,
        # but small batches are also better for convergence (inherent batch noise).
        # We make sure that we have at least about 1000 items per batch for small images,
        # which is a good minimum. For larger datasets we get bigger batches which is fine.
        batch_size = 1  # max(1, x_train.shape[0] // 256)
        lprint("Max mem: ", provider.device_max_mem)

        # Heuristic threshold here obtained by inspecting batch size per GPU memory
        # Basically ensures ratio of 700000 batch size per 12GBs of GPU memory
        batch_size = min(batch_size, (700000 * provider.device_max_mem) // 12884901888)
        lprint(f"Keras batch size for training: {batch_size}")
        self.batch_size = batch_size

        # # Effective number of epochs:
        # effective_number_of_epochs = 2 if is_batch else self.max_epochs
        # lprint(f"Effective max number of epochs: {effective_number_of_epochs}")

        # Early stopping patience:
        early_stopping_patience = EStop_patience  # if is_batch else self.patience
        lprint(f"Early stopping patience: {early_stopping_patience}")

        # Effective LR patience:
        effective_lr_patience = (
            ReduceLR_patience
        )  # if is_batch else max(1, self.patience // 2)
        lprint(f"Effective LR patience: {effective_lr_patience}")

        # Here is the list of callbacks:
        callbacks = []

        # # Set upstream callback:
        # self.keras_callback = NNCallback(regressor_callback)

        # Early stopping callback:
        self.early_stopping = EarlyStopping(
            # self,
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
            self.checkpoint = ModelCheckpoint(
                self.model_file_path, monitor='loss', verbose=1, save_best_only=True
            )

            # Add callbacks to the list:
            # callbacks.append(self.keras_callback)
            callbacks.append(self.early_stopping)
            callbacks.append(self.reduce_learning_rate)
            callbacks.append(self.checkpoint)

        with lsection(
            f"Training image translator from image of shape {input_image.shape} to image of shape {target_image.shape}:"
        ):

            lprint("Training now...")
            if is_batch:
                # TODO: YOU DON"T HAVE TO WORRY ABOUT THIS (for now)
                pass
            else:
                if self.supervised:
                    history = self.model.fit(
                        input_image,
                        target_image,
                        batch_size=batch_size,
                        epochs=self.max_epochs,
                        callbacks=callbacks,
                        verbose=0,
                    )
                elif self.shiftconv:
                    history = self.model.fit(
                        input_image,
                        input_image,
                        batch_size=batch_size,
                        epochs=self.max_epochs,
                        callbacks=callbacks,
                        verbose=0,
                    )
                else:
                    history = self.model.fit_generator(
                        maskedgen(
                            self.input_dim[:-1], mask_shape, input_image, batch_size
                        ),
                        epochs=self.max_epochs,
                        steps_per_epoch=numpy.prod(mask_shape)
                        * numpy.ceil(input_image.shape[0] / self.batch_size).astype(
                            int
                        ),
                        verbose=0
                        # callbacks=callbacks,
                    )

                if not self.shiftconv and not self.supervised:
                    return self.model.predict(
                        [input_image, numpy.ones(input_image.shape)],
                        batch_size=batch_size,
                        verbose=1,
                    )

                else:
                    return self.model.predict(
                        input_image, batch_size=batch_size, verbose=1
                    )

    def _translate(self, input_image, batch_dim=None):

        if not self.shiftconv and not self.supervised:
            return self.model.predict(
                [input_image, numpy.ones(input_image.shape)],
                batch_size=self.batch_size,
                verbose=1,
            )
        else:
            return self.model.predict(
                input_image, batch_size=self.batch_size, verbose=1
            )
