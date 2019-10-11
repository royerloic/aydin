import os, time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy
from skimage.exposure import rescale_intensity

from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.normaliser.percentile import PercentileNormaliser
from aydin.normaliser.identity import IdentityNormaliser
from aydin.normaliser.minmax import MinMaxNormaliser
from aydin.it.cnn.unet import unet3D_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


class ImageTranslatorCNN:
    def __init__(
        self,
        input_dim: tuple,
        supervised: bool = False,
        shiftconv: bool = True,
        normaliser_type: str = 'percentile',
        rot_batch=1,
        # analyse_correlation: bool = False,
        # monitor=None,
    ):
        # super().__init__(normaliser_type, analyse_correlation, monitor)
        self.model = unet3D_model(
            input_dim,
            num_lyr=5,
            shiftconv=shiftconv,
            supervised=supervised,
            rot_batch_size=rot_batch,
        )
        self.supervised = supervised
        self.shiftconv = shiftconv
        self.input_dim = input_dim
        self.normaliser_type = normaliser_type
        self.max_epochs = None
        self.input_normaliser = None
        self.target_normaliser = None
        self.self_supervised = None
        self.early_stopping = None
        self.reduce_learning_rate = None
        self.trained_ep = None
        self.batch_size = None

    def train(
        self,
        input_image,
        target_image,
        train_test_ratio=0.1,  # TODO: remove this argument from base and add it to the rest of it_xxx.
        batch_dims=None,
        batch_size=None,
        batch_shuffle=False,
        monitoring_images=None,
        # callback_period=3,
        # patience=3,
        # patience_epsilon=0.000001,
        max_epochs=1024,
    ):
        self.max_epochs = max_epochs
        self.self_supervised = input_image is target_image
        self.batch_size = batch_size

        mask_shape = (3, 3, 3)  # mask shape for masking approach
        EStop_patience = 8
        ReduceLR_patience = 4
        min_delta = 1e-6
        # ):

        # Instanciates normaliser(s):
        if self.normaliser_type == 'identity':
            self.input_normaliser = IdentityNormaliser()
            self.target_normaliser = (
                self.input_normaliser
                if self.self_supervised
                else PercentileNormaliser()
            )
        elif self.normaliser_type == 'percentile':
            self.input_normaliser = PercentileNormaliser()
            self.target_normaliser = (
                self.input_normaliser
                if self.self_supervised
                else PercentileNormaliser()
            )
        elif self.normaliser_type == 'minmax':
            self.input_normaliser = MinMaxNormaliser()
            self.target_normaliser = (
                self.input_normaliser if self.self_supervised else MinMaxNormaliser()
            )

        self.input_normaliser.calibrate(input_image)
        if not self.self_supervised:
            self.target_normaliser.calibrate(target_image)

        # 'Last minute' normalisation:
        normalised_input_image = self.input_normaliser.normalise(input_image)
        if not self.self_supervised:
            normalised_target_image = normalised_input_image
        else:
            normalised_target_image = self.target_normaliser.normalise(target_image)

        def masker(batch_vol, i, mask_shape):
            i = i % numpy.prod(mask_shape)
            mask = numpy.zeros(numpy.prod(mask_shape), dtype=bool)
            mask[i] = True
            mask = mask.reshape(mask_shape)
            rep = numpy.ceil(
                numpy.asarray(batch_vol) / numpy.asarray(mask_shape)
            ).astype(
                int
            )  # the # of repetition to cover the whole input data chunk
            mask = numpy.tile(mask, tuple(rep))
            mask = mask[: batch_vol[0], : batch_vol[1], : batch_vol[2]]
            return mask

        def maskedgen(batch_vol, mask_shape, image, batch_size):
            while True:
                for j in range(numpy.ceil(image.shape[0] / batch_size).astype(int)):
                    image_batch = image[j * batch_size : (j + 1) * batch_size, ...]
                    for i in range(numpy.prod(mask_shape).astype(int)):
                        mask = masker(batch_vol, i, mask_shape)
                        masknega = numpy.broadcast_to(
                            numpy.expand_dims(numpy.expand_dims(mask, 0), 3),
                            image_batch.shape,
                        )
                        train_img = (
                            numpy.broadcast_to(
                                numpy.expand_dims(numpy.expand_dims(~mask, 0), 3),
                                image_batch.shape,
                            )
                            * image_batch
                        )
                        target_img = masknega * image_batch
                        yield {
                            'input': train_img,
                            'input_msk': masknega.astype(numpy.float32),
                        }, target_img

        # Here is the list of callbacks:
        callbacks = []

        # # Set upstream callback:
        # self.keras_callback = NNCallback(regressor_callback)

        # Early stopping callback:
        self.early_stopping = EarlyStopping(
            # self,
            monitor='loss',
            min_delta=min_delta,  # 0.000001 if is_batch else 0.0001,
            patience=EStop_patience,
            mode='auto',
            restore_best_weights=True,
        )

        # Reduce LR on plateau:
        self.reduce_learning_rate = ReduceLROnPlateau(
            monitor='loss',
            min_delta=min_delta,
            factor=0.1,
            verbose=1,
            patience=ReduceLR_patience,
            mode='auto',
            min_lr=1e-8,
        )

        # Add callbacks to the list:
        callbacks.append(self.early_stopping)
        callbacks.append(self.reduce_learning_rate)

        if self.batch_size is None:
            self.batch_size = 1

        if self.supervised:
            history = self.model.fit(
                normalised_input_image,
                normalised_target_image,
                batch_size=self.batch_size,
                epochs=self.max_epochs,
                callbacks=callbacks,
            )
        elif self.shiftconv:
            history = self.model.fit(
                normalised_input_image,
                normalised_target_image,
                batch_size=self.batch_size,
                epochs=self.max_epochs,
                callbacks=callbacks,
            )
        else:
            history = self.model.fit_generator(
                maskedgen(
                    self.input_dim[:-1],
                    mask_shape,
                    normalised_input_image,
                    self.batch_size,
                ),
                epochs=self.max_epochs,
                steps_per_epoch=numpy.prod(mask_shape)
                * numpy.ceil(input_image.shape[0] / self.batch_size).astype(int),
                callbacks=callbacks,
            )
        hist = history.history
        self.trained_ep = len(hist['loss'])

        if not self.shiftconv and not self.supervised:
            return self.model.predict(
                [normalised_input_image, numpy.ones(input_image.shape)],
                batch_size=self.batch_size,
                verbose=1,
            )

        else:
            return self.model.predict(
                normalised_input_image, batch_size=self.batch_size, verbose=1
            )

    def translate(self, input_image, batch_dim=None):

        # First we normalise the input:
        normalised_input_image = self.input_normaliser.normalise(input_image)

        if not self.shiftconv and not self.supervised:
            translated_image = self.model.predict(
                [normalised_input_image, numpy.ones(normalised_input_image.shape)],
                batch_size=self.batch_size,
                verbose=1,
            )
        else:
            translated_image = self.model.predict(
                normalised_input_image, batch_size=self.batch_size, verbose=1
            )

        # Then we denormalise:
        return self.target_normaliser.denormalise(translated_image)


### run a demo


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


# Load image
image_path = examples_single.gardner_org.get_path()
image0, metadata = io.imread(image_path)
print(image0.shape)
image0 = n(image0.squeeze())

# Transform data to an array of small blocks to fit into the model.
blocksize = 64
batch_size = 5
image_size = image0.shape
input_data = []
for d1 in range(image_size[1] // blocksize):
    for d2 in range(image_size[2] // blocksize):
        for d3 in range(image_size[3] // blocksize):
            input_data.append(
                image0[
                    1,
                    d1 * blocksize : (d1 + 1) * blocksize,
                    d2 * blocksize : (d2 + 1) * blocksize,
                    d3 * blocksize : (d3 + 1) * blocksize,
                ]
            )
image = numpy.stack(input_data)
print(image.shape)
image = image.astype(numpy.float32)
image = numpy.expand_dims(image, axis=-1)

it = ImageTranslatorCNN(
    input_dim=image.shape[1:], supervised=False, shiftconv=True, rot_batch=batch_size
)

start = time.time()
denoised = it.train(image, image, batch_size=batch_size)
denoised = denoised.reshape(image.shape)
stop = time.time()
print(f"Training: elapsed time:  {stop-start} ")

# Testing section
# Create a larger data matrix with dimensions of multiple of blocksize.
test_data_size = (
    numpy.ceil(numpy.asarray(image0.shape[1:]) / blocksize).astype(int) * blocksize
)
test_data = numpy.zeros(test_data_size)
test_data[numpy.ones(image_size[1:], dtype=bool)] = image0[1]
input_data = []
for d1 in range(test_data_size[0] // blocksize):
    for d2 in range(test_data_size[1] // blocksize):
        for d3 in range(test_data_size[2] // blocksize):
            input_data.append(
                test_data[
                    d1 * blocksize : (d1 + 1) * blocksize,
                    d2 * blocksize : (d2 + 1) * blocksize,
                    d3 * blocksize : (d3 + 1) * blocksize,
                ]
            )
test_data = numpy.expand_dims(numpy.stack(input_data), -1)

# Start testing
start = time.time()
denoised_inf = it.translate(test_data)
denoised_inf = denoised_inf.squeeze()
stop = time.time()
print(f"inference: elapsed time:  {stop - start} ")

# Convert arrayed data to the original shape.
denoised_rec = numpy.zeros(test_data_size)
for d1 in range(test_data_size[0] // blocksize):
    for d2 in range(test_data_size[1] // blocksize):
        for d3 in range(test_data_size[2] // blocksize):
            denoised_rec[
                d1 * blocksize : (d1 + 1) * blocksize,
                d2 * blocksize : (d2 + 1) * blocksize,
                d3 * blocksize : (d3 + 1) * blocksize,
            ] = denoised_inf[
                (
                    d1
                    * (
                        (test_data_size[1] // blocksize)
                        * (test_data_size[2] // blocksize)
                    )
                    + d2 * (test_data_size[2] // blocksize)
                    + d3
                ),
                ...,
                0,
            ]
denoised_rec = denoised_rec[: image_size[1], : image_size[2], : image_size[3]]
numpy.save(f'den_msk_{blocksize}.npy', denoised_rec)
