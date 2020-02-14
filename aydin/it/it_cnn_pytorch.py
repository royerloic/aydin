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

    def __init__(
        self,
        normaliser_type='percentile',
        balance_training_data=None,
        keep_ratio=1,
        max_voxels_for_training=4e6,
        max_epochs=1024,
        patience=3,
        patience_epsilon=0.000001,
        monitor=None,
    ):
        """
        Constructs a CNN image translator using the pytorch deep learning library.

        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(normaliser_type, monitor=monitor)

        self.max_voxels_for_training = max_voxels_for_training
        self.keep_ratio = keep_ratio
        self.balance_training_data = balance_training_data

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).
        :param path: path to save to
        """
        with lsection(f"Saving 'classic' image translator to {path}"):
            frozen = super().save(path)
            pass
            # TODO: complete!

        return frozen

    def _load_internals(self, path: str):
        with lsection(f"Loading 'classic' image translator from {path}"):
            pass
            # TODO: complete!

    ## We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['XXXXXXX']
        return state

    def get_receptive_field_radius(self, nb_dim):
        # TODO: estimate receptive field radius
        return 10

    def train(
        self,
        input_image,
        target_image,
        batch_dims=None,
        train_valid_ratio=0.1,
        callback_period=3,
    ):

        super().train(
            input_image,
            target_image,
            batch_dims=batch_dims,
            train_valid_ratio=train_valid_ratio,
            callback_period=callback_period,
        )

    def stop_training(self):
        pass
        # TODO implement training stop

    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        train_valid_ratio=0.1,
        callback_period=3,
    ):
        pass
        # TODO: implement  training

    def _translate(self, input_image, batch_dims=None):
        """
            Internal method that translates an input image on the basis of the trainined model.
        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
        """
        # TODO: implement translation
        inferred_image = None
        return inferred_image
