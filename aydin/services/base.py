import glob
import os
import shutil

import numpy as np

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.it.it_base import ImageTranslatorBase
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn import NNRegressor
from aydin.util.log.log import lprint


class BaseService:
    def __init__(
        self,
        scales=None,
        widths=None,
        backend_preference=None,
        use_model_flag=None,
        input_model_path=None,
    ):
        super().__init__()
        self.scales = scales if scales is not None else [1, 3, 5, 11, 21, 23, 47, 95]
        self.widths = widths if widths is not None else [3, 3, 3, 3, 3, 3, 3, 3]
        self.backend_preference = backend_preference

        self.input_model_path = input_model_path
        self.use_model_flag = use_model_flag
        self.model_folder_path = None

        self.generator = None
        self.regressor = None
        self.it = None

        self.has_less_than_one_million_voxels = False
        self.has_less_than_one_trillion_voxels = True
        self.number_of_dims = -1

    def update_paths(self, image_path):
        if self.input_model_path is None:
            self.model_folder_path = os.path.join(
                os.path.dirname(image_path), "_aydinmodel"
            )
            self.input_model_path = self.model_folder_path + ".zip"
        else:
            self.model_folder_path = os.path.dirname(self.input_model_path)

    @staticmethod
    def archive_model(source, destination):
        name = "_aydinmodel"
        format = "zip"
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        if os.path.exists(os.path.join(destination, '%s.%s' % (name, format))):
            lprint(
                "Previously existing model will be deleted before saving the new model"
            )
            os.remove(os.path.join(destination, '%s.%s' % (name, format)))
        shutil.move('%s.%s' % (name, format), destination)

    def save_model(self, image_path):
        # Check if model should be saved, if so save the model
        if self.use_model_flag is False and image_path is not None:

            # Save the model first
            self.it.save(self.model_folder_path)

            # Make archieve for the model
            self.archive_model(self.model_folder_path, os.path.dirname(image_path))

        # clean the model folder
        self.clean_model_folder()

    def clean_model_folder(self):
        if self.model_folder_path is not None:
            # Remove the files and the folder for the model
            files = glob.glob(os.path.join(self.model_folder_path, '*'))
            for f in files:
                os.remove(f)
            os.rmdir(self.model_folder_path)

    def stop_func(self):
        self.it.stop_training()

    def set_image_metrics(self, image_shape):
        self.number_of_dims = len(image_shape)
        number_of_voxels = np.prod(np.array(image_shape))
        self.has_less_than_one_million_voxels = number_of_voxels < 1000000
        self.has_less_than_one_trillion_voxels = number_of_voxels < 1000000000000

    def get_generator(self):
        if self.has_less_than_one_million_voxels:
            self.generator = FastMultiscaleConvolutionalFeatures(
                max_level=10, include_median_features=True
            )
        elif self.number_of_dims > 2:
            if self.has_less_than_one_trillion_voxels:
                dtype = np.uint16  # TODO: what about float16?
            else:
                dtype = np.float32
            self.generator = FastMultiscaleConvolutionalFeatures(
                max_level=4, include_median_features=False, dtype=dtype
            )
        else:
            self.generator = FastMultiscaleConvolutionalFeatures(
                max_level=10, include_median_features=True
            )
        return self.generator

    def get_regressor(self):
        if self.backend_preference is not None:
            if self.backend_preference == "lgbm":
                self.regressor = GBMRegressor(
                    learning_rate=0.01,
                    num_leaves=127,
                    max_bin=512,
                    n_estimators=2048,
                    patience=5,
                )
            elif self.backend_preference == "nn":
                self.regressor = NNRegressor()
            else:
                raise Exception("Non-valid backend option")
        else:
            if self.has_less_than_one_million_voxels:
                self.regressor = GBMRegressor(
                    learning_rate=0.01,
                    num_leaves=127,
                    max_bin=512,
                    n_estimators=2048,
                    patience=5,
                )
            else:
                self.regressor = NNRegressor()
        return self.regressor

    def get_translator(self, feature_generator, regressor, normaliser_type, monitor):
        # Use a pre-saved model or train a new one from scratch and save it
        if self.use_model_flag:
            # Unarchive the model file and load its ImageTranslator object into self.it
            shutil.unpack_archive(
                self.input_model_path, os.path.dirname(self.input_model_path), "zip"
            )
            self.it = ImageTranslatorBase.load(self.input_model_path[:-4])
        else:
            self.it = ImageTranslatorClassic(
                feature_generator=feature_generator,
                regressor=regressor,
                normaliser_type=normaliser_type,
                monitor=monitor,
                balance_training_data=type(regressor) is NNRegressor,
                analyse_correlation=False,
            )
        return self.it
