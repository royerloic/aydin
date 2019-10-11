import numpy as np

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn import NNRegressor


class BaseService:
    def __init__(self, monitor=None, scales=None, widths=None):
        super().__init__()
        self.monitor = monitor
        self.scales = scales if scales is not None else [1, 3, 5, 11, 21, 23, 47, 95]
        self.widths = widths if widths is not None else [3, 3, 3, 3, 3, 3, 3, 3]

        self.generator = None
        self.regressor = None
        self.it = None

        self.has_less_than_one_million_voxels = False
        self.has_less_than_one_trillion_voxels = True
        self.number_of_dims = -1

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
        if self.has_less_than_one_million_voxels:
            self.regressor = GBMRegressor(
                learning_rate=0.01,
                num_leaves=127,
                max_bin=512,
                n_estimators=2048,
                patience=20,
            )
        else:
            self.regressor = NNRegressor()
        return self.regressor

    def get_translator(self, feature_generator, regressor, normaliser_type, monitor):
        self.it = ImageTranslatorClassic(
            feature_generator=feature_generator,
            regressor=regressor,
            normaliser_type=normaliser_type,
            monitor=monitor,
            balance_training_data=type(regressor) is NNRegressor,
            analyse_correlation=False,
        )
        return self.it
