from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.it.monitor import Monitor
from aydin.regression.nn import NNRegressor


class N2SService:
    def __init__(self, scales=None, widths=None, monitoring_variables_emit=None):
        super(N2SService, self).__init__()
        self.md = Monitor(monitoring_variables_emit)  # TODO: get a better name md
        self.scales = scales if scales is not None else [1, 3, 5, 11, 21, 23, 47, 95]
        self.widths = widths if widths is not None else [3, 3, 3, 3, 3, 3, 3, 3]

    def run(self, noisy_image, progress_callback, monitoring_images=None):
        """
        Method to run Noise2Self service

        :param monitoring_images:
        :param progress_callback:
        :param self:
        :param noisy_image: input noisy image, must be np compatible
        :return: denoised version of the input image, will be np compatible
        """
        progress_callback.emit(0)
        generator = FastMultiscaleConvolutionalFeatures(
            kernel_widths=self.widths,
            kernel_scales=self.scales,
            kernel_shapes=['l1'] * len(self.scales),
        )

        progress_callback.emit(35)
        # TODO: for now we go for NNRegressor, later we will implement machinery to choose
        # regressor = GBMRegressor(
        #     learning_rate=0.01,
        #     num_leaves=127,
        #     max_bin=512,
        #     n_estimators=2048,
        #     patience=20,
        # )
        regressor = NNRegressor()

        progress_callback.emit(51)
        it = ImageTranslatorClassic(
            feature_generator=generator, regressor=regressor, normaliser_type='identity'
        )
        response = it.train(
            noisy_image,
            noisy_image,
            # monitoring_variables=self.md,
            monitoring_images=monitoring_images,
        )
        progress_callback.emit(100)
        return response
