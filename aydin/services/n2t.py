from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.it.monitor import Monitor
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn import NNRegressor


class N2TService:
    def __init__(self, scales=None, widths=None, monitoring_variables_emit=None):
        self.monitor = None
        self.scales = scales if scales is not None else [1, 3, 5, 11, 21, 23, 47, 95]
        self.widths = widths if widths is not None else [3, 3, 3, 3, 3, 3, 3, 3]

    def run(
        self,
        noisy_image,
        truth_image,
        noisy_test,
        progress_callback,
        monitoring_callbacks=None,
        monitoring_images=None,
    ):
        # TODO: add previously trained model checks and desired behavior
        """
        Method to run Noise2Truth service

        :param monitoring_images:
        :param monitoring_callbacks:
        :param progress_callback:
        :param truth_image:
        :param noisy_image: input noisy image, must be np compatible
        :param image: input noisy image, must be np compatible
        :param noisy_test: input noisy image, must be np compatible
        :return: denoised version of the input image, will be np compatible
        """
        progress_callback.emit(0)
        generator = FastMultiscaleConvolutionalFeatures(
            kernel_widths=self.widths,
            kernel_scales=self.scales,
            kernel_shapes=['l1'] * len(self.scales),
        )

        progress_callback.emit(15)
        # TODO: for now we go for NNRegressor, later we will implement machinery to choose
        # regressor = GBMRegressor(
        #     learning_rate=0.01,
        #     num_leaves=127,
        #     max_bin=512,
        #     n_estimators=2048,
        #     patience=20,
        # )
        regressor = NNRegressor()

        progress_callback.emit(41)
        self.monitor = Monitor(
            monitoring_callbacks=monitoring_callbacks,
            monitoring_images=monitoring_images,
        )

        it = ImageTranslatorClassic(
            feature_generator=generator,
            regressor=regressor,
            normaliser_type='identity',
            monitor=self.monitor,
        )

        it.train(noisy_image, truth_image)
        progress_callback.emit(80)

        response = it.translate(noisy_test)

        return response
