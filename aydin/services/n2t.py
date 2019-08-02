from aydin.features.classic.mcfocl import MultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor


class N2TService:
    scales = [1, 3, 5, 11, 21, 23, 47, 95]
    widths = [3, 3, 3, 3, 3, 3, 3, 3]

    def __init__(self, scales=None, widths=None):
        if scales is not None:
            N2TService.scales = scales
        if widths is not None:
            N2TService.widths = widths

    @staticmethod
    def run(noisy_image, image, noisy_test):
        # TODO: add previously trained model checks and desired behavior
        """
        Method to run Noise2Truth service

        :param noisy_image: input noisy image, must be np compatible
        :param image: input noisy image, must be np compatible
        :param noisy_test: input noisy image, must be np compatible
        :return: denoised version of the input image, will be np compatible
        """

        generator = MultiscaleConvolutionalFeatures(
            kernel_widths=N2TService.widths,
            kernel_scales=N2TService.scales,
            exclude_center=False,
        )

        regressor = GBMRegressor(num_leaves=63, n_estimators=512)

        it = ImageTranslatorClassic(feature_generator=generator, regressor=regressor)

        denoised = it.train(
            noisy_image, image
        )  # TODO: figure out what is going on with return
        return it.translate(noisy_test)
