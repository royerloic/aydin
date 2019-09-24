from aydin.it.monitor import Monitor
from aydin.services.base import BaseService


class N2TService(BaseService):
    def __init__(self, scales=None, widths=None):
        super(N2TService, self).__init__(scales=scales, widths=widths)

    def run(
        self,
        noisy_image,
        truth_image,
        test_image,
        progress_callback,
        monitoring_callbacks=None,
        monitoring_images=None,
    ):
        """
        Method to run Noise2Self service

        :param test_image:
        :param truth_image:
        :param monitoring_callbacks:
        :param monitoring_images:
        :param progress_callback:
        :param self:
        :param noisy_image: input noisy image, must be np compatible
        :return: denoised version of the input image, will be np compatible
        """
        progress_callback.emit(0)
        self.set_image_metrics(noisy_image.shape)
        generator = self.get_generator()

        progress_callback.emit(15)
        regressor = self.get_regressor()

        progress_callback.emit(41)
        self.it = self.get_translator(
            feature_generator=generator,
            regressor=regressor,
            normaliser_type='percentile',
            monitor=Monitor(
                monitoring_callbacks=monitoring_callbacks,
                monitoring_images=monitoring_images,
            ),
        )

        self.it.train(noisy_image, truth_image)
        progress_callback.emit(100)

        response = self.it.translate(test_image)
        return response
