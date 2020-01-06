from aydin.it.monitor import Monitor
from aydin.services.base import BaseService


class N2SService(BaseService):
    """Noise2Self Service.
    """

    def __init__(self, backend_preference=None, scales=None, widths=None):
        super(N2SService, self).__init__(
            backend_preference=backend_preference, scales=scales, widths=widths
        )

    def run(
        self,
        noisy_image,
        progress_callback,
        monitoring_callbacks=None,
        monitoring_images=None,
        generator=None,
        regressor=None,
    ):
        """Method to run Noise2Self service

        :param regressor:
        :param generator:
        :param monitoring_callbacks:
        :param monitoring_images:
        :param progress_callback:
        :param self:
        :param noisy_image: input noisy image, must be np compatible
        :return: denoised version of the input image, will be np compatible
        """
        self.set_image_metrics(noisy_image.shape)
        progress_callback.emit(5)

        generator = generator if generator is not None else self.get_generator()
        regressor = regressor if regressor is not None else self.get_regressor()
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

        self.it.train(noisy_image, noisy_image)
        progress_callback.emit(80)

        response = self.it.translate(noisy_image)
        progress_callback.emit(100)
        return response
