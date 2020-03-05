from aydin.it.monitor import Monitor
from aydin.services.base import BaseService
from aydin.util.log.log import lsection


class N2SService(BaseService):
    """Noise2Self Service.
    """

    def __init__(
        self,
        scales=None,
        widths=None,
        backend_preference=None,
        use_model_flag=None,
        input_model_path=None,
    ):
        super(N2SService, self).__init__(
            scales=scales,
            widths=widths,
            backend_preference=backend_preference,
            use_model_flag=use_model_flag,
            input_model_path=input_model_path,
        )

    def run(
        self,
        noisy_image,
        progress_callback,
        *,
        noisy_metadata=None,
        image_path=None,
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
        with lsection("Noise2Self service is starting..."):
            self.set_image_metrics(noisy_image.shape)
            progress_callback.emit(5)

            generator = generator if generator is not None else self.get_generator()
            regressor = regressor if regressor is not None else self.get_regressor()
            progress_callback.emit(41)

            # Handle image_path if it is not None
            if image_path is not None:
                self.update_paths(image_path)

            self.it = self.get_translator(
                feature_generator=generator,
                regressor=regressor,
                normaliser_type='percentile',
                monitor=Monitor(
                    monitoring_callbacks=monitoring_callbacks,
                    monitoring_images=monitoring_images,
                ),
            )

            # Train a new model
            self.it.train(
                noisy_image,
                noisy_image,
                batch_dims=noisy_metadata.batch_dim
                if noisy_metadata is not None
                else None,
            )

            # Save the trained model
            self.save_model(image_path)

            progress_callback.emit(80)

            # Predict the resulting image
            response = self.it.translate(
                noisy_image,
                batch_dims=noisy_metadata.batch_dim
                if noisy_metadata is not None
                else None,
            )

            if noisy_metadata is not None and noisy_metadata.dtype is not None:
                response = response.astype(noisy_metadata.dtype)
            else:
                response = response.astype(noisy_image.dtype)

            progress_callback.emit(100)
            return response
