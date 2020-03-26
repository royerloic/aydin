import numpy
from scipy.signal import convolve

from aydin.it.it_base import ImageTranslatorBase
from aydin.util.log.log import lsection


class LucyRichardsonDeconvolution(ImageTranslatorBase):
    """
        aydin -- Lucy Richardson Deconvolution

        It's a little bit of a stretch since this is npot a 'learned' trasnlation,
        but we can certainly figure out when to stop the itarations based on a provided
        ground truth... The self-supervised casde is harder: there are no really good heuristics...

    """

    def __init__(
        self,
        psf_kernel,
        max_num_iterations=50,
        monitor=None,
        padding=True,
        padding_mode='reflect',
        clip=True,
        use_gpu=False,
    ):
        """
        Constructs a Lucy Richardson deconvolution image translator.
        :param psf_kernel: 2D or 3D kernel, dimensions should be odd numbers and numbers sum to 1
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(monitor=monitor)

        self.padding = padding
        self.padding_mode = padding_mode
        self.clip = clip
        self.use_gpu = use_gpu
        self.psf_kernel = psf_kernel.astype(numpy.float)
        self.psf_kernel_mirror = self.psf_kernel[::-1, ::-1]
        self.monitor = monitor

        self.max_num_iterations = max_num_iterations

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).
        :param path: path to save to
        """
        with lsection(f"Saving Lucy-Richardson image translator to {path}"):
            frozen = super().save(path)

        return frozen

    def _load_internals(self, path: str):
        with lsection(f"Loading Lucy-Richardson image translator from {path}"):
            # no internals to load here...
            pass

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        # nothing to do here...
        return state

    def get_receptive_field_radius(self, nb_dim):
        # max kernel dimension:
        max_kernel_dim = (max(self.psf_kernel.shape) - 1) // 2
        # Each iterations is two convolutions, which means:
        return self.max_num_iterations * max_kernel_dim * 2

    def train(
        self,
        input_image,
        target_image=None,
        batch_dims=None,
        train_valid_ratio=0.1,
        callback_period=3,
        force_jinv=False,
    ):

        super().train(
            input_image,
            target_image,
            batch_dims=batch_dims,
            train_valid_ratio=train_valid_ratio,
            callback_period=callback_period,
            force_jinv=force_jinv,
        )

    def stop_training(self):
        pass
        # we can't do that... for now...

    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        train_valid_ratio,
        callback_period,
        force_jinv,
    ):
        pass
        # we need to figure out what to do here...

    def _translate(self, input_image, batch_dims):
        """
            Internal method that translates an input image on the basis of the trained model.
        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
        """

        convolve_method = self.get_convolution_method(input_image)

        input_image = input_image.astype(numpy.float)

        candidate_deconvolved_image = numpy.full(
            input_image.shape, float(numpy.mean(input_image))
        )

        kernel_shape = self.psf_kernel.shape
        pad_width = tuple(((s - 1) // 2, (s - 1) // 2) for s in kernel_shape)

        for _ in range(self.max_num_iterations):

            if self.padding:
                padded_candidate_deconvolved_image = numpy.pad(
                    candidate_deconvolved_image,
                    pad_width=pad_width,
                    mode=self.padding_mode,
                )
            else:
                padded_candidate_deconvolved_image = candidate_deconvolved_image

            convolved = convolve_method(
                padded_candidate_deconvolved_image,
                self.psf_kernel,
                mode='valid' if self.padding else 'same',
            )

            convolved[convolved == 0] = 1

            relative_blur = input_image / convolved

            if self.padding:
                relative_blur = numpy.pad(
                    relative_blur, pad_width=pad_width, mode=self.padding_mode
                )

            multiplicative_correction = convolve_method(
                relative_blur,
                self.psf_kernel_mirror,
                mode='valid' if self.padding else 'same',
            )
            candidate_deconvolved_image *= multiplicative_correction

        if self.clip:
            candidate_deconvolved_image[candidate_deconvolved_image > 1] = 1
            candidate_deconvolved_image[candidate_deconvolved_image < -1] = -1

        return candidate_deconvolved_image

    def get_convolution_method(self, input_image):

        if self.use_gpu:
            try:
                # testing if gputools works:
                import gputools

                data = numpy.ones((30, 40, 50))
                h = numpy.ones((10, 11, 12))
                out = convolve(data, h)

                def gputools_convolve(in1, in2, mode='full', method='auto'):
                    return gputools.convolve(in1, in2)

                # gpu tools does not support padding:
                self.padding = False

                return gputools_convolve

            except:
                pass

        # this is scipy's convolve:
        return convolve
