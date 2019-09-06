import numpy
import numpy.fft as fft


def resolution(image):

    nb_dim = len(image.shape)

    for axis in range(0, nb_dim):

        projection = numpy.sum(image, axis=axis)[image.shape[axis] // 2]

        dft = fft.fft(projection)

        dft_mod = numpy.absolute(dft)
        dft_mod_log = numpy.log1p(dft_mod)
