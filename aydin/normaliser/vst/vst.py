import numpy
from scipy.interpolate import UnivariateSpline


def smooth_func(x, y, method='spline'):
    """
    return a smoothed function of y(x) using spline

    :param x:
    :param y:
    :param method:
    :return:
    """
    w = numpy.isnan(y)
    y[w] = 0
    if method == 'spline':
        fit = UnivariateSpline(x, y, w=~w, s=0.2)
    return fit


def vst_classical(image, intensity, resol=1000, method='FreemanTukey'):
    """
    return Variance stable transformation using classical methods for Poisson distribution

    :param resol:
    :return:
    """
    imax = image.max()
    di = imax / resol
    intens = (0.5 + numpy.arange(resol)) * di
    if method == 'FreemanTukey':
        vst = numpy.sqrt(intensity * intens) + numpy.sqrt(intensity * intens + 1)
    if method == 'Anscombe':
        vst = 2 * numpy.sqrt(intensity * intens + 0.375)
    return intens, vst


def vst_adaptive(image, denoised, resol=1000):
    """
    return the transformation of intensity that recovers the stable variance

    :param image:
    :param denoised:
    :param resol:
    :return:
    """
    imax = denoised.max()
    imin = denoised.min()
    di = imax / resol
    intens = (0.5 + numpy.arange(resol)) * di
    (x, y) = var_intensity(image, denoised)
    yx = smooth_func(x, y)
    vx = yx(intens)
    vx[vx < imin] = imin
    vprime = 1.0 / numpy.sqrt(vx)
    vprime[0] /= 2
    vst = numpy.cumsum(vprime * di)
    return intens, vst


def var_intensity(image, denoised, nbins=50):
    """
    return the variance of intensity noise versus intensity

    :param image:
    :return: variance of intensity curve
    """
    imax = denoised.max()
    imin = denoised.min()

    intens = (0.5 + numpy.arange(nbins)) * (imax - imin) / nbins
    indx = numpy.digitize(
        numpy.reshape(denoised, [-1]),
        imin + numpy.arange(nbins + 1) * (imax - imin) / nbins,
    )
    pvar = numpy.zeros((nbins,))
    for i in range(nbins):
        pvar[i] = numpy.var(
            numpy.reshape(image, [-1])[indx == i + 1]
            - numpy.reshape(denoised, [-1])[indx == i + 1]
        )
    return intens, pvar


def vst_transform(image, x, y, rescale=False):
    """
    Implement VST to the image
    the inverse transform can be easily implemented by flip x, y in the input

    :param image:
    :param x:
    :param y:
    :param rescale:
    :return:
    """
    image_shape = image.shape
    # dx = x[1]-x[0]
    # xmin  = x[0]
    resol = len(x)
    indx = numpy.digitize(numpy.reshape(image, [-1]), x)
    image_sc = numpy.zeros((numpy.prod(image_shape),))
    image_sc[indx == 0] = y[0]
    for i in range(resol):
        image_sc[indx == (i + 1)] = y[i]
    if rescale:
        return image_sc / y[-1]
    return numpy.reshape(image_sc, image_shape)
