import numpy
from numpy.linalg import norm
from scipy.fft import dct
from skimage.measure import compare_mse


# Interesting: https://github.com/andrewekhalel/sewar


def spectral_psnr(norm_true_image, norm_test_image):
    norm_true_image = norm_true_image / norm(norm_true_image.flatten(), 2)
    norm_test_image = norm_test_image / norm(norm_test_image.flatten(), 2)

    dct_norm_true_image = dct(dct(norm_true_image, axis=0), axis=1)
    dct_norm_test_image = dct(dct(norm_test_image, axis=0), axis=1)

    norm_dct_norm_true_image = dct_norm_true_image / norm(
        dct_norm_true_image.flatten(), 2
    )
    norm_dct_norm_test_image = dct_norm_test_image / norm(
        dct_norm_test_image.flatten(), 2
    )

    norm_true_image = abs(norm_dct_norm_true_image)
    norm_test_image = abs(norm_dct_norm_test_image)

    err = compare_mse(norm_true_image, norm_test_image)
    return 10 * numpy.log10(1 / err)


def spectral_mutual_information(image_a, image_b, normalised=True):
    norm_image_a = image_a / norm(image_a.flatten(), 2)
    norm_image_b = image_b / norm(image_b.flatten(), 2)

    dct_norm_true_image = dct(dct(norm_image_a, axis=0), axis=1)
    dct_norm_test_image = dct(dct(norm_image_b, axis=0), axis=1)

    return mutual_information(
        dct_norm_true_image, dct_norm_test_image, normalised=normalised
    )


def joint_information(image_a, image_b, bins=256):
    image_a = image_a.flatten()
    image_b = image_b.flatten()

    c_xy = numpy.histogram2d(image_a, image_b, bins)[0]
    ji = joint_entropy_from_contingency(c_xy)
    return ji


def mutual_information(image_a, image_b, bins=256, normalised=True):
    image_a = image_a.flatten()
    image_b = image_b.flatten()

    c_xy = numpy.histogram2d(image_a, image_b, bins)[0]
    mi = mutual_info_from_contingency(c_xy)
    mi = mi / joint_entropy_from_contingency(c_xy) if normalised else mi
    return mi


def joint_entropy_from_contingency(contingency):

    # cordinates of non-zero entries in contingency table:
    nzx, nzy = numpy.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingnecy table:
    contingency_sum = contingency.sum()

    # normalised contingency, i.e. probability:
    p = nz_val / contingency_sum

    # log contingency:
    log_p = numpy.log2(p)

    # Joint entropy:
    joint_entropy = -p * log_p

    return joint_entropy.sum()


def mutual_info_from_contingency(contingency):

    # cordinates of non-zero entries in contingency table:
    nzx, nzy = numpy.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingnecy table:
    contingency_sum = contingency.sum()

    # marginals:
    pi = numpy.ravel(contingency.sum(axis=1))
    pj = numpy.ravel(contingency.sum(axis=0))

    #
    log_contingency_nm = numpy.log2(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(numpy.int64, copy=False) * pj.take(nzy).astype(
        numpy.int64, copy=False
    )
    log_outer = -numpy.log2(outer) + numpy.log2(pi.sum()) + numpy.log2(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - numpy.log2(contingency_sum))
        + contingency_nm * log_outer
    )
    return mi.sum()
