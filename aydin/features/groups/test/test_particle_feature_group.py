import numpy
from skimage.exposure import rescale_intensity

from aydin.features.groups.dct import DCTFeatures
from aydin.features.groups.particle import ParticleFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32, copy=False), in_range='image', out_range=(0, 1)
    )


def test_particle_feature_group():
    # get image:
    image = n(camera().astype(numpy.float32))

    # Instantiates DCT features:
    particle = ParticleFeatures()
    assert particle.num_features(image.ndim) == 8
    assert particle.receptive_field_radius == 9 // 2

    # Set image:
    particle.prepare(image)

    # compute features and check their valididty:
    feature = numpy.empty_like(image)

    # Compute features:
    for index in range(particle.num_features(image.ndim)):
        particle.compute_feature(index=index, feature=feature)
        assert (feature != image).any()
