import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from scipy.stats import entropy

from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.cnn.util.data_util import random_sample_patches


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


# Test with garder 3D image
# Load image
def test_random_sample_patch_3D():
    image_path = examples_single.gardner_org.get_path()
    image0, metadata = io.imread(image_path)
    print(image0.shape)
    image0 = n(image0.squeeze()[:, 100:200, 400:500, 400:500])

    image0 = numpy.expand_dims(image0[1:2], -1)
    tile_size = (64, 64, 64)
    num_tile = 100
    adoption_rate = 0.2
    input_data = random_sample_patches(image0, tile_size, num_tile, adoption_rate)

    # Extract patched images
    img_patch = []
    for i in input_data:
        img_patch.append(image0[i])
    img_path = numpy.stack(img_patch)
    # Entropy of the whole image
    hist, _ = numpy.histogram(image0, range=(0, 1), bins=255, density=True)
    entropy_whole = entropy(hist)

    # Entropy of sampled areas.
    hist, _ = numpy.histogram(img_path, range=(0, 1), bins=255, density=True)
    entropy_smpl = entropy(hist)

    ent_ratio = entropy_smpl / entropy_whole

    assert ent_ratio >= 1.0


# Test with Cameraman
def test_random_sample_patch_2D():
    image0 = camera().astype(numpy.float32)
    image0 = numpy.expand_dims(numpy.expand_dims(n(image0), -1), 0)
    tile_size = (64, 64)
    num_tile = 500
    adoption_rate = 0.5
    input_data = random_sample_patches(image0, tile_size, num_tile, adoption_rate)

    # Extract patched images
    img_patch = []
    for i in input_data:
        img_patch.append(image0[i])
    img_path = numpy.stack(img_patch)
    # Entropy of the whole image
    hist, _ = numpy.histogram(image0, range=(0, 1), bins=255, density=True)
    entropy_whole = entropy(hist)

    # Entropy of sampled areas.
    hist, _ = numpy.histogram(img_path, range=(0, 1), bins=255, density=True)
    entropy_smpl = entropy(hist)

    ent_ratio = entropy_smpl / entropy_whole

    assert ent_ratio >= 1.01
