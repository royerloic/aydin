import numpy
from scipy.stats import entropy


def random_sample_patches(input_img, patch_size, num_patch, adoption_rate=0.5):
    """
    This function outputs a list of slices that crops a part of the input_img (i.e. patch).
    Only patches with higher entropy in their intensity histogram are selected.
    :param input_size: input_img: input image that will be sampled with patches
    :param patch_size: patch size
    :param num_patch: number of patches to be output
    :param adaption_rate: The % of patches selected from the original population of patches.
    """
    if type(patch_size) == int:
        if len(input_img.shape) == 4:
            patch_size = (patch_size, patch_size)
        if len(input_img.shape) == 5:
            patch_size = (patch_size, patch_size, patch_size)

    img_dim = input_img.shape
    tiles_per_img = numpy.ceil(
        numpy.ceil(num_patch / img_dim[0]) / adoption_rate
    ).astype(int)
    hist_ind_all = []
    coordinates = numpy.asarray(img_dim[1:-1]) - numpy.asarray(patch_size)
    for k in range(img_dim[0]):
        ind_past = []
        hist_batch = []
        while len(hist_batch) < tiles_per_img:
            # Randomly choose coordinates from an image.
            ind = numpy.hstack(
                [k]
                + [
                    numpy.random.choice(coordinates[i], 1, replace=True).astype(int)
                    for i in range(coordinates.size)
                ]
            )
            # Check if the new patch is too close to the existing patches.
            if ind_past:
                if abs(ind_past - ind).max() <= coordinates.min() // 20:
                    continue
            ind_past.append(ind)
            # Extract image patch from the input image.
            if len(patch_size) == 2:
                img0 = input_img[
                    ind[0],
                    ind[1] : ind[1] + patch_size[0],
                    ind[2] : ind[2] + patch_size[1],
                    0,
                ]
            elif len(patch_size) == 3:
                img0 = input_img[
                    ind[0],
                    ind[1] : ind[1] + patch_size[0],
                    ind[2] : ind[2] + patch_size[1],
                    ind[3] : ind[3] + patch_size[2],
                    0,
                ]
            else:
                raise ValueError('Only 2D or 3D tile are applicable.')
            # Calculate histogram and entropy
            hist, _ = numpy.histogram(img0, range=(0, 1), bins=255, density=True)
            hist_batch.append(entropy(hist))
        # Create a table with entropy and indices of each patch.
        hist_ind = numpy.hstack((numpy.vstack(hist_batch), ind_past))
        # Sort by entropy.
        hist_ind = hist_ind[(-hist_ind[:, 0]).argsort()]
        # Only leave the highest `adoption_rate` of patches.
        hist_ind = hist_ind[: int(hist_ind.shape[0] * adoption_rate), ...]
        hist_ind_all.append(hist_ind)
    hist_ind_all = numpy.vstack(hist_ind_all)
    hist_ind_all = hist_ind_all[(-hist_ind_all[:, 0]).argsort()]
    hist_ind_all = hist_ind_all[:num_patch, 1:].astype(int)

    # Create a slice list
    patch_slice = []
    for ind in hist_ind_all:
        slice_list = (
            [slice(ind[0], ind[0] + 1, 1)]
            + [
                slice(ind[i + 1], ind[i + 1] + patch_size[i], 1)
                for i in range(len(patch_size))
            ]
            + [slice(0, 1, 1)]
        )
        patch_slice.append(tuple(slice_list))

    # Return a list of slice
    return patch_slice


def sim_model_size(input_size, shiftconv=True, floattype=32):
    """
    Estimate model size for checking if memory is enough to run the training.
    :param input_size: input image size; exclude batch and channel dim
    :param shiftconv: whether shift convolution model is going to be used.
    :param floattype: type of floating number
    :return: estimated memory to be ocupied by the CNN model in byte
    """
    # input_size should not contain channel dim.
    byte = floattype / 8
    input_size = numpy.array(input_size)
    if shiftconv:
        tensor_size = numpy.prod(input_size) * 6
        tensor_size += numpy.prod(numpy.append(input_size, 48)) * 5
        tensor_size += numpy.prod(numpy.append(input_size / 2, 48)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 2, 96)) * 5
        tensor_size += numpy.prod(numpy.append(input_size / 4, 96)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 4, 144)) * 5
        tensor_size += numpy.prod(numpy.append(input_size / 8, 144)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 8, 192)) * 5
        tensor_size += numpy.prod(numpy.append(input_size / 16, 192)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 16, 240)) * 5
        tensor_size += numpy.prod(numpy.append(input_size / 32, 240)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 32, 48)) * 3
        tensor_size += numpy.prod(numpy.append(input_size / 16, 48)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 16, 240)) * 5
        tensor_size += numpy.prod(numpy.append(input_size / 8, 240)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 8, 384)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 8, 192)) * 3
        tensor_size += numpy.prod(numpy.append(input_size / 4, 192)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 4, 288)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 4, 144)) * 3
        tensor_size += numpy.prod(numpy.append(input_size / 2, 144)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 2, 192)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 4, 96)) * 3
        tensor_size += numpy.prod(numpy.append(input_size, 97)) * 3
        tensor_size += numpy.prod(numpy.append(input_size, 48)) * 12
        tensor_size += numpy.prod(numpy.append(input_size, 192)) * 1
        tensor_size += numpy.prod(numpy.append(input_size, 384)) * 2
        tensor_size += numpy.prod(numpy.append(input_size, 48)) * 2
        tensor_size += numpy.prod(input_size) * 1
        model_params = 2790673
        return (tensor_size * 4 + model_params) * byte * 1.021
    else:
        tensor_size = numpy.prod(input_size)
        tensor_size += numpy.prod(numpy.append(input_size, 48)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 2, 48)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 2, 96)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 4, 96)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 4, 144)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 8, 144)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 8, 192)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 16, 192)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 16, 240)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 32, 240)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 32, 48)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 16, 48)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 16, 240)) * 3
        tensor_size += numpy.prod(numpy.append(input_size / 8, 240)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 8, 384)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 8, 192)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 4, 192)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 4, 288)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 4, 144)) * 2
        tensor_size += numpy.prod(numpy.append(input_size / 2, 144)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 2, 192)) * 1
        tensor_size += numpy.prod(numpy.append(input_size / 2, 96)) * 2
        tensor_size += numpy.prod(numpy.append(input_size, 96)) * 1
        tensor_size += numpy.prod(numpy.append(input_size, 97)) * 1
        tensor_size += numpy.prod(numpy.append(input_size, 48)) * 2
        tensor_size += numpy.prod(input_size) * 3
        model_params = 2698081
        return (tensor_size + model_params) * byte
