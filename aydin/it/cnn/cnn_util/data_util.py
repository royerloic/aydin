import numpy
from scipy.stats import entropy


def random_sample_patches(input_img, tile_size, num_tile, adoption_rate=0.5):
    if type(tile_size) == int:
        if len(input_img.shape) == 4:
            tile_size = (tile_size, tile_size)
        if len(input_img.shape) == 5:
            tile_size = (tile_size, tile_size, tile_size)

    img_dim = input_img.shape
    tiles_per_img = numpy.ceil(
        numpy.ceil(num_tile / img_dim[0]) / adoption_rate
    ).astype(int)
    hist_ind_all = []
    coordinates = numpy.asarray(img_dim[1:-1]) - numpy.asarray(tile_size)
    for k in range(img_dim[0]):
        ind_past = []
        hist_batch = []
        for _ in range(tiles_per_img):
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
            if len(tile_size) == 2:
                img0 = input_img[
                    ind[0],
                    ind[1] : ind[1] + tile_size[0],
                    ind[2] : ind[2] + tile_size[1],
                    0,
                ]
            elif len(tile_size) == 3:
                img0 = input_img[
                    ind[0],
                    ind[1] : ind[1] + tile_size[0],
                    ind[2] : ind[2] + tile_size[1],
                    ind[3] : ind[3] + tile_size[2],
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
    hist_ind_all = hist_ind_all[:num_tile, 1:].astype(int)

    # Return a numpy array of coordinates of patches.
    return hist_ind_all


def model_memsize(model):
    total_par = model.count_params()
    tensor_size = 0
    for lyr in model.layers:
        tensor_size += numpy.array(numpy.prod(lyr.output_shape[1:]))
    byte = int(str(lyr.output.shape.dtype).split(".")[1][-2:]) / 8
    return (total_par + tensor_size) * byte


def sim_model_size(input_size, shiftconv=True, floattype=32):
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
        return (tensor_size + model_params) * byte * 1.021
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
