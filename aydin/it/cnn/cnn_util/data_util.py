import numpy
from scipy.stats import entropy


def random_sample_patches(input_img, tile_size, num_tile, adoption_rate=0.5):
    img_dim = input_img.shape
    tiles_per_img = numpy.ceil(
        numpy.ceil(num_tile / img_dim[0]) / adoption_rate
    ).astype(int)
    hist_ind_all = []
    for k in range(img_dim[0]):
        ind_past = []
        hist_batch = []
        for j in range(tiles_per_img):
            coordinates = numpy.asarray(img_dim[1:-1]) - numpy.asarray(tile_size)
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
                if abs(ind_past - ind).max() <= coordinates.min() // 10:
                    continue
            # Extract image patch from the input image.
            ind_past.append(ind)
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

    # Return a numpy array of cordinates of patches.
    return hist_ind_all
