from copy import deepcopy
import numpy as np
from scipy.ndimage import median_filter

from aydin.util.log.log import lprint


def masker(batch_vol, i=None, mask_shape=None, p=None):
    """
    Single mask generator.
    :param batch_vol: batch volume; desn't include batch and ch dimensions
    :param mask_shape: mask shape e.g. (3, 3)
    :param p: possibility of masked pixels on random masking approach
    """
    if p:
        mask = np.random.uniform(size=batch_vol)
        mask[mask > p] = 1
        mask[mask <= p] = 0
        mask = mask.astype(bool)
    else:
        i = i % np.prod(mask_shape)
        mask = np.zeros(np.prod(mask_shape), dtype=bool)
        mask[i] = True
        mask = mask.reshape(mask_shape)
        rep = np.ceil(np.asarray(batch_vol) / np.asarray(mask_shape)).astype(int)
        mask = np.tile(mask, tuple(rep))
        ind = tuple([slice(batch_vol[i]) for i in range(len(batch_vol))])
        mask = mask[ind]
    return mask


def med_filter(input_image):
    k = [1] + [3 for _ in input_image.shape[1:-1]] + [1]
    kernel = np.ones(k)
    k = [0] + [1 for _ in range(len(input_image.shape[1:-1]))] + [0]
    kernel[tuple(k)] = 0
    return median_filter(input_image, footprint=kernel)


def maskedgen(
    batch_vol,
    mask_shape,
    image,
    batch_size,
    train_valid_ratio=0,
    replace_by='zero',
    subset='training',
):
    """
    Mask generator. Returns a generator.
    :param batch_vol: batch volume; desn't include batch and ch dimensions
    :param mask_shape: mask shape e.g. (3, 3)
    :param image: input image
    :param batch_size: batch size
    :param p: possibility of masked pixels on random masking approach
    :param replace_by: Masked pixels are replaced by random number or 0 or median values.
    :param train_valid_ratio: ratio of the data will be used for validation
    """

    val_ind = np.zeros(image.shape[0], dtype=bool)
    num_val = (
        np.floor(image.shape[0] * train_valid_ratio / batch_size).astype(int)
        * batch_size
    )
    val_ind[:num_val] = True
    np.random.shuffle(val_ind)  # indices for validation slices

    if 'train' in subset:
        img_output = deepcopy(image[~val_ind])
    else:
        img_output = deepcopy(image[val_ind])

    j = 0
    num_cycle = np.ceil(img_output.shape[0] / batch_size)
    lprint(f'Masked pixels are replaced by {replace_by}')
    while True:
        i = np.mod(j, num_cycle).astype(int)
        image_batch = img_output[batch_size * i : batch_size * (i + 1)]
        for i in range(np.prod(mask_shape).astype(int)):
            mask = masker(
                batch_vol, i, mask_shape, p=None
            )  # generate a 2D mask with same size of image_batch; True value is masked.
            masknega = np.broadcast_to(
                np.expand_dims(np.expand_dims(mask, 0), -1), image_batch.shape
            )  # broadcast the 2D mask to batch dimension ready for multiplication
            train_img = (
                np.broadcast_to(
                    np.expand_dims(np.expand_dims(~mask, 0), -1), image_batch.shape
                )
                * image_batch
            )  # pixels in training image are blocked by multiply by 0

            if replace_by == 'random':
                train_img = train_img + np.random.random(batch_vol) * masknega
            elif replace_by == 'median':
                train_img = train_img + med_filter(image_batch) * masknega

            target_img = masknega * image_batch
            j += 1
            yield {
                'input': train_img,
                'input_msk': masknega.astype(np.float32),
            }, target_img


def randmaskgen(
    image,
    batch_size,
    p=None,
    train_valid_ratio=0,
    replace_by='zero',  # 'zero' or 'random' or 'median'
    subset='training',
    reduce_p=False,
    p_reduce_rate=0.5,
    p_reduce_patience=50,
):
    """
    Mask generator. Returns a generator.
    :param image: input image
    :param batch_size: batch size
    :param replace_by: Masked pixels are replaced by random number or 0 or median values.
    :param p: possibility of masked pixels on random masking approach
    :param reduce_p: if True, p will dynamically change
    :param p_reduce_rate: reduction rate of p
    :param p_reduce_patience: number of epochs before reducing p
    """

    batch_vol = (batch_size,) + image.shape[1:]

    val_ind = np.zeros(image.shape[0], dtype=bool)
    num_val = (
        np.floor(image.shape[0] * train_valid_ratio / batch_size).astype(int)
        * batch_size
    )
    val_ind[:num_val] = True
    np.random.shuffle(val_ind)  # indices for validation slices

    if 'train' in subset:
        img_output = deepcopy(image[~val_ind])
    else:
        img_output = deepcopy(image[val_ind])

    j = 0
    num_cycle = np.ceil(img_output.shape[0] / batch_size)
    lprint(f'Masked pixels are replaced by {replace_by}')
    while True:
        i = np.mod(j, num_cycle).astype(int)
        image_batch = img_output[batch_size * i : batch_size * (i + 1)]

        if reduce_p:
            if j % p_reduce_patience == 0 and j != 0:
                p *= p_reduce_rate
                lprint(f'p is reduced to {p}')
        mask = masker(
            batch_vol, p=p
        )  # generate a 2D mask with same size of image_batch; p of the pix are 0
        train_img = (
            mask * image_batch
        )  # pixels in training image are blocked by multiply by 0
        masknega = ~mask  # p of the pixels are 1

        if replace_by == 'random':
            train_img = train_img + np.random.random(batch_vol) * masknega
        elif replace_by == 'median':
            train_img = train_img + med_filter(image_batch) * masknega

        target_img = masknega * image_batch
        j += 1

        yield {'input': train_img, 'input_msk': masknega.astype(np.float32)}, target_img
