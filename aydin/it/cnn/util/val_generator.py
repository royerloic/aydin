import numpy
from copy import deepcopy
from scipy.ndimage.filters import convolve
import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


def mean_filter(input_image):
    k = [1] + [3 for _ in input_image.shape[1:-1]] + [1]
    kernel = numpy.ones(k) / 8
    k = [0] + [1 for _ in range(len(input_image.shape[1:-1]))] + [0]
    kernel[tuple(k)] = 0
    return convolve(input_image, kernel)


def val_img_generator(input_image, p=0.1):
    '''
    :param input_image: input image
    :param p: ratio of pixels being used for validation
    :return: training image: p pixels were repalced by surrounding average, validation image: original image, mask: markers of validation pixels
    '''
    marker = numpy.random.uniform(size=input_image.shape)
    marker[marker > p] = 1
    marker[marker <= p] = 0
    marker = (1 - marker).astype(bool)

    img_filtered = mean_filter(input_image)
    img_train = deepcopy(input_image)
    img_train[marker] = img_filtered[marker]
    return img_train, input_image, marker


def val_data_generator(img_train, img_val, marker, batch_size, train_valid_ratio):
    val_ind = numpy.random.randint(
        0, img_train.shape[0], int(img_train.shape[0] * train_valid_ratio)
    )
    datagen_train = ImageDataGenerator().flow(
        img_train[val_ind], batch_size=batch_size, shuffle=True
    )
    datagen_val = ImageDataGenerator().flow(
        img_val[val_ind], batch_size=batch_size, shuffle=True
    )
    datagen_marker = ImageDataGenerator().flow(
        ~marker[val_ind], batch_size=batch_size, shuffle=True
    )

    while True:
        train_batch = datagen_train.next()
        val_batch = datagen_val.next()
        marker_batch = datagen_marker.next()
        yield {
            'input': train_batch,
            'input_msk': marker_batch.astype(numpy.float32),
        }, val_batch


# may be useful in the future for generating validation data with less memory used.
def val_ind_generator(image_train, p=0.1):
    '''
    :param image_train: input image
    :param p: ratio of pixels being used for validation
    :return: training image: p pixels were repalced by surrounding average, val_pix_values: original image, mask: markers of validation pixels
    '''
    marker = numpy.random.randint(0, image_train.size, size=int(image_train.size * p))
    marker = numpy.unravel_index(marker, image_train.shape)
    val_pix_values = image_train[marker]
    img_filtered = mean_filter(image_train)
    image_train[marker] = img_filtered[marker]
    return image_train, val_pix_values, marker


def val_data_generator_from_ind(
    img_train, img_val, marker, batch_size, train_valid_ratio
):
    val_ind = numpy.random.randint(
        0, img_train.shape[0], int(img_train.shape[0] * train_valid_ratio)
    )
    datagen_train = ImageDataGenerator().flow(
        img_train[val_ind], batch_size=batch_size, shuffle=False
    )
    datagen_val = ImageDataGenerator().flow(
        img_val[val_ind], batch_size=batch_size, shuffle=False
    )
    datagen_marker = ImageDataGenerator().flow(
        ~marker[val_ind], batch_size=batch_size, shuffle=False
    )

    while True:
        train_batch = datagen_train.next()
        val_batch = datagen_val.next()
        marker_batch = datagen_marker.next()
        yield {
            'input': train_batch,
            'input_msk': marker_batch.astype(numpy.float32),
        }, val_batch
