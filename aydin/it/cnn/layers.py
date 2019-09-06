import numpy as np
from keras import backend as K
from keras.layers import (
    Conv2D,
    LeakyReLU,
    ZeroPadding2D,
    Cropping2D,
    Lambda,
    BatchNormalization,
    Layer,
    multiply,
    MaxPooling2D,
    Activation,
)
from aydin.it.cnn.intancenormalization import InstanceNormalization


def rot90(xx, kk=1, lyrname=None):  # rotate tensor by 90 degrees on the HW plane
    def rot(x1, k=1):
        if k < 0:
            direction = 2
        else:
            direction = 1
        for _ in range(abs(k)):
            x1 = K.reverse(K.transpose(x1), axes=direction)
            if k % 2 == 1:
                pattern = (3, 1, 2, 0)
            else:
                pattern = (0, 1, 2, 3)
        return K.permute_dimensions(x1, pattern)

    out_shape = list(K.int_shape(xx))
    if kk % 2 == 1:
        out_shape[1:3] = np.flip(out_shape[1:3], 0)
    return Lambda(lambda xx: rot(xx, kk), output_shape=out_shape[1:], name=lyrname)


def split(x, idx, batchsize=1, lyrname=None):  # split tensor at the batch axis
    out_shape = K.int_shape(x[0])
    # assert (int(out_shape / 4)) * 4 == out_shape
    # batchsize = int(out_shape / 4)
    return Lambda(
        lambda xx: xx[idx : idx + batchsize], output_shape=out_shape, name=lyrname
    )


def conv2d_bn(xx, unit, shiftconv, norm, act, lyrname=None):
    if shiftconv:
        x1 = ZeroPadding2D(((0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Conv2D(unit, (3, 3), padding='same', name=lyrname + '_cv2')(x1)
        x1 = Cropping2D(((0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = Conv2D(unit, (3, 3), padding='same', name=lyrname + '_cv2')(xx)
    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    else:
        return LeakyReLU(alpha=0.1, name=lyrname + '_lrel')(x1)


def conv2d_bn_noshift(xx, unit, norm, act, lyrname=None):
    x1 = Conv2D(unit, (1, 1), padding='same', name=lyrname + '_cv2')(xx)
    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    else:
        return LeakyReLU(alpha=0.1, name=lyrname + '_lrel')(x1)


def mxpooling_down(xx, shiftconv, lyrname=None):
    if shiftconv:
        x1 = ZeroPadding2D(((0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Cropping2D(((0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = xx
    x1 = MaxPooling2D((2, 2), name=lyrname + '_mpl')(x1)
    return x1


class Maskout(Layer):  # A layer that mutiply mask with image
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(Maskout, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(Maskout, self).build(input_shape)

    def call(self, x, training=None):
        assert isinstance(x, list)

        return K.in_train_phase(multiply(x), x[0], training=training)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a
