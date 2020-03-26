from tensorflow_core.python.keras.layers import AveragePooling3D
from tensorflow_core.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow_core.python.keras.layers.convolutional import (
    ZeroPadding2D,
    Conv2D,
    Cropping2D,
    ZeroPadding3D,
    Conv3D,
    Cropping3D,
)
from tensorflow_core.python.keras.layers.core import Activation
from tensorflow_core.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow_core.python.keras.layers.pooling import (
    MaxPooling2D,
    MaxPooling3D,
    AveragePooling2D,
)
from tensorflow_core.python.keras.regularizers import l1

from aydin.it.cnn.layers.instance_norm import InstanceNormalization
from aydin.it.cnn.layers.util import Swish


def conv2d_bn(
    xx,
    unit,
    kernel_size=3,
    shiftconv=False,
    norm=None,
    act='ReLU',
    weight_decay=0,
    lyrname=None,
):
    if shiftconv:
        x1 = ZeroPadding2D(((0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Conv2D(unit, (3, 3), padding='same', name=lyrname + '_cv2')(x1)
        x1 = Cropping2D(((0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = Conv2D(
            unit,
            kernel_size,
            padding='same',
            kernel_regularizer=l1(weight_decay),
            bias_regularizer=l1(weight_decay),
            name=lyrname + '_cv2',
        )(xx)

    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
    elif act == 'lrel':
        return LeakyReLU(alpha=0.1, name=lyrname + '_lrel')(x1)
    else:
        return x1


def conv3d_bn(
    xx,
    unit,
    kernel_size=3,
    shiftconv=False,
    norm=None,
    act='ReLU',
    weight_decay=0,
    lyrname=None,
):
    if shiftconv:
        x1 = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Conv3D(unit, (3, 3, 3), padding='same', name=lyrname + '_cv')(x1)
        x1 = Cropping3D(((0, 0), (0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = Conv3D(
            unit,
            kernel_size,
            padding='same',
            kernel_regularizer=l1(weight_decay),
            bias_regularizer=l1(weight_decay),
            name=lyrname + '_cv3',
        )(xx)
    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
    elif act == 'lrel':
        return LeakyReLU(alpha=0.1, name=lyrname + '_lrel')(x1)
    else:
        return x1


def pooling_down2D(xx, shiftconv, mode='max', lyrname=None):
    if shiftconv:
        x1 = ZeroPadding2D(((0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Cropping2D(((0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = xx
    if mode == 'ave':
        x1 = AveragePooling2D((2, 2), name=lyrname + '_apl')(x1)
    elif mode == 'max':
        x1 = MaxPooling2D((2, 2), name=lyrname + '_mpl')(x1)
    else:
        raise ValueError('pooling mode only accepts "max" or "ave".')
    return x1


def pooling_down3D(xx, shiftconv, pool_size=(2, 2, 2), mode='max', lyrname=None):
    if shiftconv:
        x1 = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Cropping3D(((0, 0), (0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = xx
    if mode == 'ave':
        x1 = AveragePooling3D(pool_size, name=lyrname + '_apl')(x1)
    elif mode == 'max':
        x1 = MaxPooling3D(pool_size, name=lyrname + '_mpl')(x1)
    else:
        raise ValueError('pooling mode only accepts "max" or "ave".')
    return x1
