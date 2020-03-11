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
from tensorflow_core.python.keras.layers.pooling import MaxPooling2D, MaxPooling3D

from aydin.it.cnn.layers.instance_norm import InstanceNormalization
from aydin.it.cnn.layers.util import Swish


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
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
    else:
        return LeakyReLU(alpha=0.1, name=lyrname + '_lrel')(x1)


def conv3d_bn(xx, unit, shiftconv, norm, act, lyrname=None):
    if shiftconv:
        x1 = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Conv3D(unit, (3, 3, 3), padding='same', name=lyrname + '_cv')(x1)
        x1 = Cropping3D(((0, 0), (0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = Conv3D(unit, (3, 3, 3), padding='same', name=lyrname + '_cv')(xx)
    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
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
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
    else:
        return LeakyReLU(alpha=0.1, name=lyrname + '_lrel')(x1)


def conv3d_bn_noshift(xx, unit, norm, act, lyrname=None):
    x1 = Conv3D(unit, (1, 1, 1), padding='same', name=lyrname + '_cv')(xx)
    if norm == 'instance':
        x1 = InstanceNormalization(name=lyrname + '_in')(x1)
    elif norm == 'batch':
        x1 = BatchNormalization(name=lyrname + '_bn')(x1)
    if act == 'ReLU':
        return Activation('relu', name=lyrname + '_relu')(x1)
    elif act == 'swish':
        return Swish(name=lyrname + '_swsh')(x1)
    else:
        return LeakyReLU(alpha=0.1, name=lyrname + '_lrel')(x1)


def mxpooling_down2D(xx, shiftconv, lyrname=None):
    if shiftconv:
        x1 = ZeroPadding2D(((0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Cropping2D(((0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = xx
    x1 = MaxPooling2D((2, 2), name=lyrname + '_mpl')(x1)
    return x1


def mxpooling_down3D(xx, shiftconv, pool_size=(2, 2, 2), lyrname=None):
    if shiftconv:
        x1 = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Cropping3D(((0, 0), (0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = xx
    x1 = MaxPooling3D(pool_size, name=lyrname + '_mpl')(x1)
    return x1
