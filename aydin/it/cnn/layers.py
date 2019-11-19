import numpy as np
from keras import backend as K
from keras.layers import (
    Conv2D,
    Conv3D,
    LeakyReLU,
    ZeroPadding2D,
    ZeroPadding3D,
    Cropping2D,
    Cropping3D,
    Lambda,
    BatchNormalization,
    Layer,
    multiply,
    MaxPooling2D,
    MaxPooling3D,
    Activation,
)
from aydin.it.cnn.instancenormalization import InstanceNormalization


def rot90(xx, kk=1, lyrname=None):  # rotate tensor by 90 degrees for 2D, 3D images
    out_shape = list(K.int_shape(xx))
    # out_shape = list(in_shape)
    if kk % 2 == 1 and 0 < kk % 6 < 5:
        out_shape[-3:-1] = np.flip(out_shape[-3:-1], 0)
    elif kk % 6 == 5 or kk % 6 == 0:
        out_shape[1:3] = np.flip(out_shape[1:3], 0)
    if len(out_shape) == 4:
        tp_axis = [0, 2, 1, 3]  # (batch, longitudinal, horizontal, channel)
    elif len(out_shape) == 5:
        tp_axis = [
            0,
            1,
            3,
            2,
            4,
        ]  # (batch, z-direction, longitudinal, horizontal, channel)
        tp_axis2 = (0, 3, 2, 1, 4)  # rotation along another axis
    else:
        raise ValueError(
            'Input shape has to be 4D or 5D. e.g. (Batch, (depth), longitudinal, horizontal, channel)'
        )

    if kk < 0:
        direction = [-2, -2, 1]
    else:
        direction = [-3, 1, -2]
    if kk % 6 == 5 and len(out_shape) == 5:
        return Lambda(
            lambda xx: K.reverse(K.permute_dimensions(xx, tp_axis2), axes=direction[1]),
            output_shape=out_shape[1:],
            name=lyrname,
        )
    elif kk % 6 == 0 and len(out_shape) == 5 and kk != 0:
        return Lambda(
            lambda xx: K.reverse(K.permute_dimensions(xx, tp_axis2), axes=direction[2]),
            output_shape=out_shape[1:],
            name=lyrname,
        )
    else:
        if kk % 4 == 1:
            return Lambda(
                lambda xx: K.reverse(
                    K.permute_dimensions(xx, tp_axis), axes=direction[0]
                ),
                output_shape=out_shape[1:],
                name=lyrname,
            )
        elif kk % 4 == 2:
            return Lambda(
                lambda xx: K.reverse(K.reverse(xx, axes=-2), axes=-3),
                output_shape=out_shape[1:],
                name=lyrname,
            )
        elif kk % 4 == 3:
            return Lambda(
                lambda xx: K.reverse(K.permute_dimensions(xx, tp_axis), axes=-2),
                output_shape=out_shape[1:],
                name=lyrname,
            )
        elif kk % 4 == 0:
            return Lambda(lambda xx: xx, output_shape=out_shape[1:], name=lyrname)


# The following code is more sofisticated and works fine with everything except serializing to JSON,
# and thus is not used anymore.
# def rot90(xx, kk=1, lyrname=None):  # rotate tensor by 90 degrees for 2D, 3D images
#     out_shape = list(K.int_shape(xx))
#
#     def rot(x1, out_shape, k=1):
#
#         if len(out_shape) == 4:
#             tp_axis = (0, 2, 1, 3)  # (batch, longitudinal, horizontal, channel)
#         elif len(out_shape) == 5:
#             tp_axis = (
#                 0,
#                 1,
#                 3,
#                 2,
#                 4,
#             )  # (batch, z-direction, longitudinal, horizontal, channel)
#             tp_axis2 = (0, 3, 2, 1, 4)  # rotation along another axis
#         else:
#             raise ValueError(
#                 'Input shape has to be 4D or 5D. e.g. (Batch, (depth), longitudinal, horizontal, channel)'
#             )
#
#         if k < 0:
#             direction = [-2, -2, 1]
#         else:
#             direction = [-3, 1, -2]
#         if k % 6 == 5 and len(out_shape) == 5:
#             x1 = K.reverse(K.permute_dimensions(x1, tp_axis2), axes=direction[1])
#         elif k % 6 == 0 and len(out_shape) == 5:
#             x1 = K.reverse(K.permute_dimensions(x1, tp_axis2), axes=direction[2])
#         elif 0 < k % 6 < 5:
#             for i in range(abs(k)):
#                 x1 = K.reverse(K.permute_dimensions(x1, tp_axis), axes=direction[0])
#         return x1
#
#     if kk % 2 == 1 and 0 < kk % 6 < 5:
#         out_shape[-3:-1] = np.flip(out_shape[-3:-1], 0)
#     elif kk % 6 == 5 or kk % 6 == 0:
#         out_shape[1:3] = np.flip(out_shape[1:3], 0)
#     return Lambda(
#         lambda xx: rot(xx, out_shape, kk), output_shape=out_shape[1:], name=lyrname
#     )


def split(x, idx, batchsize=1, lyrname=None):  # split tensor at the batch axis
    out_shape = K.int_shape(x[0])
    # assert (int(out_shape / 4)) * 4 == out_shape
    # batchsize = int(out_shape / 4)
    return Lambda(
        lambda xx: xx[idx * batchsize : (idx + 1) * batchsize],
        output_shape=out_shape,
        name=lyrname,
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


def conv3d_bn_noshift(xx, unit, norm, act, lyrname=None):
    x1 = Conv3D(unit, (1, 1, 1), padding='same', name=lyrname + '_cv')(xx)
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


def mxpooling_down3D(xx, shiftconv, lyrname=None):
    if shiftconv:
        x1 = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name=lyrname + '_0pd')(xx)
        x1 = Cropping3D(((0, 0), (0, 0), (0, 1)), name=lyrname + '_crp')(x1)
    else:
        x1 = xx
    x1 = MaxPooling3D((2, 2, 2), name=lyrname + '_mpl')(x1)
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
