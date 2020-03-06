import numpy as np
import tensorflow as tf

K = tf.keras.backend
Lambda = tf.keras.layers.Lambda


def Swish(name=None):
    return Lambda(tf.nn.swish, name=name)


def rot90(xx, kk=1, lyrname=None):
    """
    Rotate tensor by 90 degrees for 2D, 3D images. Only for shift convolution architecture.
    :param xx: input tensor from previous layer
    :param kk: index for rotation (crock wise).
    """
    out_shape = list(K.int_shape(xx))
    if kk % 2 == 1 and 0 < kk % 6 < 5:
        out_shape[-3:-1] = np.flip(out_shape[-3:-1], 0)
    elif abs(kk) % 6 == 5 or kk % 6 == 0:
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
        direction = [-2, -3, -2, 1]
    else:
        direction = [-3, -2, 1, -2]
    if abs(kk) % 6 == 5 and len(out_shape) == 5:
        return Lambda(
            lambda xx: K.reverse(K.permute_dimensions(xx, tp_axis2), axes=direction[2]),
            output_shape=out_shape[1:],
            name=lyrname,
        )
    elif kk % 6 == 0 and len(out_shape) == 5 and kk != 0:
        return Lambda(
            lambda xx: K.reverse(K.permute_dimensions(xx, tp_axis2), axes=direction[3]),
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
                lambda xx: K.reverse(
                    K.permute_dimensions(xx, tp_axis), axes=direction[1]
                ),
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
