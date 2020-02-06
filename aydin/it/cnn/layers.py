import copy
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
    InputSpec,
)

# from aydin.it.cnn.instancenormalization import InstanceNormalization
from keras import initializers, regularizers, constraints


def rot90(xx, kk=1, lyrname=None):  # rotate tensor by 90 degrees for 2D, 3D images
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


def split(x, idx, batchsize=1, lyrname=None):  # split tensor at the batch axis
    out_shape = K.int_shape(x[0])
    # assert (int(out_shape / 4)) * 4 == out_shape
    # batchsize = int(out_shape / 4)
    return Lambda(
        lambda xx: xx[idx * batchsize : (idx + 1) * batchsize],
        output_shape=out_shape,
        name=lyrname,
    )


class Split1(Layer):
    def __init__(self, idx, batchsize=1, **kwargs):
        self.idx = idx
        self.batchsize = batchsize
        super(Split1, self).__init__(**kwargs)

    def call(self, x):
        shape = K.int_shape(x)
        starts = np.zeros(len(shape)).astype(int)
        starts[0] = self.idx * self.batchsize
        # return K.slice(x, starts, (self.batchsize,) + shape[1:])
        return x[self.idx * self.batchsize : (self.idx + 1) * self.batchsize]

    def compute_output_shape(self, input_shape):
        return input_shape


class Split(Layer):
    def __init__(self, number_of_splits, batchsize=1, axis=0, **kwargs):
        self.number_of_splits = number_of_splits
        self.axis = axis
        self.pattern = None
        self.batchsize = batchsize
        super(Split, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) > self.axis
        if input_shape[self.axis] is not None:
            assert input_shape[self.axis] % self.number_of_splits == 0

        # create a pattern to indicate the axis for split
        pattern = np.arange(len(input_shape))
        pattern[0] = self.axis
        pattern[self.axis] = 0
        self.pattern = pattern
        super(Split, self).build(input_shape)

    def call(self, x):
        shape = K.int_shape(x)
        if shape[self.axis] is not None:
            self.batchsize = int(shape[self.axis] / self.number_of_splits)
        x = K.permute_dimensions(x, self.pattern)
        x_list = [
            x[i * self.batchsize : (i + 1) * self.batchsize, ...]
            for i in range(self.number_of_splits)
        ]
        return [K.permute_dimensions(x, self.pattern) for x in x_list]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if input_shape[self.axis] is not None:
            output_shape[self.axis] = self.batchsize
        return [output_shape for _ in range(self.number_of_splits)]


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


def masker(batch_vol, i, mask_shape):
    i = i % np.prod(mask_shape)
    mask = np.zeros(np.prod(mask_shape), dtype=bool)
    mask[i] = True
    mask = mask.reshape(mask_shape)
    rep = np.ceil(np.asarray(batch_vol) / np.asarray(mask_shape)).astype(int)
    mask = np.tile(mask, tuple(rep))
    mask = mask[: batch_vol[0], : batch_vol[1]]
    return mask


def maskedgen(batch_vol, mask_shape, image, batch_size):
    while True:
        for j in range(np.ceil(image.shape[0] / batch_size).astype(int)):
            image_batch = image[j * batch_size : (j + 1) * batch_size, ...]
            for i in range(np.prod(mask_shape).astype(int)):
                mask = masker(batch_vol, i, mask_shape)
                masknega = np.broadcast_to(
                    np.expand_dims(np.expand_dims(mask, 0), -1), image_batch.shape
                )
                train_img = (
                    np.broadcast_to(
                        np.expand_dims(np.expand_dims(~mask, 0), -1), image_batch.shape
                    )
                    * image_batch
                )
                target_img = masknega * image_batch
                yield {
                    'input': train_img,
                    'input_msk': masknega.astype(np.float32),
                }, target_img


class InstanceNormalization(Layer):
    """Instance normalization layer.

    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.

    # Output shape
        Same shape as input.

    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(
        self,
        axis=None,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name='gamma',
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name='beta',
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
