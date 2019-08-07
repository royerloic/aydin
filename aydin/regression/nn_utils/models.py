from keras import layers
from keras.initializers import VarianceScaling, RandomNormal
from keras.layers import Input, Dense, LeakyReLU, PReLU, ReLU, GaussianNoise
from keras.models import Model


def block(x, outputs=1, layers=1, layer_name=None, trainable=True, initialiser=None):

    for i in range(0, layers):

        x = Dense(
            outputs,
            name=layer_name + 'd1l' + str(i),
            trainable=trainable,
            use_bias=False,
            kernel_initializer='glorot_uniform' if initialiser is None else initialiser,
            bias_initializer='zeros',
        )(x)
        # x = BatchNormalization(name=layer_name + 'bn1', center=True, scale=False)(x)
        x = LeakyReLU(name=layer_name + 'act1l' + str(i))(x)

    return x


def yinyang(feature_dim, depth=16):
    input_feature = Input(shape=(feature_dim,), name='input')

    width = feature_dim

    x = input_feature
    outputs = []
    outputs.append(x)

    for d in range(0, depth):
        if d != 0:
            u = layers.concatenate([input_feature, x])
        else:
            u = x
        x = block(u, outputs=width, layer_name=f'fc{d}')
        outputs.append(x)

    x = layers.add([y for y in outputs])

    x = Dense(1, name='fc_last')(x)

    model = Model(input_feature, x)

    return model


def feed_forward_width(feature_dim, width=None, depth=16):

    if not width:
        width = feature_dim

    initialiser = 'glorot_uniform'  # RandomNormal(stddev = 0.1)

    input = Input(shape=(feature_dim,), name='input')

    if width <= feature_dim:
        x = input
    else:
        x = Dense(
            width - feature_dim,
            name='fc_first',
            kernel_initializer=initialiser,
            trainable=True,
        )(input)
        x = layers.concatenate([input, x])

    outputs = []
    outputs.append(x)

    for d in range(0, depth):
        x = block(x, outputs=width, layer_name=f'fc{d}', initialiser=initialiser)
        outputs.append(x)

    x = layers.add([y for y in outputs])

    x = Dense(1, name='fc_last')(x)

    model = Model(input, x)

    return model


def feed_forward(feature_dim, depth=16, noise=0.0001):

    width = feature_dim

    input = Input(shape=(feature_dim,), name='input')

    if noise == 0:
        x = input
    else:
        x = GaussianNoise(noise)(input)

    outputs = []
    outputs.append(x)

    for d in range(0, depth):
        x = block(x, outputs=width, layer_name=f'fc{d}')
        outputs.append(x)

    x = layers.add([y for y in outputs])

    x = Dense(1, name='fc_last')(x)

    model = Model(input, x)

    return model


def yinyang2(feature_dim, depth=8):
    input_feature = Input(shape=(feature_dim,), name='input')

    width = feature_dim

    x = input_feature
    outputs = []
    outputs.append(x)

    for d in range(0, depth):
        if d != 0:
            u = layers.add([input_feature, x])
        else:
            u = x
        x = block(u, outputs=width, layer_name=f'fc{d}')
        outputs.append(x)

    x = layers.add([y for y in outputs])

    x = Dense(1, name='fc_last')(x)

    model = Model(input_feature, x)

    return model


def back_feed(feature_dim, depth=8):
    input_feature = Input(shape=(feature_dim,), name='input')

    width = feature_dim

    x = input_feature

    for d in range(0, depth):
        if d != 0:
            u = layers.add([input_feature, x])
        else:
            u = x
        x = block(u, outputs=width, layer_name=f'fc{d}')

    x = Dense(1, name='fc_last')(x)

    model = Model(input_feature, x)

    return model
