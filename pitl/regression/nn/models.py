import keras
from keras import layers
from keras.layers import Input, Dense, LeakyReLU, Lambda, BatchNormalization
from keras.models import Model
from keras_layer_normalization import LayerNormalization


def block(input, outputs=1, layer_name=None, trainable=True):
    x = Dense(outputs, name=layer_name + 'd1', trainable=trainable, use_bias=False)(
        input
    )
    # x = BatchNormalization(name=layer_name + 'bn1', center=True, scale=False)(x)
    x = LeakyReLU(name=layer_name + 'act1')(x)
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


def feed_forward(feature_dim, width=None, depth=16):
    if not width:
        width = feature_dim
    width = max(width, feature_dim)

    input = Input(shape=(feature_dim,), name='input')

    outputs = []
    outputs.append(input)

    x = input

    for d in range(0, depth):
        x = block(x, outputs=width, layer_name=f'fc{d}')
        outputs.append(x)

    x = layers.add([y for y in outputs])

    x = Dense(1, name='fc_last')(x)

    model = Model(input, x)

    return model
