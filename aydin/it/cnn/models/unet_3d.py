import tensorflow as tf
import numpy
from aydin.it.cnn.layers.split import split
from aydin.it.cnn.layers.masking import Maskout
from aydin.it.cnn.layers.layers import rot90
from aydin.it.cnn.layers.conv_block import (
    conv3d_bn_noshift,
    conv3d_bn,
    mxpooling_down3D,
)
from aydin.it.cnn.models.utils.checks import unet_checks

optimizers = tf.keras.optimizers
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Concatenate = tf.keras.layers.Concatenate
Conv3D = tf.keras.layers.Conv3D
ZeroPadding3D = tf.keras.layers.ZeroPadding3D
Cropping3D = tf.keras.layers.Cropping3D
UpSampling3D = tf.keras.layers.UpSampling3D
K = tf.keras.backend


def unet_3d_model(
    input_dim,
    rot_batch_size=1,
    num_lyr=4,
    normalization='batch',  # None,  # 'instance',
    activation='ReLU',
    supervised=False,
    shiftconv=True,
    initial_unit=48,
    learning_rate=0.001,
    original_zdim=None,
):

    """
    Create a Unet model

    3 training modes are available:
    supervised: noisy and clean images are required
    shiftconv: self-supervised learning with shift and conv scheme
    non-shiftconv: self-supervised learning by masking pixels at each iteration

    :param input_dim: input dimension
    :param rot_batch_size: batch size for rotation. Only for shift convolution architecture
    :param num_lyr: number of layers
    :param normalization: type of batch normalization
    :param activation: type of activation function
    :param supervised: supervised or unsupervised
    :param initial_unit: number of filters in the first layer
    :param learning_rate: learning rate
    :param original_zdim: size of z dimension used in training. <inference only>
    """

    # Configure
    shiftconv = unet_checks(input_dim, num_lyr, supervised, shiftconv)
    if original_zdim:
        zdim = original_zdim
    else:
        zdim = input_dim[0]

    # Generate a model
    input_lyr = Input(input_dim, name='input')
    if not shiftconv and not supervised:
        input_msk = Input(input_dim, name='input_msk')

    # Rotation & stack of the input images
    if shiftconv:
        input1 = rot90(input_lyr, kk=1, lyrname='rot1')(input_lyr)
        input2 = rot90(input_lyr, kk=2, lyrname='rot2')(input_lyr)
        input3 = rot90(input_lyr, kk=3, lyrname='rot3')(input_lyr)
        input5 = rot90(input_lyr, kk=5, lyrname='rot5')(input_lyr)
        input6 = rot90(input_lyr, kk=6, lyrname='rot6')(input_lyr)
        x = Concatenate(name='conc_in', axis=0)(
            [input_lyr, input1, input2, input3, input5, input6]
        )
    else:
        x = input_lyr

    skiplyr = [x]
    down2D_n = 0
    for i in range(num_lyr):
        x = conv3d_bn(
            x,
            initial_unit * (i + 1),  # * 2**i,  #  initial_unit,  #
            shiftconv,
            normalization,
            activation,
            lyrname=f'enc{i}',
        )
        if zdim > 3:
            pool_size = (2, 2, 2)
            zdim = numpy.floor(zdim / 2)
        else:
            pool_size = (1, 2, 2)
            down2D_n += 1
        x = mxpooling_down3D(x, shiftconv, pool_size, lyrname=f'enc{i}pl')

        if i != num_lyr - 1:
            skiplyr.append(x)

    x = conv3d_bn(
        x,
        initial_unit,
        shiftconv,
        normalization,
        activation,
        lyrname='bottm',  # * num_layer,
    )

    for i in range(num_lyr):
        if down2D_n > 0:
            x = UpSampling3D((1, 2, 2), name=f'up{i}')(x)
            down2D_n -= 1
        else:
            x = UpSampling3D((2, 2, 2), name=f'up{i}')(x)
        x = Concatenate(name=f'cnct{i}')([x, skiplyr.pop()])
        x = conv3d_bn(
            x,
            initial_unit * (num_lyr - i),  # * 2 ** (num_layer - i),  #  * 2,  #
            shiftconv,
            normalization,
            activation,
            lyrname=f'dec{i}',
        )

    # Shift the center pixel
    if shiftconv:
        x = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name='shiftc_pd')(x)
        x = Cropping3D(((0, 0), (0, 0), (0, 1)), name='shiftc_crp')(x)

    # Rotation & stack for the output
    if shiftconv:
        output0 = split(x, 0, rot_batch_size, 'split0')(x)
        output1 = split(x, 1, rot_batch_size, 'split1')(x)
        output2 = split(x, 2, rot_batch_size, 'split2')(x)
        output3 = split(x, 3, rot_batch_size, 'split3')(x)
        output5 = split(x, 4, rot_batch_size, 'split5')(x)
        output6 = split(x, 5, rot_batch_size, 'split6')(x)
        output1 = rot90(output1, -1, lyrname='rot-1')(output1)
        output2 = rot90(output2, -2, lyrname='rot-2')(output2)
        output3 = rot90(output3, -3, lyrname='rot-3')(output3)
        output5 = rot90(output5, -5, lyrname='rot-5')(output5)
        output6 = rot90(output6, -6, lyrname='rot-6')(output6)
        x = Concatenate(name='cnct_last', axis=-1)(
            [output0, output1, output2, output3, output5, output6]
        )
        x = conv3d_bn_noshift(
            x, initial_unit * 2 * 4, normalization, act=activation, lyrname='last1'
        )
        x = conv3d_bn_noshift(
            x, initial_unit, normalization, act=activation, lyrname='last2'
        )

    # x = Conv3D(1, (1, 1, 1), padding='same', name='last0', activation='linear')(x)
    x = Conv3D(1, 3, padding='same', name='last0', activation='linear')(x)

    if not shiftconv and not supervised:
        x = Maskout(name='maskout')([x, input_msk])
        model = Model([input_lyr, input_msk], x)
    else:
        model = Model(input_lyr, x)

    opt = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='mse')

    return model

    # Helper methods
