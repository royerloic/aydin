import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import (
    Input,
    Concatenate,
    Conv2D,
    ZeroPadding2D,
    Cropping2D,
    UpSampling2D,
)


from aydin.it.cnn.layers import (
    split,
    rot90,
    conv2d_bn_noshift,
    Maskout,
    conv2d_bn,
    mxpooling_down,
)


def unet_model(
    input_dim,
    rot_batch_size=1,
    num_lyr=5,
    normalization=None,  # 'instance',
    activation='ReLU',
    supervised=False,
    shiftconv=True,
    initial_unit=48,
    learning_rate=0.001,
):

    """
    Create a Unet model

    3 training modes are available:
    supervised: noisy and clean images are required
    shiftconv: self-supervised learning with shift and conv scheme
    non-shiftconv: self-supervised learning by masking pixels at each iteration

    """

    # Configure
    configure(input_dim, num_lyr, supervised, shiftconv)

    # Generate a model
    input_lyr = Input(input_dim, name='input')
    if not shiftconv and not supervised:
        input_msk = Input(input_dim, name='input_msk')

    # Rotation & stack of the input images
    if shiftconv:
        input1 = rot90(input_lyr, kk=1, lyrname='rot1')(input_lyr)
        input2 = rot90(input_lyr, kk=2, lyrname='rot2')(input_lyr)
        input3 = rot90(input_lyr, kk=3, lyrname='rot3')(input_lyr)
        x = Concatenate(name='conc_in', axis=0)([input_lyr, input1, input2, input3])
    else:
        x = input_lyr

    skiplyr = [x]
    for i in range(num_lyr):
        x = conv2d_bn(
            x,
            initial_unit * (i + 1),  # * 2**i,  #  initial_unit,  #
            shiftconv,
            normalization,
            activation,
            lyrname=f'enc{i}',
        )
        x = mxpooling_down(x, shiftconv, lyrname=f'enc{i}pl')
        if i != num_lyr - 1:
            skiplyr.append(x)

    x = conv2d_bn(
        x,
        initial_unit,
        shiftconv,
        normalization,
        activation,
        lyrname='bottm',  # * num_lyr,
    )

    for i in range(num_lyr):
        x = UpSampling2D((2, 2), interpolation='nearest', name=f'up{i}')(x)
        x = Concatenate(name=f'cnct{i}')([x, skiplyr.pop()])
        x = conv2d_bn(
            x,
            initial_unit * (num_lyr - i),  # * 2 ** (num_lyr - i),  #  * 2,  #
            shiftconv,
            normalization,
            activation,
            lyrname=f'dec{i}',
        )

    # Shift the center pixel
    if shiftconv:
        x = ZeroPadding2D(((0, 0), (1, 0)), name='shiftc_pd')(x)
        x = Cropping2D(((0, 0), (0, 1)), name='shiftc_crp')(x)

    # Rotation & stack for the output
    if shiftconv:
        output0 = split(x, 0, rot_batch_size, 'split0')(x)
        output1 = split(x, 1, rot_batch_size, 'split1')(x)
        output2 = split(x, 2, rot_batch_size, 'split2')(x)
        output3 = split(x, 3, rot_batch_size, 'split3')(x)
        output1 = rot90(output1, -1, lyrname='rot4')(output1)
        output2 = rot90(output2, -2, lyrname='rot5')(output2)
        output3 = rot90(output3, -3, lyrname='rot6')(output3)
        x = Concatenate(name='cnct_last', axis=-1)([output0, output1, output2, output3])
        x = conv2d_bn_noshift(
            x, initial_unit * 2 * 4, normalization, act=activation, lyrname='last1'
        )
        x = conv2d_bn_noshift(
            x, initial_unit, normalization, act=activation, lyrname='last2'
        )

    x = Conv2D(1, (1, 1), padding='same', name='last0', activation='linear')(x)

    if not shiftconv and not supervised:
        x = Maskout(name='maskout')([x, input_msk])
        model = Model([input_lyr, input_msk], x)
        model.needs_mask = True
    else:
        model = Model(input_lyr, x)
        model.needs_mask = False

    opt = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='mse')

    return model

    # Helper methods


def configure(input_dim, num_lyr, supervised, shiftconv):
    # TODO: refactor this part
    if supervised:
        shiftconv = False
        print(
            'Shift convolution will be turned off automatically because supervised learning was selected.'
        )
    # assert (
    #     supervised != shiftconv
    # ), 'Shift convolution scheme is only available for self-supervised learning.'
    assert (
        np.mod(input_dim[:-1], np.repeat(2 ** num_lyr, len(input_dim[:-1]))) == 0
    ).all(), 'Each dimension of the input image has to be a multiple of 2^num_lyr. '
    if supervised:
        print('Model will be created for supervised learning.')
    elif not supervised and shiftconv:
        print(
            'Model will be generated for self-supervised learning with shift convlution scheme.'
        )
        assert (
            np.diff(input_dim[:2]) == 0
        ), 'Make sure the input image shape is cubic as shiftconv mode involves rotation.'
        assert (
            np.mod(input_dim[:-1], np.repeat(2 ** (num_lyr - 1), len(input_dim[:-1])))
            == 0
        ).all(), (
            'Each dimension of the input image has to be a multiple of '
            '2^(num_lyr-1) as shiftconv mode involvs pixel shift. '
        )
    elif not supervised and not shiftconv:
        print(
            'Model will be generated for self-supervised with moving-blind spot scheme.'
        )
    ###
