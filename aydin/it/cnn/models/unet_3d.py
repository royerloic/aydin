import numpy
from tensorflow_core.python.keras.api.keras import optimizers
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers.convolutional import (
    UpSampling3D,
    ZeroPadding3D,
    Cropping3D,
    Conv3D,
)
from tensorflow_core.python.keras.layers.merge import Concatenate
from tensorflow_core.python.keras.models import Model

from aydin.it.cnn.layers.util import split, rot90
from aydin.it.cnn.layers.maskout import Maskout
from aydin.it.cnn.models.utils.conv_block import (
    conv3d_bn_noshift,
    conv3d_bn,
    mxpooling_down3D,
)


class Unet3DModel(Model):

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

    def __init__(
        self,
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

        :param input_dim:
        :param rot_batch_size:
        :param num_lyr:
        :param normalization:
        :param activation:
        :param supervised:
        :param shiftconv:
        :param initial_unit:
        :param learning_rate:
        :param original_zdim:
        """
        self.compiled = False

        self.rot_batch_size = rot_batch_size
        self.num_lyr = num_lyr
        self.normalization = normalization
        self.activation = activation
        self.shiftconv = shiftconv
        self.initial_unit = initial_unit
        self.learning_rate = learning_rate

        if original_zdim:
            self.zdim = original_zdim
        else:
            self.zdim = input_dim[0]

        # Generate a model
        self.input_lyr = Input(input_dim, name='input')

        x = self.unet_core()

        if not shiftconv and not supervised:
            input_msk = Input(input_dim, name='input_msk')
            x = Maskout(name='maskout')([x, input_msk])
            super().__init__([self.input_lyr, input_msk], x)
        else:
            super().__init__(self.input_lyr, x)

        self.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='mse')

    def unet_core(self):
        # Rotation & stack of the input images
        if self.shiftconv:
            input1 = rot90(self.input_lyr, kk=1, lyrname='rot1')(self.input_lyr)
            input2 = rot90(self.input_lyr, kk=2, lyrname='rot2')(self.input_lyr)
            input3 = rot90(self.input_lyr, kk=3, lyrname='rot3')(self.input_lyr)
            input5 = rot90(self.input_lyr, kk=5, lyrname='rot5')(self.input_lyr)
            input6 = rot90(self.input_lyr, kk=6, lyrname='rot6')(self.input_lyr)
            x = Concatenate(name='conc_in', axis=0)(
                [self.input_lyr, input1, input2, input3, input5, input6]
            )
        else:
            x = self.input_lyr

        skiplyr = [x]
        down2D_n = 0
        for i in range(self.num_lyr):
            x = conv3d_bn(
                x,
                self.initial_unit * (i + 1),  # * 2**i,  #  initial_unit,  #
                self.shiftconv,
                self.normalization,
                self.activation,
                lyrname=f'enc{i}',
            )
            if self.zdim > 3:
                pool_size = (2, 2, 2)
                self.zdim = numpy.floor(self.zdim / 2)
            else:
                pool_size = (1, 2, 2)
                down2D_n += 1
            x = mxpooling_down3D(x, self.shiftconv, pool_size, lyrname=f'enc{i}pl')

            if i != self.num_lyr - 1:
                skiplyr.append(x)

        x = conv3d_bn(
            x,
            self.initial_unit,
            self.shiftconv,
            self.normalization,
            self.activation,
            lyrname='bottm',  # * num_layer,
        )

        for i in range(self.num_lyr):
            if down2D_n > 0:
                x = UpSampling3D((1, 2, 2), name=f'up{i}')(x)
                down2D_n -= 1
            else:
                x = UpSampling3D((2, 2, 2), name=f'up{i}')(x)
            x = Concatenate(name=f'cnct{i}')([x, skiplyr.pop()])
            x = conv3d_bn(
                x,
                self.initial_unit
                * (self.num_lyr - i),  # * 2 ** (num_layer - i),  #  * 2,  #
                self.shiftconv,
                self.normalization,
                self.activation,
                lyrname=f'dec{i}',
            )

        if self.shiftconv:
            # Shift the center pixel
            x = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name='shiftc_pd')(x)
            x = Cropping3D(((0, 0), (0, 0), (0, 1)), name='shiftc_crp')(x)

            # Rotation & stack for the output
            output0 = split(x, 0, self.rot_batch_size, 'split0')(x)
            output1 = split(x, 1, self.rot_batch_size, 'split1')(x)
            output2 = split(x, 2, self.rot_batch_size, 'split2')(x)
            output3 = split(x, 3, self.rot_batch_size, 'split3')(x)
            output5 = split(x, 4, self.rot_batch_size, 'split5')(x)
            output6 = split(x, 5, self.rot_batch_size, 'split6')(x)
            output1 = rot90(output1, -1, lyrname='rot-1')(output1)
            output2 = rot90(output2, -2, lyrname='rot-2')(output2)
            output3 = rot90(output3, -3, lyrname='rot-3')(output3)
            output5 = rot90(output5, -5, lyrname='rot-5')(output5)
            output6 = rot90(output6, -6, lyrname='rot-6')(output6)
            x = Concatenate(name='cnct_last', axis=-1)(
                [output0, output1, output2, output3, output5, output6]
            )
            x = conv3d_bn_noshift(
                x,
                self.initial_unit * 2 * 4,
                self.normalization,
                act=self.activation,
                lyrname='last1',
            )
            x = conv3d_bn_noshift(
                x,
                self.initial_unit,
                self.normalization,
                act=self.activation,
                lyrname='last2',
            )

        x = Conv3D(1, 3, padding='same', name='last0', activation='linear')(x)

        return x
