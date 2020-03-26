from tensorflow_core.python.keras.api.keras import optimizers
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers.convolutional import (
    Conv2D,
    Cropping2D,
    ZeroPadding2D,
    UpSampling2D,
)
from tensorflow_core.python.keras.layers.merge import Concatenate, Add
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.regularizers import l1

from aydin.it.cnn.layers.util import split, rot90
from aydin.it.cnn.layers.maskout import Maskout
from aydin.it.cnn.models.utils.conv_block import conv2d_bn, pooling_down2D


class UNet2DModel(Model):

    """
    Create a Unet model for 2D images

    3 training modes are available:
    supervised: noisy and clean images are required
    shiftconv: self-supervised learning with shift and conv scheme
    non-shiftconv: self-supervised learning by masking pixels at each iteration
    """

    def __init__(
        self,
        input_dim,
        rot_batch_size=1,
        num_lyr=4,
        normalization=None,  # 'instance' or 'batch'
        activation='ReLU',
        supervised=False,
        shiftconv=True,
        initial_unit=8,
        learning_rate=0.01,
        weight_decay=0,
        residual=False,
        pooling_mode='max',
        interpolation='nearest',
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
        :param weight_decay: coefficient of l1 regularizer
        :param residual: whether to use add or concat at merging layers
        :param pooling_mode: 'max' for max pooling, 'ave' for average pooling
        :param interpolation: 'nearest' or 'bilinear' for Upsampling2D
        """
        self.compiled = False

        self.rot_batch_size = rot_batch_size
        self.num_lyr = num_lyr
        self.normalization = normalization
        self.activation = activation
        self.shiftconv = shiftconv
        self.initial_unit = initial_unit
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.residual = residual
        self.pooling_mode = pooling_mode
        self.interpolation = interpolation

        # Generate a model
        input_lyr = Input(input_dim, name='input')
        x = self.unet_core(input_lyr)

        if not shiftconv and not supervised:
            input_msk = Input(input_dim, name='input_msk')
            x = Maskout(name='maskout')([x, input_msk])
            super().__init__([input_lyr, input_msk], x)
        else:
            super().__init__(input_lyr, x)

        # Compile the model
        self.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='mse')
        self.compiled = True

    def unet_core(self, input_lyr):
        # Rotation & stack of the input images
        if self.shiftconv:
            input1 = rot90(input_lyr, kk=1, lyrname='rot1')(input_lyr)
            input2 = rot90(input_lyr, kk=2, lyrname='rot2')(input_lyr)
            input3 = rot90(input_lyr, kk=3, lyrname='rot3')(input_lyr)
            x = Concatenate(name='conc_in', axis=0)([input_lyr, input1, input2, input3])
        else:
            x = input_lyr

        skiplyr = [x]
        for i in range(self.num_lyr):
            if i == 0:
                x = conv2d_bn(
                    x,
                    unit=self.initial_unit * (i + 1),
                    shiftconv=self.shiftconv,
                    norm=None,
                    act=None,
                    weight_decay=self.weight_decay,
                    lyrname=f'enc{i}_cv0',
                )

            x = conv2d_bn(
                x,
                unit=self.initial_unit * (i + 1),
                shiftconv=self.shiftconv,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname=f'enc{i}',
            )
            x = pooling_down2D(
                x, self.shiftconv, mode=self.pooling_mode, lyrname=f'enc{i}pl'
            )
            if i != self.num_lyr - 1:
                skiplyr.append(x)

        x = conv2d_bn(
            x,
            self.initial_unit,
            shiftconv=self.shiftconv,
            norm=self.normalization,
            act=self.activation,
            weight_decay=self.weight_decay,
            lyrname='bottm',
        )

        for i in range(self.num_lyr):
            x = UpSampling2D((2, 2), interpolation=self.interpolation, name=f'up{i}')(x)
            if self.residual:
                x = Add(name=f'add{i}')([x, skiplyr.pop()])
            else:
                x = Concatenate(name=f'cnct{i}')([x, skiplyr.pop()])
            x = conv2d_bn(
                x,
                self.initial_unit * max((self.num_lyr - i - 2), 1),
                shiftconv=self.shiftconv,
                norm=None,
                act=None,
                weight_decay=self.weight_decay,
                lyrname=f'dec{i}_cv0',
            )
            x = conv2d_bn(
                x,
                self.initial_unit * max((self.num_lyr - i - 2), 1),
                shiftconv=self.shiftconv,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname=f'dec{i}',
            )

        if self.shiftconv:
            # Shift the center pixel
            x = ZeroPadding2D(((0, 0), (1, 0)), name='shiftc_pd')(x)
            x = Cropping2D(((0, 0), (0, 1)), name='shiftc_crp')(x)

            # Rotation & stack for the output
            output0 = split(x, 0, self.rot_batch_size, 'split0')(x)
            output1 = split(x, 1, self.rot_batch_size, 'split1')(x)
            output2 = split(x, 2, self.rot_batch_size, 'split2')(x)
            output3 = split(x, 3, self.rot_batch_size, 'split3')(x)
            output1 = rot90(output1, -1, lyrname='rot4')(output1)
            output2 = rot90(output2, -2, lyrname='rot5')(output2)
            output3 = rot90(output3, -3, lyrname='rot6')(output3)
            x = Concatenate(name='cnct_last', axis=-1)(
                [output0, output1, output2, output3]
            )
            x = conv2d_bn(
                x,
                self.initial_unit * 2 * 4,
                kernel_size=1,
                shiftconv=False,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname='last1',
            )
            x = conv2d_bn(
                x,
                self.initial_unit,
                kernel_size=1,
                shiftconv=False,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname='last2',
            )

        x = Conv2D(
            1,
            (1, 1),
            padding='same',
            name='last0',
            kernel_regularizer=l1(self.weight_decay),
            bias_regularizer=l1(self.weight_decay),
            activation='linear',
        )(x)

        return x
