import os, importlib
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Concatenate, Conv2D, LeakyReLU, ZeroPadding2D, Cropping2D, \
    Lambda, BatchNormalization, Layer, multiply
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


class DNCNN:
    def __init__(self,
                 input_dim,
                 num_lyr=3,
                 normalization='instance',
                 supervised=False,
                 shiftconv=True,
                 learn_rate=0.001,
                 batch_size=None,
                 backend=None):

        """
        Create a DnCNN model

        3 training modes are available:
        supervised: noisy and clean images are required
        shiftconv: self-supervised learning with shift and conv scheme
        non-shiftconv: self-supervised learning by masking pixels at each iteration

        """

        # Configuring backend
        if backend != None and backend not in K.backend():
            if backend == 'plaidml':
                os.environ["KERAS_BACKEND"] = 'plaidml.keras.backend'
            elif backend == 'tensorflow':
                os.environ["KERAS_BACKEND"] = 'tensorflow'
            importlib.reload(K)
            assert backend in K.backend()

        self.input_dim = input_dim
        self.num_lyr = num_lyr
        self.normalization = normalization  # 'instance' or 'batch'
        self.supervised = supervised
        self.shiftconv = shiftconv
        self.lr = learn_rate
        self.batch_size = batch_size

        assert supervised != shiftconv, 'Shift convolution scheme is only available for self-supervised learning.'
        if supervised:
            print('Model will be created for supervised learning.')
        elif not supervised and shiftconv:
            print('Model will be generated for self-supervised learning with shift convlution scheme.')
        elif not supervised and not shiftconv:
            print('Model will be generated for self-supervised with moving-blind spot scheme.')

        def rot90(xx, kk=1, lyrname=None):  # rotate tensor by 90 degrees on the HW plane
            def rot(x1, k=1):
                if k < 0:
                    direction = 2
                else:
                    direction = 1
                for _ in range(abs(k)):
                    x1 = K.reverse(K.transpose(x1), axes=direction)
                    if k % 2 == 1:
                        pattern = (3, 1, 2, 0)
                    else:
                        pattern = (0, 1, 2, 3)
                return K.permute_dimensions(x1, pattern)

            out_shape = list(K.int_shape(xx))
            if kk % 2 == 1:
                out_shape[1:3] = np.flip(out_shape[1:3], 0)
            return Lambda(lambda xx: rot(xx, kk), output_shape=out_shape[1:], name=lyrname)

        def split(x, idx, batchsize=None, lyrname=None):    # split tensor at the batch axis
            out_shape = K.int_shape(x[0])
            if batchsize == None:
                batchsize = 1
            return Lambda(lambda xx: xx[idx:idx + batchsize], output_shape=out_shape, name=lyrname)

        def conv2d_bn(xx, unit, shiftconv=shiftconv, norm=normalization, lyrname=None):
            if shiftconv:
                x1 = ZeroPadding2D(((0, 0), (2, 0)), name=lyrname + '_0pd')(xx)
                x1 = Conv2D(unit, (3, 3), padding='same', name=lyrname + '_cv2')(x1)
                x1 = Cropping2D(((0, 0), (0, 2)), name=lyrname + '_crp')(x1)
            else:
                x1 = Conv2D(unit, (3, 3), padding='same', name=lyrname + '_cv2')(xx)
            if norm == 'instance':
                x1 = InstanceNormalization(name=lyrname + '_in')(x1)
            elif norm == 'batch':
                x1 = BatchNormalization(name=lyrname + '_bn')(x1)
            return LeakyReLU(alpha=0.1, name=lyrname + '_act')(x1)

        def conv2d_bn_noshift(xx, unit, norm=normalization, lyrname=None):
            x1 = Conv2D(unit, (1, 1), padding='same', name=lyrname + '_cv2')(xx)
            if norm == 'instance':
                x1 = InstanceNormalization(name=lyrname + '_in')(x1)
            elif norm == 'batch':
                x1 = BatchNormalization(name=lyrname + '_bn')(x1)
            return LeakyReLU(alpha=0.1, name=lyrname + '_act')(x1)

        class Maskout(Layer):   # A layer that mutiply mask with image
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

        # Generate a model
        input_lyr = Input(self.input_dim, name='input')
        if not shiftconv:
            input_msk = Input(self.input_dim, name='input_msk')

        # Rotation & stack of the input images
        if shiftconv:
            input1 = rot90(input_lyr, kk=1, lyrname='rot1')(input_lyr)
            input2 = rot90(input_lyr, kk=2, lyrname='rot2')(input_lyr)
            input3 = rot90(input_lyr, kk=3, lyrname='rot3')(input_lyr)
            x = Concatenate(name='conc_in', axis=0)([input_lyr, input1, input2, input3])
        else:
            x = input_lyr

        # Iteratively create conv layers
        for i in range(num_lyr):
            x = conv2d_bn(x, 48, shiftconv, normalization, lyrname=f'lyr{i}')

        # Rotation & stack for the output
        if shiftconv:
            output0 = split(x, 0, batch_size, 'split0')(x)
            output1 = split(x, 1, batch_size, 'split1')(x)
            output2 = split(x, 2, batch_size, 'split2')(x)
            output3 = split(x, 3, batch_size, 'split3')(x)
            output1 = rot90(output1, -1, lyrname='rot4')(output1)
            output2 = rot90(output2, -2, lyrname='rot5')(output2)
            output3 = rot90(output3, -3, lyrname='rot6')(output3)
            x = Concatenate(name='cnct_last', axis=-1)([output0, output1, output2, output3])
            x = conv2d_bn_noshift(x, 32, lyrname='last1')

        x = Conv2D(1, (1, 1), padding='same', name='last0', activation='linear')(x)

        if not shiftconv:
            x = Maskout(name='maskout')([x, input_msk])

        self.model = Model(input_lyr, x)
        self.model.compile(optimizer=optimizers.Adam(lr=self.lr), loss='mse')

    def fit(self,
            input_img,
            target_img=None,
            mask_shape=None,
            num_epoch=500,
            EStop_patience=5,
            ReduceLR_patience=3):

        def masker(batch_vol, i, mask_shape):
            i = i % np.prod(mask_shape)
            mask = np.zeros(np.prod(mask_shape), dtype=bool)
            mask[i] = True
            mask = mask.reshape(mask_shape)
            rep = np.ceil(np.asarray(batch_vol) / np.asarray(mask_shape)).astype(int)
            mask = np.tile(mask, tuple(rep))
            mask = mask[:batch_vol[0], :batch_vol[1]]
            return mask

        def maskedgen(batch_vol, mask_shape, image):
            while True:
                for i in range(np.prod(mask_shape)):
                    mask = masker(batch_vol, i, mask_shape)
                    masknega = np.expand_dims(np.expand_dims(mask, 0), 3)
                    train_img = np.expand_dims(np.expand_dims(~mask, 0), 3) * image
                    target_img = masknega * image
                    yield {'input': train_img, 'input_msk': masknega.astype(np.float32)}, target_img

        EStop = EarlyStopping(monitor='loss',
                              min_delta=1e-5,
                              patience=EStop_patience,
                              verbose=1,
                              mode='auto')
        ReduceLR = ReduceLROnPlateau(monitor='loss',
                                     factor=0.1,
                                     verbose=1,
                                     patience=ReduceLR_patience,
                                     mode='auto',
                                     min_lr=1e-8)

        if self.supervised:
            history = self.model.fit(input_img,
                                     target_img,
                                     batch_size=self.batch_size,
                                     epochs=num_epoch,
                                     callbacks=[EStop, ReduceLR])
        elif self.shiftconv:
            history = self.model.fit(input_img,
                                     input_img,
                                     batch_size=self.batch_size,
                                     epochs=num_epoch,
                                     callbacks=[EStop, ReduceLR])
        else:
            history = self.model.fit_generator(maskedgen(self.input_dim, mask_shape, input_img),
                                               epochs=num_epoch,
                                               steps_per_epoch=np.prod(mask_shape),
                                               callbacks=[EStop, ReduceLR])
        return history

    def predict(self,
                input_img,
                batch_size=None):
        return self.model.predict(input_img,
                                  batch_size=batch_size,
                                  verbose=1)

    def summary(self):
        return self.model.summary()