import numpy
import pytest
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.util import random_noise

from aydin.clnn.modules.linear.dense import Dense
from aydin.clnn.modules.special.resize import Resize
from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor
from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.util.log.log import Log


def test_module_state():
    module_state_with(CLTensor)
    module_state_with(NPTensor)


def test_module_computation():
    array1 = computation_with(CLTensor)
    array2 = computation_with(NPTensor)
    print(f"Difference in computation: {abs(array1-array2).mean()}")
    assert abs(array1 - array2).mean() < 0.00001


def test_to_tensor_class():
    to_tensor_class_with(CLTensor)
    to_tensor_class_with(NPTensor)


def module_state_with(tensor_class):
    x = tensor_class([[1.1, 2]])

    dense = Dense(nb_inputs=2, nb_outputs=2)
    dense.ensure_parameters_allocated(x)

    dense.weights.nparray = [[1.01, -1.01], [1, -1]]
    dense.biases.nparray = [0.0, 1]

    y = dense(x)

    assert y.nparray[0, 0] == pytest.approx(3.111)
    assert y.nparray[0, 1] == pytest.approx(-2.111)

    del y

    dense_ = Dense(nb_inputs=2, nb_outputs=2)
    dense_.ensure_parameters_allocated(x)

    dense_.state = dense.state

    y = dense_(x)

    assert y.nparray[0, 0] == pytest.approx(3.111)
    assert y.nparray[0, 1] == pytest.approx(-2.111)


def computation_with(tensor_class):

    Log.set_log_max_depth(0)

    def n(image):
        return rescale_intensity(
            image.astype(numpy.float32), in_range='image', out_range=(0, 1)
        )

    image = camera().astype(numpy.float32)
    image = n(image)
    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    generator = FastMultiscaleConvolutionalFeatures(
        max_level=1, exclude_scale_one=False
    )
    features = generator.compute(
        noisy, exclude_center_value=True, exclude_center_feature=True
    )
    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)

    module = Resize(None, shape=(1,))
    # module = Dense(None, nb_inputs=18, nb_outputs=1)

    x_t = tensor_class(x)
    denoised_image_test_tensor = module(x_t)
    denoised_image_test_array = denoised_image_test_tensor.nparray
    denoised_image_test = denoised_image_test_array.reshape(image.shape)

    # print(f"x.shape={x.shape}")
    # print(f"x.strides={x.strides}")
    # print(f"x_t.shape={x_t.shape}")
    # print(f"x_t.strides={x_t.strides}")
    # print(f"denoised_image_test_tensor.shape={denoised_image_test_tensor.shape}")
    # print(f"denoised_image_test.shape={denoised_image_test.shape}")

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(n(image), name='image')
    #     viewer.add_image(n(noisy), name='noisy')
    #     viewer.add_image(n(denoised_image_test), name='denoised')

    return denoised_image_test


def to_tensor_class_with(tensor_class):

    x_np = numpy.random.rand(3, 2)
    x = tensor_class(x_np)

    dense = Dense(nb_inputs=2, nb_outputs=2)
    dense.ensure_parameters_allocated(x)

    y = dense(x)

    dense_np = dense.to_tensor_class(NPTensor)
    x_np = x.to_class(NPTensor)
    y_np = dense_np(x_np)
    assert (y_np.nparray - y.nparray) == pytest.approx(0.0, abs=0.01)

    dense_cl = dense.to_tensor_class(CLTensor)
    x_cl = x.to_class(CLTensor)
    y_cl = dense_cl(x_cl)
    assert (y_cl.nparray - y.nparray) == pytest.approx(0.0, abs=0.01)

    print(f"Difference: {(y.nparray-y_cl.nparray).mean()}")
    print(f'{tensor_class} to all classes ok!')
