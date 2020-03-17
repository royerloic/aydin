import numpy as np
from aydin.it.cnn.models.unet_2d import UNet2DModel
from aydin.it.cnn.models.unet_3d import Unet3DModel


def test_supervised_2D():
    input_array = np.zeros((1, 64, 64, 1), dtype=np.float32)
    model2d = UNet2DModel((64, 64, 1), num_lyr=2, shiftconv=False, supervised=True)
    result = model2d.predict(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_shiftconv_2D():
    input_array = np.zeros((1, 64, 64, 1), dtype=np.float32)
    model2d = UNet2DModel((64, 64, 1), num_lyr=2, shiftconv=True, supervised=False)
    result = model2d.predict(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_masking_2D():
    input_array = np.zeros((1, 64, 64, 1), dtype=np.float32)
    model2d = UNet2DModel((64, 64, 1), num_lyr=2, shiftconv=False, supervised=False)
    result = model2d.predict([input_array, input_array])
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_supervised_3D():
    input_array = np.zeros((1, 64, 64, 64, 1), dtype=np.float32)
    model3d = Unet3DModel((64, 64, 64, 1), num_lyr=2, shiftconv=False, supervised=True)
    result = model3d.predict(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_shiftconv_3D():
    input_array = np.zeros((1, 64, 64, 64, 1), dtype=np.float32)
    model3d = Unet3DModel((64, 64, 64, 1), num_lyr=2, shiftconv=True, supervised=False)
    result = model3d.predict(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_masking_3D():
    input_array = np.zeros((1, 64, 64, 64, 1), dtype=np.float32)
    model3d = Unet3DModel((64, 64, 64, 1), num_lyr=2, shiftconv=False, supervised=False)
    result = model3d.predict([input_array, input_array])
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
