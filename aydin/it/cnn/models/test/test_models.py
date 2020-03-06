import numpy as np
from aydin.it.cnn.models.unet_2d import unet_2d_model
from aydin.it.cnn.models.unet_3d import unet_3d_model


def test_supervised_2D():
    a = np.zeros((1, 64, 64, 1))
    model2d = unet_2d_model((64, 64, 1), num_lyr=2, shiftconv=False, supervised=True)
    test_past = True
    try:
        model2d.predict(a)
    except Exception:
        test_past = False

    assert test_past, 'An error was raised in supervised Unet 2D model.'


def test_shiftconv_2D():
    a = np.zeros((1, 64, 64, 1))
    model2d = unet_2d_model((64, 64, 1), num_lyr=2, shiftconv=True, supervised=False)
    test_past = True
    try:
        model2d.predict(a)
    except Exception:
        test_past = False

    assert test_past, 'An error was raised in shiftconv Unet 2D model.'


def test_masking_2D():
    a = np.zeros((1, 64, 64, 1))
    model2d = unet_2d_model((64, 64, 1), num_lyr=2, shiftconv=False, supervised=False)
    test_past = True
    try:
        model2d.predict([a, a])
    except Exception:
        test_past = False

    assert test_past, 'An error was raised in masking Unet 2D model.'


def test_supervised_3D():
    a = np.zeros((1, 64, 64, 64, 1))
    model3d = unet_3d_model(
        (64, 64, 64, 1), num_lyr=2, shiftconv=False, supervised=True
    )
    test_past = True
    try:
        model3d.predict(a)
    except Exception:
        test_past = False

    assert test_past, 'An error was raised in supervised Unet 3D model.'


def test_shiftconv_3D():
    a = np.zeros((1, 64, 64, 64, 1))
    model3d = unet_3d_model(
        (64, 64, 64, 1), num_lyr=2, shiftconv=True, supervised=False
    )
    test_past = True
    try:
        model3d.predict(a)
    except Exception:
        test_past = False

    assert test_past, 'An error was raised in shiftconv Unet 3D model.'


def test_masking_3D():
    a = np.zeros((1, 64, 64, 64, 1))
    model3d = unet_3d_model(
        (64, 64, 64, 1), num_lyr=2, shiftconv=False, supervised=False
    )
    test_past = True
    try:
        model3d.predict([a, a])
    except Exception:
        test_past = False

    assert test_past, 'An error was raised in masking Unet 3D model.'
