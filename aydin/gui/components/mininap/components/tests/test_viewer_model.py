import numpy as np

from aydin.gui.components.mininap.components import ViewerModel


def test_viewer_model():
    """Test instantiating viewer model."""
    viewer = ViewerModel()
    assert viewer.title == 'napari'
    assert len(viewer.layers) == 0

    # Create viewer model with custom title
    viewer = ViewerModel(title='testing')
    assert viewer.title == 'testing'


def test_add_image():
    """Test adding image."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)


def test_mix_dims():
    """Test adding images of mixed dimensionality."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 2

    data = np.random.random((6, 10, 15))
    viewer.add_image(data)
    assert len(viewer.layers) == 2
    assert np.all(viewer.layers[1].data == data)
    assert viewer.dims.ndim == 3


def test_svg():
    "Test generating svg"
    viewer = ViewerModel()

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Generate svg
    svg = viewer.to_svg()
    assert type(svg) == str


def test_add_layer_dims_change():
    """Test dims change appropriately when adding and removing layers."""
    np.random.seed(0)
    viewer = ViewerModel()

    # Check ndim starts at 2
    assert viewer.dims.ndim == 2

    # Check ndim increase to 3 when 3D data added
    data = np.random.random((10, 15, 20))
    layer = viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 3
