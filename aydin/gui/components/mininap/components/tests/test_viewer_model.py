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
