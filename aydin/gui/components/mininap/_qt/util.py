from .qt_base_layer import QtLayerProperties, QtLayerControls


def create_qt_properties(layer):
    """
    Create a qt properties widget for a layer based on its layer type.

    Parameters
    ----------
        layer : napari.layers._base_layer.Layer
            Layer that needs its propetry widget created.

    Returns
    ----------
        properties : napari.layers.base.QtLayerProperties
            Qt propetry widget
    """
    name = 'Qt' + type(layer).__name__ + 'Properties'
    try:
        qt_props = globals()[name]
        properties = qt_props(layer)
    except KeyError:
        properties = QtLayerProperties(layer)

    return properties


def create_qt_controls(layer):
    """
    Create a qt controls widget for a layer based on its layer type.

    Parameters
    ----------
        layer : napari.layers._base_layer.Layer
            Layer that needs its controls widget created.

    Returns
    ----------
        controls : napari.layers.base.QtLayerControls
            Qt controls widget
    """
    name = 'Qt' + type(layer).__name__ + 'Controls'
    try:
        qt_controls = globals()[name]
        controls = qt_controls(layer)
    except KeyError:
        controls = QtLayerControls(layer)

    return controls
