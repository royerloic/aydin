import os
from napari.components import LayerList
from napari._qt.qt_layerlist import QtLayerList, QtDivider
from napari import gui_qt
from napari.layers import Image
import numpy as np

os.environ['NAPARI_TEST'] = '1'


def check_layout_layers(layout, layers):
    """
    Check the layer widget order matches the layers order in the layout

    Parameters
    ----------
    layout : QLayout
        Layout to test
    layers : napari.components.LayerList
        LayersList to compare to

    Returns
    ----------
    match : bool
        Boolean if layout matches layers
    """
    layers_layout = [
        layout.itemAt(2 * i - 1).widget().layer for i in range(len(layers), 0, -1)
    ]
    return layers_layout == list(layers)


def check_layout_dividers(layout, nlayers):
    """
    Check the layout contains dividers at the right places

    Parameters
    ----------
    layout : QLayout
        Layout to test
    nlayers : int
        Number of layers that should be present

    Returns
    ----------
    match : bool
        Boolean if layout contains dividers in the right places
    """
    dividers_layout = [type(layout.itemAt(2 * i).widget()) for i in range(1 + nlayers)]
    return dividers_layout == [QtDivider] * (1 + nlayers)


def test_divider():
    """
    Test creating the divider.
    """
    with gui_qt():
        divider = QtDivider()

        # Check divider was created properly
        assert type(divider) == QtDivider

        # Check divider property defaults to False
        assert divider.property('selected') == False

        # Set divider property
        divider.setSelected(True)
        assert divider.property('selected') == True
        divider.setSelected(False)
        assert divider.property('selected') == False


def test_creating_empty_view():
    """
    Test creating LayerList view.
    """
    with gui_qt():
        layers = LayerList()
        view = QtLayerList(layers)

        # Check that the layers model has been appended to the layers view
        assert view.layers == layers

        # Check that vbox_layout only contains one QtDivider and one spacer
        assert view.vbox_layout.count() == 2
        assert check_layout_layers(view.vbox_layout, layers)
        assert check_layout_dividers(view.vbox_layout, 0)


def test_adding_layers():
    """
    Test adding layers.
    """
    with gui_qt():
        layers = LayerList()
        view = QtLayerList(layers)

        # Check that new layer and divider get added to vbox_layout
        layer_a = Image(np.random.random((10, 10)))
        layers.append(layer_a)
        assert view.vbox_layout.count() == 2 * (len(layers) + 1)
        assert check_layout_layers(view.vbox_layout, layers)
        assert check_layout_dividers(view.vbox_layout, len(layers))

        # Check that new layers and dividers get added to vbox_layout
        layer_b = Image(np.random.random((15, 15)))
        layer_c = Image(np.random.random((15, 15)))
        layer_d = Image(np.random.random((15, 15)))
        layers.append(layer_b)
        layers.append(layer_c)
        layers.append(layer_d)
        assert view.vbox_layout.count() == 2 * (len(layers) + 1)
        assert check_layout_layers(view.vbox_layout, layers)
        assert check_layout_dividers(view.vbox_layout, len(layers))


def test_removing_layers():
    """
    Test removing layers.
    """
    with gui_qt():
        layers = LayerList()
        view = QtLayerList(layers)

        layer_a = Image(np.random.random((10, 10)))
        layer_b = Image(np.random.random((15, 15)))
        layer_c = Image(np.random.random((15, 15)))
        layer_d = Image(np.random.random((15, 15)))
        layers.append(layer_a)
        layers.append(layer_b)
        layers.append(layer_c)
        layers.append(layer_d)

        # Check layout and layers list match after removing a layer
        layers.remove(layer_b)
        assert view.vbox_layout.count() == 2 * (len(layers) + 1)
        assert check_layout_layers(view.vbox_layout, layers)
        assert check_layout_dividers(view.vbox_layout, len(layers))

        # Check layout and layers list match after removing a layer
        layers.remove(layer_d)
        assert view.vbox_layout.count() == 2 * (len(layers) + 1)
        assert check_layout_layers(view.vbox_layout, layers)
        assert check_layout_dividers(view.vbox_layout, len(layers))

        layers.append(layer_b)
        layers.append(layer_d)
        # Select first and third layers
        for l, s in zip(layers, [True, True, False, False]):
            l.selected = s
        layers.remove_selected()
        assert view.vbox_layout.count() == 2 * (len(layers) + 1)
        assert check_layout_layers(view.vbox_layout, layers)
        assert check_layout_dividers(view.vbox_layout, len(layers))


def test_reordering_layers():
    """
    Test reordering layers.
    """
    with gui_qt():
        layers = LayerList()
        view = QtLayerList(layers)

        layer_a = Image(np.random.random((10, 10)), name='image_a')
        layer_b = Image(np.random.random((15, 15)), name='image_b')
        layer_c = Image(np.random.random((15, 15)), name='image_c')
        layer_d = Image(np.random.random((15, 15)), name='image_d')
        layers.append(layer_a)
        layers.append(layer_b)
        layers.append(layer_c)
        layers.append(layer_d)

        # Check layout and layers list match after rearranging layers
        layers[:] = layers[(1, 0, 3, 2)]
        assert view.vbox_layout.count() == 2 * (len(layers) + 1)
        assert check_layout_layers(view.vbox_layout, layers)
        assert check_layout_dividers(view.vbox_layout, len(layers))

        # Check layout and layers list match after swapping two layers
        layers['image_b', 'image_c'] = layers['image_c', 'image_b']
        assert view.vbox_layout.count() == 2 * (len(layers) + 1)
        assert check_layout_layers(view.vbox_layout, layers)
        assert check_layout_dividers(view.vbox_layout, len(layers))

        # Check layout and layers list match after reversing list
        # TEST CURRENTLY FAILING
        # layers.reverse()
        # assert view.vbox_layout.count() == 2 * (len(layers) + 1)
        # assert check_layout_layers(view.vbox_layout, layers)
        # assert check_layout_dividers(view.vbox_layout, len(layers))

        # Check layout and layers list match after rearranging selected layers
        layer_e = Image(np.random.random((15, 15)))
        layer_f = Image(np.random.random((15, 15)))
        layers.append(layer_e)
        layers.append(layer_f)
        for l, s in zip(layers, [False, True, False, False, True, False]):
            l.selected = s
        layers.move_selected(1, 2)
        assert view.vbox_layout.count() == 2 * (len(layers) + 1)
        assert check_layout_layers(view.vbox_layout, layers)
        assert check_layout_dividers(view.vbox_layout, len(layers))
