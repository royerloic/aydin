import os
import numpy as np
from napari.components import Dims
from napari._qt.qt_dims import QtDims
from napari import gui_qt

os.environ['NAPARI_TEST'] = '1'


def test_creating_view():
    """
    Test creating dims view.
    """
    with gui_qt():
        ndim = 4
        dims = Dims(ndim)
        view = QtDims(dims)

        # Check that the dims model has been appended to the dims view
        assert view.dims == dims

        # Check the number of sliders is two less than the number of dimensions
        assert view.nsliders == ndim - 2


def test_changing_ndim():
    """
    Test changing the number of dimensions
    """
    with gui_qt():
        ndim = 4
        view = QtDims(Dims(ndim))

        # Check that adding dimensions adds sliders
        view.dims.ndim = 5
        assert view.nsliders == 3

        # Check that removing dimensions removes sliders
        view.dims.ndim = 2
        assert view.nsliders == 0


def test_changing_display():
    """
    Test changing the displayed property of an axis
    """
    with gui_qt():
        ndim = 4
        view = QtDims(Dims(ndim))
        assert view.nsliders == 2
        assert np.sum(view._displayed) == 2

        # Check changing displayed removes a slider
        view.dims.set_display(1, True)
        assert np.sum(view._displayed) == 1


def test_slider_values():
    """
    Test the values of a slider stays matched to the values of the dims point.
    """
    with gui_qt():
        ndim = 4
        view = QtDims(Dims(ndim))

        # Check that values of the dimension slider matches the values of the
        # dims point at initialization
        assert view.sliders[0].getValues() == [view.dims.point[0]] * 2

        # Check that values of the dimension slider matches the values of the
        # dims point after the point has been moved within the dims
        view.dims.set_point(0, 2)
        assert view.sliders[0].getValues() == [view.dims.point[0]] * 2

        # Check that values of the dimension slider matches the values of the
        # dims point after the point has been moved within the slider
        view.sliders[0].setValue(1)
        assert view.sliders[0].getValues() == [view.dims.point[0]] * 2


def test_slider_range():
    """
    Tests range of the slider is matched to the range of the dims
    """
    with gui_qt():
        ndim = 4
        view = QtDims(Dims(ndim))

        # Check the range of slider matches the values of the range of the dims
        # at initialization
        assert view.sliders[0].start == view.dims.range[0][0]
        assert view.sliders[0].end == view.dims.range[0][1] - view.dims.range[0][2]
        assert view.sliders[0].single_step == view.dims.range[0][2]

        # Check the range of slider stays matched to the values of the range of
        # the dims
        view.dims.set_range(0, (1, 5, 2))
        assert view.sliders[0].start == view.dims.range[0][0]
        assert view.sliders[0].end == view.dims.range[0][1] - view.dims.range[0][2]
        assert view.sliders[0].single_step == view.dims.range[0][2]


def test_order_when_changing_ndim():
    """
    Test order of the sliders when changing the number of dimensions.
    """
    with gui_qt():
        ndim = 4
        view = QtDims(Dims(ndim))

        # Check that values of the dimension slider matches the values of the
        # dims point after the point has been moved within the dims
        view.dims.set_point(0, 2)
        view.dims.set_point(1, 1)
        for i in range(view.dims.ndim - 2):
            assert view.sliders[i].getValues() == [view.dims.point[i]] * 2

        # Check the matching dimensions and sliders are preserved when
        # dimensions are added
        view.dims.ndim = 5
        for i in range(view.dims.ndim - 2):
            assert view.sliders[i].getValues() == [view.dims.point[i]] * 2

        # Check the matching dimensions and sliders are preserved when dims
        # dimensions are removed
        view.dims.ndim = 4
        for i in range(view.dims.ndim - 2):
            assert view.sliders[i].getValues() == [view.dims.point[i]] * 2

        # Check the matching dimensions and sliders are preserved when dims
        # dimensions are removed
        view.dims.ndim = 3
        for i in range(view.dims.ndim - 2):
            assert view.sliders[i].getValues() == [view.dims.point[i]] * 2
