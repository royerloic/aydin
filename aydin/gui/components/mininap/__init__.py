# flake8: noqa
from distutils.version import StrictVersion


from qtpy import QtCore

# When QT is not the specific version, we raise a warning:
from warnings import warn

if StrictVersion(QtCore.__version__) < StrictVersion('5.12.3'):
    warn_message = f"""
    napari was tested with QT library `>=5.12.3`.
    The version installed is {QtCore.__version__}. Please report any issues with this
    specific QT version at https://github.com/Napari/napari/issues.
    """
    warn(message=warn_message)

from .viewer import Viewer
from ._qt import gui_qt
