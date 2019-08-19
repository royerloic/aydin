import os
from distutils.version import StrictVersion
from pathlib import Path
from qtpy import API_NAME


# if API_NAME == 'PySide2':
#     # Set plugin path appropriately if using PySide2. This is a bug fix
#     # for when both PyQt5 and Pyside2 are installed
#     import PySide2
#
#     os.environ['QT_PLUGIN_PATH'] = str(Path(PySide2.__file__).parent / 'Qt' / 'plugins')

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
