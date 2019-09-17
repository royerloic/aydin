# -*- mode: python ; coding: utf-8 -*-
import sys
print("python ver:", sys.version[:3])
# if sys.version[:3] < "3.7":
#     sys.exit()
# else:
#     print("python version okay")

from glob import glob

def get_qt5_binaries():

    qt_sos = glob("/home/royerlab/anaconda3/envs/aydin/lib/python3.6/site-packages/PyQt5/*.so")

    return [(so,os.path.basename(so)) for so in qt_sos]

import vispy.glsl
import vispy.io
import distributed
import dask

from distutils.sysconfig import get_python_lib

from os import path
skimage_plugins = Tree(
    path.join(get_python_lib(), "skimage","io","_plugins"),
    prefix=path.join("skimage","io","plugins"),
)

block_cipher = None

binaries = get_qt5_binaries()
# binaries += [( '/home/royerlab/.aydin/plaidml/libplaidml2.so', 'plaidml/' )]

a = Analysis(['../../aydin/cli/cli.py'],
             # pathex=['/Users/ahmetcan.solak/Dev/AhmetCanSolak/aydin'],
             binaries=binaries,
             datas=[(os.path.join(os.path.dirname(dask.__file__)), 'dask'),
                    (os.path.join(os.path.dirname(distributed.__file__)), 'distributed'),
                    (os.path.dirname(vispy.glsl.__file__), os.path.join("vispy", "glsl")),
                    (os.path.join(os.path.dirname(vispy.io.__file__), "_data"), os.path.join("vispy", "io", "_data"))],
             hiddenimports=["gdown.download","plaidml.keras.backend","vispy.app.backends._pyqt5","vispy.glsl","sentry_sdk.integrations.argv",
                                                                     "sentry_sdk.integrations.modules",
                                                                     "sentry_sdk.integrations.logging",
                                                                     "sentry_sdk.integrations.stdlib",
                                                                     "sentry_sdk.integrations.excepthook",
                                                                     "sentry_sdk.integrations.dedupe",

                                                                     "sentry_sdk.integrations.wsgi",
                                                                     "sentry_sdk.integrations.tornado",
                                                                     "sentry_sdk.integrations.threading",
                                                                     "sentry_sdk.integrations.serverless",
                                                                     "sentry_sdk.integrations.sanic",
                                                                     "sentry_sdk.integrations.rq",
                                                                     "sentry_sdk.integrations.redis",
                                                                     "sentry_sdk.integrations.pyramid",

                                                                     "sentry_sdk.integrations.gnu_backtrace",
                                                                     "sentry_sdk.integrations.flask",
                                                                     "sentry_sdk.integrations.falcon",
                                                                     "sentry_sdk.integrations.celery",
                                                                     "sentry_sdk.integrations.bottle",
                                                                     "sentry_sdk.integrations.aws_lambda",
                                                                     "sentry_sdk.integrations.aiohttp",
                                                                     "sentry_sdk.integrations._sql_common",
                                                                     "sentry_sdk.integrations._wsgi_common",
                                                                     "sentry_sdk.integrations.atexit"],
             hookspath=["hooks"],
             runtime_hooks=["runtimehooks/hook-multiprocessing.py", "runtimehooks/hook-plaidml.py", "runtimehooks/hook-pyopencl.py"],
             excludes=["matplotlib", "napari"])



# filter binaries.. exclude some dylibs that pyinstaller packaged but
# we actually dont need (e.g. wxPython)

import re
reg = re.compile(".*(PyQt4|PyQt5\.Qt|mpl-data|tcl|zmq|QtWebKit|wxPython|matplotlib).*")
a.binaries = [s for s in a.binaries if reg.match(s[1]) is None]


pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='aydin',
          #debug=False,
          bootloader_ignore_signals=False,
          strip=None,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True)

app = BUNDLE(exe,
             a.binaries,
             a.zipfiles,
             name='aydin.app',
             icon=None)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               skimage_plugins,
               strip=False,
               upx=True,
               name='full_folder')

