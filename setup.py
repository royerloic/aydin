import sys
import os
from setuptools import setup


if sys.version_info < (3, 6):
    sys.stderr.write(
        f'You are using Python '
        + "{'.'.join(str(v) for v in sys.version_info[:3])}.\n\n"
        + 'aydin only supports Python 3.6 and above.\n\n'
        + 'Please install Python 3.6 using:\n'
        + '  $ pip install python==3.6\n\n'
    )
    sys.exit(1)

with open(os.path.join('requirements', 'default.txt')) as f:
    default_requirements = [
        line.strip() for line in f if line and not line.startswith('#')
    ]

INSTALL_REQUIRES = []
REQUIRES = []

for default_requirement in default_requirements:
    os.system("pip install " + str(default_requirement))

# Handle pyopencl
os.system("pip install -r " + os.path.join('requirements', 'pyopencl.txt'))

setup(
    name='aydin',
    version='0.0.3',
    py_modules=['aydin'],
    entry_points='''
            [console_scripts]
            aydin=aydin.cli.cli:aydin
        ''',
)
