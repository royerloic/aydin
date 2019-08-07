import tempfile
from typing import Tuple

import numpy


def offcore_array(shape: Tuple[int], dtype: numpy.dtype):
    """
    Instanciates an array of given shape and dtype in  'off-core' fashion i.e. not in main memory.
    Right now it simply uses memory mapping on temp file that is deleted after the file is closed

    :param shape:
    :type shape:
    :param dtype:
    :type dtype:
    :return:
    :rtype:
    """
    temp_file = tempfile.NamedTemporaryFile()
    array = numpy.memmap(temp_file, dtype=dtype, mode='w+', shape=shape)

    return array
