import math

import numpy
import psutil

from aydin.util.log.logging import lsection, lprint
from aydin.providers.opencl.opencl_provider import OpenCLProvider


class MemoryCheckCNN:
    def __init__(self, CNNparams, dtype=numpy.float32):

        """
        :CNNparams number of parameters in the CNN model:
        :dtype data type of the input data:
        """

        self.CNNparams = CNNparams
        self.dtype = dtype
        assert dtype == numpy.float32 or dtype == numpy.uint8 or dtype == numpy.uint16

    def _ensure_opencl_prodider_initialised(self):
        if not hasattr(self, 'opencl_provider') or self.opencl_provider is None:
            self.opencl_provider = OpenCLProvider()

    def is_enough_memory(self, array):
        with lsection('Is enough memory for feature generation ?'):
            # Ensure we have an openCL provider initialised:
            self._ensure_opencl_prodider_initialised()

            # 20% buffer:
            buffer = 1.2

            max_avail_cpu_mem = 0.9 * psutil.virtual_memory().total
            max_avail_gpu_mem = 0.9 * self.opencl_provider.device.global_mem_size
            lprint(f"Maximum cpu mem: {max_avail_cpu_mem / 1E6} MB")
            lprint(f"Maximum gpu mem: {max_avail_gpu_mem / 1E6} MB")

            # This is what we need on the GPU to store the image and one feature:
            # needed_gpu_mem = 2 * int(buffer * 4 * array.size * 1)

            # TODO: CHECK: this assumes floats, is this correct? use dtype:
            needed_gpu_mem = 2 * int(buffer * 4 * self.CNNparams * 1)
            lprint(f"Memory needed on the gpu: {needed_gpu_mem / 1E6} MB (CNN model)")

            # nb_dim = len(array.shape)
            # approximate_num_features = len(self.get_feature_descriptions(nb_dim, False))
            # lprint(f"Approximate number of features: {approximate_num_features}")

            # This is what we need on the CPU:
            needed_cpu_mem = int(
                buffer
                * numpy.dtype(self.dtype).itemsize
                * array.size
                # * self.CNNparams
            )
            lprint(f"Memory needed on the cpu: {needed_cpu_mem / 1E6} MB")

            min_nb_batches_cpu = math.ceil(needed_cpu_mem / max_avail_cpu_mem)
            min_nb_batches_gpu = math.ceil(needed_gpu_mem / max_avail_gpu_mem)
            min_nb_batches = max(min_nb_batches_cpu, min_nb_batches_gpu)
            lprint(
                f"Minimum number of batches: {min_nb_batches} ( cpu:{min_nb_batches_cpu}, gpu:{min_nb_batches_gpu} )"
            )

            max_batch_size = (array.itemsize * array.size) / min_nb_batches
            is_enough_memory = (
                needed_cpu_mem < max_avail_cpu_mem
                and needed_gpu_mem < max_avail_gpu_mem
            )
            lprint(f"Maximum batch size: {max_batch_size} ")
            lprint(f"Is enough memory: {is_enough_memory} ")

            return is_enough_memory, max_batch_size
