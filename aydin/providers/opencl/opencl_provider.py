import itertools
import os

import numpy
import pyopencl as cl
from pyopencl._cl import get_platforms

from aydin.util.log.logging import lsection, lprint


class OpenCLProvider:
    def __init__(self, includes=[], excludes=['CPU']):

        with lsection(f"Initialising OpenCL device and context:"):
            os.environ['PYOPENCL_NO_CACHE'] = '1'

            self.devices = self.get_filtered_device_list(includes, excludes)

            self.device = self.devices[0]
            lprint(f"Selected device: {self.device.name}")

            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)

            self.program_cache = {}

    def get_all_devices(self):
        return list(
            itertools.chain.from_iterable(
                [platform.get_devices() for platform in get_platforms()]
            )
        )

    def get_filtered_device_list(self, includes=[], excludes=[], sort_by_mem_size=True):

        with lsection(f"Obtaining list of OpenCL devices:"):
            devices = self.get_all_devices()

            with lsection(f"All OpenCL devices:"):
                for device in devices:
                    lprint(device.name)

            for exclude in excludes:
                devices = [device for device in devices if not exclude in device.name]

            for include in includes:
                devices = [device for device in devices if include in device.name]

            if sort_by_mem_size:
                devices = sorted(devices, key=lambda x: x.global_mem_size, reverse=True)

            devices = [device for device in devices if self.test_device(device)]

            with lsection(f"Filtered and sorted OpenCL devices:"):
                for device in devices:
                    lprint(
                        f"Device {device.name} with {device.global_mem_size / 1e6} MB "
                    )

            return list(devices)

    def test_device(self, device):

        with lsection(f"Testing OpenCL device: {device.name} "):

            try:

                a_np = numpy.random.rand(50000).astype(numpy.float32)
                b_np = numpy.random.rand(50000).astype(numpy.float32)

                ctx = cl.create_some_context()
                queue = cl.CommandQueue(ctx)

                mf = cl.mem_flags
                a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
                b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

                prg = cl.Program(
                    ctx,
                    """
                __kernel void sum(
                    __global const float *a_g, __global const float *b_g, __global float *res_g)
                {
                  int gid = get_global_id(0);
                  res_g[gid] = a_g[gid] + b_g[gid];
                }
                """,
                ).build()

                res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
                prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

                res_np = numpy.empty_like(a_np)
                cl.enqueue_copy(queue, res_np, res_g)

                # Check on CPU with Numpy:
                # print(res_np - (a_np + b_np))
                # print(numpy.linalg.norm(res_np - (a_np + b_np)))
                assert numpy.allclose(res_np, a_np + b_np)

                lprint(f"Device {device.name} _is_ operational.")

                return True

            except Exception as e:

                lprint(e)
                lprint(
                    f"Device {device.name} is not operational: it failed to run some basic tensor operation."
                )

                return False

    def build(self, program_code, disable_opts=False):

        # lprint(f"Building program. disable_opts={disable_opts}")

        if program_code in self.program_cache:
            return self.program_cache[program_code]
        else:
            options = []

            if disable_opts:
                options.append("-cl-opt-disable")

            program = cl.Program(self.context, program_code).build(options=options)
            self.program_cache[program_code] = program
            return program
