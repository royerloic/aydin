import itertools
import os
import warnings

import numpy

import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

from aydin.util.log.log import lsection, lprint


def compiler_output(text):
    lprint("Non-empty OpenCL compiler output encountered")


cl.compiler_output = compiler_output


class OpenCLProvider:
    def __init__(self, includes=[], excludes=[]):

        with lsection(f"Initialising OpenCL device and context:"):
            os.environ['PYOPENCL_NO_CACHE'] = '1'

            self.devices = self.get_filtered_device_list(includes, excludes)

            self.device = self.devices[0]
            lprint(f"Selected device: {self.device.name}")

            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)

            self._program_cache = {}
            self._kernel_cache = {}

    def get_all_devices(self):
        return list(
            itertools.chain.from_iterable(
                [platform.get_devices() for platform in cl.get_platforms()]
            )
        )

    def get_filtered_device_list(self, includes=[], excludes=[], sort_by_mem_size=True):

        with lsection(f"Obtaining list of OpenCL devices:"):
            devices = self.get_all_devices()

            with lsection(f"All OpenCL devices:"):
                for device in devices:
                    lprint(device.name)

            for exclude in excludes:
                devices = [device for device in devices if not (exclude in device.name)]

            for include in includes:
                devices = [device for device in devices if include in device.name]

            cpu_devices = []
            gpu_devices = []
            for device in devices:
                if 'CPU' in device.name:
                    cpu_devices.append(device)
                else:
                    gpu_devices.append(device)

            if sort_by_mem_size:
                cpu_devices = sorted(
                    cpu_devices, key=lambda x: x.global_mem_size, reverse=True
                )
                gpu_devices = sorted(
                    gpu_devices, key=lambda x: x.global_mem_size, reverse=True
                )

            devices = gpu_devices + cpu_devices
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

                # ctx = cl.create_some_context()
                ctx = cl.Context([device])
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

    def _cache_sanity_check(self):
        if len(self._program_cache) > 100:
            warnings.warn(
                "Too many kernels instantiated (>100), perhaps something is wrong!",
                Warning,
            )
        if len(self._kernel_cache) > 100:
            warnings.warn(
                "Too many kernels instantiated (>100), perhaps something is wrong!",
                Warning,
            )

    @property
    def build_options(self):
        return [
            '-cl-mad-enable',
            '-cl-no-signed-zeros',
            '-cl-unsafe-math-optimizations',
            '-cl-finite-math-only',
        ]

    def build(self, program_code, disable_opts=False):

        self._cache_sanity_check()

        if program_code in self._program_cache:
            return self._program_cache[program_code]
        else:
            options = self.build_options

            if disable_opts:
                options.append("-cl-opt-disable")

            program = cl.Program(self.context, program_code).build(options=options)
            self._program_cache[program_code] = program
            return program

    def get_kernel(self, kernel_name: str, program_code: str):
        self._cache_sanity_check()
        key = program_code
        if key in self._kernel_cache:
            kernel = self._kernel_cache[key]
        else:
            # print(f"!! COMPILING KERNEL '{kernel_name}' !! ")
            program = self.build(program_code)
            kernel = program.custom_kernel
            self._kernel_cache[key] = kernel
        return kernel

    def get_elwise_kernel(self, arguments: str, operation: str, *args, **kwargs):
        self._cache_sanity_check()
        key = arguments + operation + str(args) + str(kwargs)
        if key in self._kernel_cache:
            kernel = self._kernel_cache[key]
        else:
            kernel = ElementwiseKernel(
                self.context,
                arguments,
                operation,
                "elwise_kernel",
                options=self.build_options,
                *args,
                **kwargs,
            )

            self._kernel_cache[key] = kernel
        return kernel

    def get_reduction_kernel(
        self,
        arguments: str,
        reduce_expression: str,
        map_expression: str,
        neutral: float = 0,
        dtype_out=numpy.float32,
        *args,
        **kwargs,
    ):
        self._cache_sanity_check()
        key = (
            str(dtype_out)
            + arguments
            + reduce_expression
            + map_expression
            + str(neutral)
            + str(dtype_out)
            + str(args)
            + str(kwargs)
            + str(self.context)
        )
        if key in self._kernel_cache:
            kernel = self._kernel_cache[key]
        else:
            kernel = ReductionKernel(
                self.context,
                dtype_out,  # numpy.float32,
                neutral=neutral,  # "0",
                reduce_expr=reduce_expression,  # "a+b",
                map_expr=map_expression,  # "x[i]*y[i]",
                arguments=arguments,  # "__global float *x, __global float *y",
                options=self.build_options,
                *args,
                **kwargs,
            )
            self._kernel_cache[key] = kernel
        return kernel
