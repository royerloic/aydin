import numpy
import pyopencl
from pyopencl import cltypes
from pyopencl.array import Array

from aydin.providers.opencl.opencl_provider import OpenCLProvider


def integral_1d(
    opencl_provider: OpenCLProvider, input: Array, output: Array, mean: float = None
):
    """
    Compute a 1D integral image

    """
    image_x = input.shape[0]

    if mean is None:
        mean = float(pyopencl.array.sum(input).get() / input.size)

    # TODO: This is a silly kernel that does not do parallelism...
    program_code = f"""

      __kernel void integral_1d(__global float *input, __global float *output, float mean)
      {{
        //const int ox = get_global_id(0);
        
        float sum =0.0f;
        float cmp =0.0f;
         
        for(int i=0; i<{image_x}; i++)
        {{
            const float y = (input[i] - mean) - cmp ;
            const float t = sum + y;
            cmp = (t-sum) - y;
            sum = t;
            output[i] = sum;
        }}        
      }}
      """
    # print(program_code)

    integral_1d = opencl_provider.build(program_code).integral_1d

    mean = mean if isinstance(mean, numpy.ndarray) else cltypes.float(mean)

    integral_1d(opencl_provider.queue, (1,), None, input.data, output.data, mean)

    return output, mean


def integral_2d(
    opencl_provider: OpenCLProvider,
    input: Array,
    temp1: Array,
    temp2,
    mean: float = None,
):
    """
    Compute a 2D integral image

    """
    image_y, image_x = input.shape

    if mean is None:
        mean = pyopencl.array.sum(input).get() / input.size

    def get_program_code(u, v, l):
        return f"""
      __kernel void integral_2d(__global float *input, __global float *output, float mean)
      {{
        const int a      = get_global_id(0);
        const int offset = a*{u};
        
        float sum = 0.0f;
        float cmp = 0.0f;
        
        for(int j=0; j<{l}; j++)
        {{
            const int index = offset + j*{v};
            const float y = (input[index] - mean) - cmp;
            const float t = sum + y;
            cmp = (t-sum) - y;
            sum = t;
            output[index] = sum;
        }}
      }}
      """

    # print(program_code)

    integral_2d_y = opencl_provider.build(
        get_program_code(1, image_x, image_y)
    ).integral_2d

    integral_2d_x = opencl_provider.build(
        get_program_code(image_x, 1, image_x)
    ).integral_2d

    mean = mean if isinstance(mean, numpy.ndarray) else cltypes.float(mean)

    integral_2d_y(opencl_provider.queue, (image_x,), None, input.data, temp1.data, mean)
    integral_2d_x(
        opencl_provider.queue,
        (image_y,),
        None,
        temp1.data,
        temp2.data,
        mean * numpy.float32(0),
    )

    return temp2, mean


def integral_3d(
    opencl_provider: OpenCLProvider,
    input: Array,
    temp1: Array,
    temp2: Array,
    mean: float = None,
):
    """
    Compute a 3D integral image

    """
    image_z, image_y, image_x = input.shape

    if mean is None:
        mean = pyopencl.array.sum(input).get() / input.size

    def get_program_code(u, v, g, l):
        return f"""
      __kernel void integral_3d(__global float *input, __global float *output, float mean)
      {{
        const int a = get_global_id(0);
        const int b = get_global_id(1);

        const int offset = a*{u} + b*{v};
        
        float sum = 0.0f;
        float cmp = 0.0f;
        
        for(int j=0; j<{l}; j++)
        {{
            const int index = offset + j*{g};
            const float y = (input[index] - mean) - cmp;
            const float t = sum + y;
            cmp  = (t-sum) - y;
            sum = t;
            output[index] = sum;
            
        }}
      }}
      """

    # print(program_code)

    integral_3d_z = opencl_provider.build(
        get_program_code(1, image_x, image_x * image_y, image_z)
    ).integral_3d
    integral_3d_y = opencl_provider.build(
        get_program_code(1, image_x * image_y, image_x, image_y)
    ).integral_3d
    integral_3d_x = opencl_provider.build(
        get_program_code(image_x, image_x * image_y, 1, image_x)
    ).integral_3d

    mean = mean if isinstance(mean, numpy.ndarray) else cltypes.float(mean)

    integral_3d_z(
        opencl_provider.queue, (image_x, image_y), None, input.data, temp1.data, mean
    )
    integral_3d_y(
        opencl_provider.queue,
        (image_x, image_z),
        None,
        temp1.data,
        temp2.data,
        mean * numpy.float32(0),
    )
    integral_3d_x(
        opencl_provider.queue,
        (image_y, image_z),
        None,
        temp2.data,
        temp1.data,
        mean * numpy.float32(0),
    )

    return temp1, mean


def integral_4d(
    opencl_provider: OpenCLProvider,
    input: Array,
    temp1: Array,
    temp2: Array,
    mean: float = None,
):
    """
    Compute a 4D integral image

    """
    image_w, image_z, image_y, image_x = input.shape

    if mean is None:
        mean = pyopencl.array.sum(input).get() / input.size

    def get_program_code(u, v, g, h, l):
        return f"""
      __kernel void integral_4d(__global float *input, __global float *output, float mean)
      {{
          const int a = get_global_id(0);
          const int b = get_global_id(1);
          const int c = get_global_id(2);

          const int offset = a*{u} + b*{v} + c*{g};

          float sum = 0.0f;
          float cmp   = 0.0f;

          for(int j=0; j<{l}; j++)
          {{
            const int index = offset + j*{h};
            const float y = (input[index] - mean) - cmp;
            const float t = sum + y;
            cmp  = (t-sum) - y;
            sum = t;
            output[index] = sum;
          }}
      }}
      """

    # print(program_code)

    integral_4d_w = opencl_provider.build(
        get_program_code(
            1, image_x, image_x * image_y, image_x * image_y * image_z, image_w
        )
    ).integral_4d

    integral_4d_z = opencl_provider.build(
        get_program_code(
            1, image_x, image_x * image_y * image_z, image_x * image_y, image_z
        )
    ).integral_4d

    integral_4d_y = opencl_provider.build(
        get_program_code(
            1, image_x * image_y, image_x * image_y * image_z, image_x, image_y
        )
    ).integral_4d

    integral_4d_x = opencl_provider.build(
        get_program_code(
            image_x, image_x * image_y, image_x * image_y * image_z, 1, image_x
        )
    ).integral_4d

    mean = mean if isinstance(mean, numpy.ndarray) else cltypes.float(mean)

    integral_4d_w(
        opencl_provider.queue,
        (image_x, image_y, image_z),
        None,
        input.data,
        temp1.data,
        mean,
    )
    integral_4d_z(
        opencl_provider.queue,
        (image_x, image_y, image_w),
        None,
        temp1.data,
        temp2.data,
        mean * numpy.float32(0),
    )
    integral_4d_y(
        opencl_provider.queue,
        (image_x, image_z, image_w),
        None,
        temp2.data,
        temp1.data,
        mean * numpy.float32(0),
    )
    integral_4d_x(
        opencl_provider.queue,
        (image_y, image_z, image_w),
        None,
        temp1.data,
        temp2.data,
        mean * numpy.float32(0),
    )

    return temp2, mean
