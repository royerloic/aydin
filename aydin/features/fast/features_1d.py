import numpy
from pyopencl import cltypes


def collect_feature_1d(
    opencl_provider,
    image_gpu,
    integral_image_gpu,
    feature_gpu,
    dx,
    lx,
    exclude_center=True,
    mean: float = 0,
    optimisation=True,
):
    """
    Compute a given feature for a displacement (dx) relative to the center pixel, and a patch size (lx)

    """

    image_x, = image_gpu.shape
    feature_x, = feature_gpu.shape

    assert image_x == feature_x

    rx = lx // 2

    if optimisation and lx == 1:
        program_code = f"""

        __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
         {{
           const int x = get_global_id(0);

           const int x0 = x+{dx};
           
           const int i0 = x0;
           const float value = x0<0||x0>{image_x-1} ? 0.0f : image[i0];

           const int i = x;
           feature[i] = {"x0==x ? 0.0f : value" if exclude_center else "value"};
         }}
        """
    else:
        program_code = f"""
        
        inline float integral_lookup(__global float *integral, int x)
        {{
            x = min(x, {image_x-1});
            return x<0 ? 0.0f : integral[x];
        }}

      __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
      {{
        int x = get_global_id(0);

        const int xl  = x+{dx-rx}-1;
        const int xh  = x+{dx+rx}; 

        const float value0 = integral_lookup(integral, xl);
        const float value1 = integral_lookup(integral, xh);
        
        const bool center_inside = (xl<x) && (x<=xh);
        const bool all_inside    = (xl<0) && ({image_x-1}<=xh); 
        const bool exclude_center = {"center_inside && !all_inside" if exclude_center else "false"}; 

        const long raw_volume = (xh-xl);
        const long volume = raw_volume - (exclude_center ? 1 : 0);

        const long i = x;
        const float center = exclude_center ? image[i] : 0.0f;
        
        const float value = (
                            +value1
                            -value0
                            -center
                            )
                            /volume
                            + mean;

        feature[i] = clamp(value, 0.0f, nextafter(1.0f,0.0f));
      }}
      """
    # print(program_code)

    program = opencl_provider.build(program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(
        opencl_provider.queue,
        feature_gpu.shape,
        None,
        image_gpu.data,
        integral_image_gpu.data,
        feature_gpu.data,
        mean if isinstance(mean, numpy.ndarray) else cltypes.float(mean),
    )
