# flake8: noqa

import numpy
from pyopencl import cltypes


def collect_feature_2d(
    opencl_provider,
    image_gpu,
    integral_image_gpu,
    feature_gpu,
    dx,
    dy,
    nx,
    ny,
    px,
    py,
    exclude_center=True,
    mean: float = 0,
    optimisation=True,
):
    """
    Compute a given feature for a displacement (dx,dy) relative to the center pixel, and a patch size (lx,ly)

    """

    image_y, image_x = image_gpu.shape
    feature_y, feature_x = feature_gpu.shape

    if image_x != feature_x or image_y != feature_y:
        raise ValueError('Dimensions of image_gpu and feature_gpu has to be same')

    # exclude_center = exclude_center  and abs(dx) <= rx and abs(dy) <= ry

    if optimisation and nx + px == 1 and ny + py == 1:
        program_code = f"""

        __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
         {{
           const int x = get_global_id(1);
           const int y = get_global_id(0);

           const int x0  = x+{dx};
           const int y0  = y+{dy};
           
           const long i0  = x0 + {image_x}*y0;
           const float value = x0<0||y0<0||x0>{image_x-1}||y0>{image_y-1} ? 0.0f : image[i0];
           
           const long i   = x  + {image_x}*y;
           feature[i] = {"x0==x && y0==y ? 0.0f : value" if exclude_center else "value"};
         }}
        """
    else:
        program_code = f"""

        inline float integral_lookup(__global float *integral, int x, int y)
        {{
            x = min(x, {image_x-1});
            y = min(y, {image_y-1});
            
            const long i = x + {image_x} * y;
            
            return (x<0||y<0) ? 0.0f : integral[i];
        }}

      __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
      {{

        const int x = get_global_id(1);
        const int y = get_global_id(0);

        const int xl  = x+{dx - nx}-1;
        const int xh  = x+{dx + px}  ; 

        const int yl  = y+{dy - ny}-1;
        const int yh  = y+{dy + py}  ;

        const float value0 = integral_lookup(integral, xl, yl);
        const float value1 = integral_lookup(integral, xh, yl);
        const float value2 = integral_lookup(integral, xl, yh);
        const float value3 = integral_lookup(integral, xh, yh);

        const bool center_inside = (xl<x) && (x<=xh) && (yl<y) && (y<=yh);
        const bool all_inside    = (xl<0) && ({image_x-1}<=xh) && (yl<0) && ({image_y-1}<=yh); 
        const bool exclude_center = {"center_inside && !all_inside" if exclude_center else "false"}; 

        const long raw_volume = (xh-xl)*(yh-yl);
        const long volume = raw_volume - (exclude_center ? 1 : 0);

        const long i = x + {image_x}*y;
        const float center = exclude_center ? image[i] : 0.0f;

        const float value = (
                            +value3
                            -value2-value1
                            +value0
                            -center
                            )
                            / volume
                            + mean;  

        feature[i] = clamp(value, 0.0f, nextafter(1.0f,0.0f));
      }}
      """
    # print(program_code)
    # //

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

    pass
