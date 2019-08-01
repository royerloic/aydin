import numpy
from pyopencl import cltypes


def collect_feature_2d(
    opencl_provider,
    image_gpu,
    integral_image_gpu,
    feature_gpu,
    dx,
    dy,
    lx,
    ly,
    exclude_center=True,
    mean: float = 0,
    optimisation=True,
):
    """
    Compute a given feature for a displacement (dx,dy) relative to the center pixel, and a patch size (lx,ly)

    """

    image_y, image_x = image_gpu.shape
    feature_y, feature_x = feature_gpu.shape

    assert image_x == feature_x
    assert image_y == feature_y

    rx = lx // 2
    ry = ly // 2

    exclude_center = exclude_center and abs(dx) <= rx and abs(dy) <= ry

    if optimisation and lx == 1 and ly == 1:
        program_code = f"""

        __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
         {{
           const int x = get_global_id(1);
           const int y = get_global_id(0);

           const int x0  = x+{dx};
           const int y0  = y+{dy};
           const int i0 = x0 + {image_x}*y0;
           const int i = x + {image_x}*y;
           
           const float value = x0<0||y0<0||x0>{image_x-1}||y0>{image_y-1} ? 0.0f : image[i0];

           feature[i] = {"0.0f" if exclude_center else "value" };
         }}
        """
    else:
        program_code = f"""

      __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
      {{

        const int x = get_global_id(1);
        const int y = get_global_id(0);

        const int x0  = min(x+{dx}-{rx}-1, {image_x - 1});
        const int x1  = min(x+{dx}+{rx},   {image_x - 1}); 
        const int x2  = min(x+{dx}-{rx}-1, {image_x - 1});
        const int x3  = min(x+{dx}+{rx},   {image_x - 1}); 
        
        const int y0  = min(y+{dy}-{ry}-1, {image_y - 1});
        const int y1  = min(y+{dy}-{ry}-1, {image_y - 1}); 
        const int y2  = min(y+{dy}+{ry},   {image_y - 1});
        const int y3  = min(y+{dy}+{ry},   {image_y - 1}); 
        
        const uint i0 = x0 + {image_x} * y0;
        const uint i1 = x1 + {image_x} * y1;
        const uint i2 = x2 + {image_x} * y2;
        const uint i3 = x3 + {image_x} * y3;
        
        const uint i = x + {image_x}*y;
       
        const float value0 = x0<0||y0<0 ? 0.0f : integral[i0];
        const float value1 = x1<0||y1<0 ? 0.0f : integral[i1];
        const float value2 = x2<0||y2<0 ? 0.0f : integral[i2];
        const float value3 = x3<0||y3<0 ? 0.0f : integral[i3];
        const float value4 = {"image[i]" if exclude_center else "0.0f"};
        
        const float adj = {lx*ly-1 if exclude_center else lx*ly}*mean;

        const float value = (value3
                            -value2-value1
                            +value0
                            -value4
                            +adj)*{1.0 / (lx * ly)};


        feature[i] = value;
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

    pass
