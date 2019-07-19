def collect_feature_1d(
    opencl_provider,
    image_gpu,
    integral_image_gpu,
    feature_gpu,
    dx,
    lx,
    exclude_center=True,
    optimisation=True,
):
    """
    Compute a given feature for a displacement (dx) relative to the center pixel, and a patch size (lx)

    """

    image_x, = image_gpu.shape
    feature_x, = feature_gpu.shape

    assert image_x == feature_x

    rx = lx // 2

    exclude_center = exclude_center and abs(dx) <= rx

    if optimisation and lx == 1:
        program_code = f"""

        __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature)
         {{
           const int x = get_global_id(0);

           const int x0  = x+{dx};
           const int i0 = x0;
           const int i = x;

           const float value = x0<0||x0>{image_x-1} ? 0.0f : image[i0];

           feature[i] = {"0.0f" if exclude_center else "value" };
         }}
        """
    else:
        program_code = f"""

      __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature)
      {{
        int x = get_global_id(0);

        const int x0  = min(x+{dx}-{rx}-1, {image_x - 1});
        const int x1  = min(x+{dx}+{rx},   {image_x - 1}); 

        const uint i0 = x0;
        const uint i1 = x1;
        
        const int i = x;

        const float value0 = x0<0 ? 0.0f : integral[i0];
        const float value1 = x1<0 ? 0.0f : integral[i1];
        const float value2 = {"image[i]" if exclude_center else "0.0f"};

        const float value = (value1-value0-value2)*{1.0 / lx};


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
    )
