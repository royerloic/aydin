def collect_feature_1d(opencl_provider, image_gpu, feature_gpu, dx, scale):
    """
    Compute a given feature for a displacement (dx) relative to the center pixel, and a patch size (lx)

    """
    image_x = image_gpu.shape[0]
    feature_x = feature_gpu.shape[1]

    program_code = f"""

      __kernel void feature_kernel(__global float *image, __global float *feature)
      {{
          int fx = get_global_id(0);

          int x = fx/{scale}+{dx};
          x = max(0,x);
          x = min(x,{image_x-1});

          int image_index = x ;
          float value = image[image_index];

          int feature_index = fx;
          feature[feature_index] = value;
      }}
      """
    # print(program_code)

    program = opencl_provider.build(program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(
        opencl_provider.queue, feature_gpu.shape, None, image_gpu.data, feature_gpu.data
    )


## TODO: implement these other ones:


def collect_feature_2d(opencl_provider, image_gpu, feature_gpu, dx, dy, scale):
    """
    Compute a given feature for a displacement (dx) relative to the center pixel, and a patch size (lx)

    """
    image_x = image_gpu.shape[1]
    image_y = image_gpu.shape[0]

    feature_x = feature_gpu.shape[1]
    feature_y = feature_gpu.shape[0]

    program_code = f"""

      __kernel void feature_kernel(__global float *image, __global float *feature)
      {{
          int fx = get_global_id(1);
          int fy = get_global_id(0);

          int x = fx/{scale}+{dx};
          x = max(0,x);
          x = min(x,{image_x-1});
          
          int y = fy/{scale}+{dy};
          y = max(0,y);
          y = min(y,{image_y-1});

          int image_index = x + {image_x}*y;
          float value = image[image_index];

          int feature_index =  fx + {feature_x}*fy;
          feature[feature_index] = value;
      }}
      """
    # print(program_code)

    program = opencl_provider.build(program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(
        opencl_provider.queue, feature_gpu.shape, None, image_gpu.data, feature_gpu.data
    )

    pass


def collect_feature_3d(
    generator, image_gpu, feature_gpu, dx, lx, exclude_center=True, reduction='sum'
):
    pass


def collect_feature_4d(
    generator, image_gpu, feature_gpu, dx, lx, exclude_center=True, reduction='sum'
):
    pass
