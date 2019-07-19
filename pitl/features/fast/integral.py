def integral_1d(opencl_provider, input, output):
    """
    Compute a 1D integral image

    """
    image_x = input.shape[0]

    # TODO: This is a silly kernel that does not do parallelism...
    program_code = f"""

      __kernel void integral_1d(__global float *input, __global float *output)
      {{
        //const int ox = get_global_id(0);
        
        float acc=0.0f;
        for(int i=0; i<{image_x}; i++)
        {{
            acc+=input[i];
            output[i] = acc;
        }}        
      }}
      """
    # print(program_code)

    integral_1d = opencl_provider.build(program_code).integral_1d

    integral_1d(opencl_provider.queue, (1,), None, input.data, output.data)

    return output


def integral_2d(opencl_provider, input, temp1, temp2):
    """
    Compute a 2D integral image

    """
    image_y, image_x = input.shape

    def get_program_code(u, v, l):
        return f"""
      __kernel void integral_2d(__global float *input, __global float *output)
      {{
        const int a      = get_global_id(0);
        const int offset = a*{u};
        
        float sum = 0.0f;
        for(int j=0; j<{l}; j++)
        {{
            const int index = offset + j*{v};
            const float value = input[index];
            sum += value;
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

    integral_2d_y(opencl_provider.queue, (image_x,), None, input.data, temp1.data)
    integral_2d_x(opencl_provider.queue, (image_y,), None, temp1.data, temp2.data)

    return temp2


def integral_3d(opencl_provider, input, temp1, temp2):
    """
    Compute a 3D integral image

    """
    image_z, image_y, image_x = input.shape

    def get_program_code(u, v, g, l):
        return f"""
      __kernel void integral_3d(__global float *input, __global float *output)
      {{
        const int a = get_global_id(0);
        const int b = get_global_id(1);

        const int offset = a*{u} + b*{v};
        
        float sum = 0.0f;
        
        for(int j=0; j<{l}; j++)
        {{
            const int index = offset + j*{g};
            const float value = input[index];
            sum += value;
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

    integral_3d_z(
        opencl_provider.queue, (image_x, image_y), None, input.data, temp1.data
    )
    integral_3d_y(
        opencl_provider.queue, (image_x, image_z), None, temp1.data, temp2.data
    )
    integral_3d_x(
        opencl_provider.queue, (image_y, image_z), None, temp2.data, temp1.data
    )

    return temp1


def integral_4d(opencl_provider, input, temp1, temp2):
    """
    Compute a 4D integral image

    """
    image_w, image_z, image_y, image_x = input.shape

    def get_program_code(u, v, g, h, l):
        return f"""
      __kernel void integral_4d(__global float *input, __global float *output)
      {{
          const int a = get_global_id(0);
          const int b = get_global_id(1);
          const int c = get_global_id(2);

          const int offset = a*{u} + b*{v} + c*{g};

          float sum = 0.0f;

          for(int j=0; j<{l}; j++)
          {{
            const int index = offset + j*{h};
            const float value = input[index];
            sum += value;
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

    integral_4d_w(
        opencl_provider.queue, (image_x, image_y, image_z), None, input.data, temp1.data
    )
    integral_4d_z(
        opencl_provider.queue, (image_x, image_y, image_w), None, temp1.data, temp2.data
    )
    integral_4d_y(
        opencl_provider.queue, (image_x, image_z, image_w), None, temp2.data, temp1.data
    )
    integral_4d_x(
        opencl_provider.queue, (image_y, image_z, image_w), None, temp1.data, temp2.data
    )

    return temp2
