# flake8: noqa

median = """

		//extract next color channel
		uint r0,r1,r2,r3,r4,r5,r6,r7,r8;
		r0=uiRGBA[0]& uiMask;
		r1=uiRGBA[1]& uiMask;
		r2=uiRGBA[2]& uiMask;
		r3=uiRGBA[3]& uiMask;
		r4=uiRGBA[4]& uiMask;
		r5=uiRGBA[5]& uiMask;
		r6=uiRGBA[6]& uiMask;
		r7=uiRGBA[7]& uiMask;
		r8=uiRGBA[8]& uiMask;
		
		//perform partial bitonic sort to find current channel median
		uint uiMin = min(r0, r1);
		uint uiMax = max(r0, r1);
		r0 = uiMin;
		r1 = uiMax;

		uiMin = min(r3, r2);
		uiMax = max(r3, r2);
		r3 = uiMin;
		r2 = uiMax;

		uiMin = min(r2, r0);
		uiMax = max(r2, r0);
		r2 = uiMin;
		r0 = uiMax;

		uiMin = min(r3, r1);
		uiMax = max(r3, r1);
		r3 = uiMin;
		r1 = uiMax;

		uiMin = min(r1, r0);
		uiMax = max(r1, r0);
		r1 = uiMin;
		r0 = uiMax;

		uiMin = min(r3, r2);
		uiMax = max(r3, r2);
		r3 = uiMin;
		r2 = uiMax;

		uiMin = min(r5, r4);
		uiMax = max(r5, r4);
		r5 = uiMin;
		r4 = uiMax;

		uiMin = min(r7, r8);
		uiMax = max(r7, r8);
		r7 = uiMin;
		r8 = uiMax;

		uiMin = min(r6, r8);
		uiMax = max(r6, r8);
		r6 = uiMin;
		r8 = uiMax;

		uiMin = min(r6, r7);
		uiMax = max(r6, r7);
		r6 = uiMin;
		r7 = uiMax;

		uiMin = min(r4, r8);
		uiMax = max(r4, r8);
		r4 = uiMin;
		r8 = uiMax;

		uiMin = min(r4, r6);
		uiMax = max(r4, r6);
		r4 = uiMin;
		r6 = uiMax;

		uiMin = min(r5, r7);
		uiMax = max(r5, r7);
		r5 = uiMin;
		r7 = uiMax;

		uiMin = min(r4, r5);
		uiMax = max(r4, r5);
		r4 = uiMin;
		r5 = uiMax;

		uiMin = min(r6, r7);
		uiMax = max(r6, r7);
		r6 = uiMin;
		r7 = uiMax;

		uiMin = min(r0, r8);
		uiMax = max(r0, r8);
		r0 = uiMin;
		r8 = uiMax;

		r4 = max(r0, r4);
		r5 = max(r1, r5);

		r6 = max(r2, r6);
		r7 = max(r3, r7);

		r4 = min(r4, r6);
		r5 = min(r5, r7);

		//store found median into result
		uiResult |= min(r4, r5);


"""


def downscale_2d(opencl_provider, input, output, reduction='median'):
    """
    Compute a 2x downscaled image using either mean or median

    """
    image_x = input.shape[1]
    image_y = input.shape[0]

    if reduction == 'mean':
        reduction_code = '(input[i1]+input[i2]+input[i3]+input[i4])/4.0f'
    elif reduction == 'median':
        reduction_code = """
        float pixel1 = input[i1];
        float pixel2 = input[i2];
        float pixel3 = input[i3];
        float pixel4 = input[i4];
        sort(&(pixel1), &(pixel2), &(pixel3), &(pixel4));
        float value = (input[i2]+input[i3])/2.0f;
                         """

    program_code = f"""

      __kernel void feature_kernel(__global float *input, __global float *output)
      {{
          int ox = get_global_id(1);
          int oy = get_global_id(0);

          int x1 = min(2*ox,{image_x - 1});
          int x2 = min(2*ox+1,{image_x - 1});

          int y1 = min(2*oy,{image_y - 1});
          int y2 = min(2*oy+1,{image_y - 1});

          int i1 = x1 + {image_x}*y1;
          int i2 = x1 + {image_x}*y2;
          int i3 = x2 + {image_x}*y1;
          int i4 = x2 + {image_x}*y2;
          {reduction_code};


          int di = ox+{image_x / 2}*oy;
          output[di] = value;
      }}
      """
    # print(program_code)

    program = opencl_provider.build(median + program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(opencl_provider.queue, output.shape, None, input.data, output.data)
