median4 = """
inline void sort(float *a, float *b, float *c, float *d) 
{
   float swap;
   
   if(*a > *b) 
   {
      swap = *a;
      *a = *b;
      *b = swap;
   }
   
   if(*a > *c) 
   {
      swap = *a;
      *a = *c;
      *c = swap;
   }
   
   if(*b > *c) 
   {
      swap = *b;
      *b = *c;
      *c = swap;
   }
   
   if(*a > *d) 
   {
      swap = *a;
      *a = *d;
      *d = swap;
   }
   
   if(*b > *d) 
   {
      swap = *b;
      *b = *d;
      *d = swap;
   }
   
   if(*c > *d) 
   {
      swap = *c;
      *c = *d;
      *d = swap;
   }
}
  

"""


def downscale_1d(opencl_provider, input, output, reduction='mean'):
    """
    Compute a 2x downscaled image using either mean or median

    """
    image_x = input.shape[0]

    if reduction == 'mean' or reduction == 'median':
        reduction_code = '(input[i1]+input[i2])/2.0f'

    program_code = f"""

      __kernel void feature_kernel(__global float *input, __global float *output)
      {{
          int ox = get_global_id(0);
       
          int x1 = min(2*ox,{image_x-1});
          int x2 = min(2*ox+1,{image_x-1});
          
          int i1 = x1;
          int i2 = x2;
          float value = {reduction_code};
          
          int di = ox;
          output[di] = value;
      }}
      """
    # print(program_code)

    program = opencl_provider.build(median4 + program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(opencl_provider.queue, output.shape, None, input.data, output.data)


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
          
          
          int di = ox+{image_x/2}*oy;
          output[di] = value;
      }}
      """
    # print(program_code)

    program = opencl_provider.build(median4 + program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(opencl_provider.queue, output.shape, None, input.data, output.data)
