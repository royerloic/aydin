import numpy
from pyopencl import cltypes


def collect_feature_4d(
    opencl_provider,
    image_gpu,
    integral_image_gpu,
    feature_gpu,
    dx,
    dy,
    dz,
    dw,
    lx,
    ly,
    lz,
    lw,
    exclude_center=True,
    mean: float = 0,
    optimisation=True,
):
    """
       Compute a given feature for a displacement (dx,dy,dz) relative to the center pixel, and a patch size (lx,ly,lz)

    """

    image_w, image_z, image_y, image_x = image_gpu.shape
    feature_w, feature_z, feature_y, feature_x = feature_gpu.shape

    assert image_x == feature_x
    assert image_y == feature_y
    assert image_z == feature_z
    assert image_w == feature_w

    rx = lx // 2
    ry = ly // 2
    rz = lz // 2
    rw = lw // 2

    if optimisation and lx == 1 and ly == 1 and lz == 1 and lw == 1:
        program_code = f"""

        __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
         {{

           const int x = get_global_id(2);
           const int y = get_global_id(1);
           const int z = get_global_id(0);
           
           for(int w=0; w<{image_w}; w++)
           {{

               const int x0  = x+{dx};
               const int y0  = y+{dy};
               const int z0  = z+{dz};
               const int w0  = w+{dw};
               
               const long i0  = x0 + {image_x}*y0 + {image_x * image_y}*z0 + {image_x * image_y * image_z}*w0;
               const float value = x0<0||y0<0||z0<0||w0<0||x0>{image_x-1}||y0>{image_y-1}||z0>{image_z-1}||w0>{image_w-1} ? 0.0f : image[i0];
               
               const long i   = x  + {image_x}*y  + {image_x * image_y}*z  + {image_x * image_y * image_z}*w;
               feature[i] = {"x0==x && y0==y && z0==z && w0==w ? 0.0f : value" if exclude_center else "value"};
           }}
           
         }}
        """
    else:
        program_code = f"""

        inline float integral_lookup(__global float *integral, int x, int y, int z, int w)
        {{
            x = min(x, {image_x-1});
            y = min(y, {image_y-1});
            z = min(z, {image_z-1});
            w = min(w, {image_w-1});
            
            const long i = x + {image_x} * y + {image_x*image_y} * z  + {image_x*image_y*image_z} * w;
            
            return (x<0||y<0||z<0||w<0) ? 0.0f : integral[i];
        }}

         __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
         {{

           const int x = get_global_id(2);
           const int y = get_global_id(1);
           const int z = get_global_id(0);
           
           for(int w=0; w<{image_w}; w++)
           {{
                const int xl   = x+{dx}-{rx}-1;
                const int xh   = x+{dx}+{rx}  ; 
                
                const int yl   = y+{dy}-{ry}-1; 
                const int yh   = y+{dy}+{ry}  ;
                
                const int zl   = z+{dz}-{rz}-1; 
                const int zh   = z+{dz}+{rz}  ;
                
                const int wl   = w+{dw}-{rw}-1; 
                const int wh   = w+{dw}+{rw}  ;
                
                const float value0  = integral_lookup(integral, xl, yl, zl, wl);
                const float value1  = integral_lookup(integral, xh, yl, zl, wl);
                const float value2  = integral_lookup(integral, xl, yh, zl, wl);
                const float value3  = integral_lookup(integral, xh, yh, zl, wl);
                const float value4  = integral_lookup(integral, xl, yl, zh, wl);
                const float value5  = integral_lookup(integral, xh, yl, zh, wl);
                const float value6  = integral_lookup(integral, xl, yh, zh, wl);
                const float value7  = integral_lookup(integral, xh, yh, zh, wl);
                const float value8  = integral_lookup(integral, xl, yl, zl, wh);
                const float value9  = integral_lookup(integral, xh, yl, zl, wh);
                const float value10 = integral_lookup(integral, xl, yh, zl, wh);
                const float value11 = integral_lookup(integral, xh, yh, zl, wh);
                const float value12 = integral_lookup(integral, xl, yl, zh, wh);
                const float value13 = integral_lookup(integral, xh, yl, zh, wh);
                const float value14 = integral_lookup(integral, xl, yh, zh, wh);
                const float value15 = integral_lookup(integral, xh, yh, zh, wh);
               
                const bool center_inside = (xl<x) && (x<=xh) && (yl<y) && (y<=yh) && (zl<z) && (z<=zh) && (wl<w) && (w<=wh);
                const bool all_inside    = (xl<0) && ({image_x-1}<=xh) && (yl<0) && ({image_y-1}<=yh) && (zl<0) && ({image_z-1}<=zh) && (wl<0) && ({image_w-1}<=wh); 
                const bool exclude_center = {"center_inside && !all_inside" if exclude_center else "false"}; 
                
                const long raw_volume = (xh-xl)*(yh-yl)*(zh-zl)*(wh-wl);
                const long volume = raw_volume - (exclude_center ? 1 : 0);
                
                const long i = x + {image_x}*y + {image_x * image_y}*z + {image_x * image_y * image_z}*w;  
                const float center = exclude_center ? image[i] : 0.0f;  
                
                const float value = (+1*(value15)
                                     -1*(value7+value11+value13+value14)
                                     +1*(value3+value5+value6+value9+value10+value12)
                                     -1*(value1+value2+value4+value8)
                                     +1*value0
                                     -center
                                    ) / volume
                                    + mean;
    

               feature[i] = clamp(value, 0.0f, nextafter(1.0f,0.0f));
           }}
         }}
         """
    # print(program_code)

    program = opencl_provider.build(program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(
        opencl_provider.queue,
        feature_gpu.shape[1:4],
        None,
        image_gpu.data,
        integral_image_gpu.data,
        feature_gpu.data,
        mean if isinstance(mean, numpy.ndarray) else cltypes.float(mean),
    )

    pass
