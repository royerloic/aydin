# flake8: noqa

import numpy
from pyopencl import cltypes


def collect_feature_3d(
    opencl_provider,
    image_gpu,
    integral_image_gpu,
    feature_gpu,
    dx,
    dy,
    dz,
    nx,
    ny,
    nz,
    px,
    py,
    pz,
    exclude_center=True,
    mean: float = 0,
    optimisation=True,
):
    """
       Compute a given feature for a displacement (dx,dy,dz) relative to the center pixel, and a patch size (lx,ly,lz)

    """

    image_z, image_y, image_x = image_gpu.shape
    feature_z, feature_y, feature_x = feature_gpu.shape

    assert image_x == feature_x
    assert image_y == feature_y
    assert image_z == feature_z

    if optimisation and nx + px == 1 and ny + py == 1 and nz + pz == 1:
        program_code = f"""
        
        __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
         {{

           const int x = get_global_id(2);
           const int y = get_global_id(1);
           const int z = get_global_id(0);
           
           const int x0  = x+{dx};
           const int y0  = y+{dy};
           const int z0  = z+{dz};
           
           const long i0  = x0 + {image_x}*y0 + {image_x * image_y}*z0;
           const float value = (x0<0||y0<0||z0<0||x0>{image_x-1}||y0>{image_y-1}||z0>{image_z-1}) ? 0.0f : image[i0];
           
           const long i   = x  + {image_x}*y  + {image_x * image_y}*z;           
           feature[i] = {"x0==x && y0==y && z0==z ? 0.0f : value" if exclude_center else "value"};
         }}
        """
    else:
        program_code = f"""
        
        inline float integral_lookup(__global float *integral, int x, int y, int z)
        {{
            x = min(x, {image_x-1});
            y = min(y, {image_y-1});
            z = min(z, {image_z-1});
            
            long i = x + {image_x} * y + {image_x*image_y} * z;
            
            return (x<0||y<0||z<0) ? 0.0f : integral[i];
        }}

         __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
         {{

            const int x = get_global_id(2);
            const int y = get_global_id(1);
            const int z = get_global_id(0);
            
            const int xl  = x+{dx}-{nx}-1;
            const int xh  = x+{dx}+{px}  ; 
            
            const int yl  = y+{dy}-{ny}-1; 
            const int yh  = y+{dy}+{py}  ;
            
            const int zl  = z+{dz}-{nz}-1; 
            const int zh  = z+{dz}+{pz}  ;
            
            const float value0 = integral_lookup(integral,xl, yl, zl);
            const float value1 = integral_lookup(integral,xh, yl, zl);
            const float value2 = integral_lookup(integral,xl, yh, zl);
            const float value3 = integral_lookup(integral,xh, yh, zl);
            const float value4 = integral_lookup(integral,xl, yl, zh);
            const float value5 = integral_lookup(integral,xh, yl, zh);
            const float value6 = integral_lookup(integral,xl, yh, zh);
            const float value7 = integral_lookup(integral,xh, yh, zh);
           
            const bool center_inside = (xl<x) && (x<=xh) && (yl<y) && (y<=yh) && (zl<z) && (z<=zh);
            const bool all_inside    = (xl<0) && ({image_x-1}<=xh) && (yl<0) && ({image_y-1}<=yh) && (zl<0) && ({image_z-1}<=zh); 
            const bool exclude_center = {"center_inside && !all_inside" if exclude_center else "false"}; 
    
            const long raw_volume = (xh-xl)*(yh-yl)*(zh-zl);
            const long volume = raw_volume - (exclude_center ? 1 : 0);
    
            const long i = x + {image_x}*y + {image_x * image_y}*z;  
            const float center = exclude_center ? image[i] : 0.0f;        


            float value = ( +1*(value7)
                            -1*(value6+value5+value3)
                            +1*(value1+value2+value4)
                            -1*value0
                            -center
                           )/volume
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

    pass
