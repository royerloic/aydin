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
    lx,
    ly,
    lz,
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

    rx = lx // 2
    ry = ly // 2
    rz = lz // 2

    exclude_center = (
        exclude_center and abs(dx) <= rx and abs(dy) <= ry and abs(dz) <= rz
    )

    if optimisation and lx == 1 and ly == 1 and lz == 1:
        program_code = f"""
        
        __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
         {{

           const int x = get_global_id(2);
           const int y = get_global_id(1);
           const int z = get_global_id(0);
           
           const int x0  = x+{dx};
           const int y0  = y+{dy};
           const int z0  = x+{dz};
           const int i0 = x0 + {image_x}*y0 + {image_x * image_y}*z0;
           const int i = x + {image_x}*y + {image_x * image_y}*z;
           
           const float value = x0<0||y0<0||z0<0||x0>{image_x-1}||y0>{image_y-1}||z0>{image_z-1} ? 0.0f : image[i0];
           feature[i] = {"0.0f" if exclude_center else "value" };
         }}
        """
    else:
        program_code = f"""

         __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature, float mean)
         {{

           const int x = get_global_id(2);
           const int y = get_global_id(1);
           const int z = get_global_id(0);

           const int x0  = min(x+{dx}-{rx}-1, {image_x - 1});
           const int x1  = min(x+{dx}+{rx},   {image_x - 1}); 
           const int x2  = min(x+{dx}-{rx}-1, {image_x - 1});
           const int x3  = min(x+{dx}+{rx},   {image_x - 1});
           const int x4  = min(x+{dx}-{rx}-1, {image_x - 1});
           const int x5  = min(x+{dx}+{rx},   {image_x - 1}); 
           const int x6  = min(x+{dx}-{rx}-1, {image_x - 1});
           const int x7  = min(x+{dx}+{rx},   {image_x - 1});  

           const int y0  = min(y+{dy}-{ry}-1, {image_y - 1});
           const int y1  = min(y+{dy}-{ry}-1, {image_y - 1}); 
           const int y2  = min(y+{dy}+{ry},   {image_y - 1});
           const int y3  = min(y+{dy}+{ry},   {image_y - 1}); 
           const int y4  = min(y+{dy}-{ry}-1, {image_y - 1});
           const int y5  = min(y+{dy}-{ry}-1, {image_y - 1}); 
           const int y6  = min(y+{dy}+{ry},   {image_y - 1});
           const int y7  = min(y+{dy}+{ry},   {image_y - 1}); 
           
           const int z0  = min(z+{dz}-{rz}-1, {image_z - 1});
           const int z1  = min(z+{dz}-{rz}-1, {image_z - 1}); 
           const int z2  = min(z+{dz}-{rz}-1, {image_z - 1});
           const int z3  = min(z+{dz}-{rz}-1, {image_z - 1}); 
           const int z4  = min(z+{dz}+{rz},   {image_z - 1});
           const int z5  = min(z+{dz}+{rz},   {image_z - 1}); 
           const int z6  = min(z+{dz}+{rz},   {image_z - 1});
           const int z7  = min(z+{dz}+{rz},   {image_z - 1}); 
        
           const uint i0 = x0 + {image_x} * y0 + {image_x * image_y} * z0;
           const uint i1 = x1 + {image_x} * y1 + {image_x * image_y} * z1;
           const uint i2 = x2 + {image_x} * y2 + {image_x * image_y} * z2;
           const uint i3 = x3 + {image_x} * y3 + {image_x * image_y} * z3;
           const uint i4 = x4 + {image_x} * y4 + {image_x * image_y} * z4;
           const uint i5 = x5 + {image_x} * y5 + {image_x * image_y} * z5;
           const uint i6 = x6 + {image_x} * y6 + {image_x * image_y} * z6;
           const uint i7 = x7 + {image_x} * y7 + {image_x * image_y} * z7;
           
           const int i = x + {image_x}*y + {image_x * image_y}*z;
           
           const float value0 = x0<0||y0<0||z0<0 ? 0.0f : integral[i0];
           const float value1 = x1<0||y1<0||z1<0 ? 0.0f : integral[i1];
           const float value2 = x2<0||y2<0||z2<0 ? 0.0f : integral[i2];
           const float value3 = x3<0||y3<0||z3<0 ? 0.0f : integral[i3];
           const float value4 = x4<0||y4<0||z4<0 ? 0.0f : integral[i4];
           const float value5 = x5<0||y5<0||z5<0 ? 0.0f : integral[i5];
           const float value6 = x6<0||y6<0||z6<0 ? 0.0f : integral[i6];
           const float value7 = x7<0||y7<0||z7<0 ? 0.0f : integral[i7];
           //float value8 = {"image[i]" if exclude_center else f"(x<{-dx} || y<{-dy} || z<{-dz} || x>={image_x-1-dx} || y>={image_y-1-dy} || z>={image_z-1-dz}) ? image[i] : 0.0f"};
           float value8 = {"image[i]" if exclude_center else f"0.0f"};

           const float adj = {lx*ly*lz-1 if exclude_center else lx*ly*lz}*mean;           

           float value = (+1*(value7)
                                -1*(value6+value5+value3)
                                +1*(value1+value2+value4)
                                -1*value0
                                -value8
                                +adj)*{1.0 / (lx * ly * lz)};
                                
           
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
