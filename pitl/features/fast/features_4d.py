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

    exclude_center = (
        exclude_center
        and abs(dx) <= rx
        and abs(dy) <= ry
        and abs(dz) <= rz
        and abs(dw) <= rw
    )

    if optimisation and lx == 1 and ly == 1 and lz == 1 and lw == 1:
        program_code = f"""

        __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature)
         {{

           const int x = get_global_id(2);
           const int y = get_global_id(1);
           const int z = get_global_id(0);
           
           for(int w=0; w<{image_w}; w++)
           {{

               const int x0  = max(min(x+{dx}, {image_x - 1}),0);
               const int y0  = max(min(y+{dy}, {image_y - 1}),0);
               const int z0  = max(min(x+{dz}, {image_z - 1}),0);
               const int w0  = max(min(w+{dw}, {image_w - 1}),0);
               const int i0  = x0 + {image_x}*y0 + {image_x * image_y}*z0 + {image_x * image_y * image_z}*w0;
               const int i   = x  + {image_x}*y  + {image_x * image_y}*z  + {image_x * image_y * image_z}*w;
    
               const float value = x0<0||y0<0||z0<0||w0<0||x0>{image_x-1}||y0>{image_y-1}||z0>{image_z-1}||w0>{image_w-1} ? 0.0f : image[i0];
               feature[i] = {"0.0f" if exclude_center else "value" };
           }}
           
         }}
        """
    else:
        program_code = f"""

         __kernel void feature_kernel(__global float *image, __global float *integral, __global float *feature)
         {{

           const int x = get_global_id(2);
           const int y = get_global_id(1);
           const int z = get_global_id(0);
           
           for(int w=0; w<{image_w}; w++)
           {{
               const int x0   = min(x+{dx}-{rx}-1, {image_x - 1});
               const int x1   = min(x+{dx}+{rx},   {image_x - 1}); 
               const int x2   = min(x+{dx}-{rx}-1, {image_x - 1});
               const int x3   = min(x+{dx}+{rx},   {image_x - 1});
               const int x4   = min(x+{dx}-{rx}-1, {image_x - 1});
               const int x5   = min(x+{dx}+{rx},   {image_x - 1}); 
               const int x6   = min(x+{dx}-{rx}-1, {image_x - 1});
               const int x7   = min(x+{dx}+{rx},   {image_x - 1});
               const int x8   = min(x+{dx}-{rx}-1, {image_x - 1});
               const int x9   = min(x+{dx}+{rx},   {image_x - 1}); 
               const int x10  = min(x+{dx}-{rx}-1, {image_x - 1});
               const int x11  = min(x+{dx}+{rx},   {image_x - 1});
               const int x12  = min(x+{dx}-{rx}-1, {image_x - 1});
               const int x13  = min(x+{dx}+{rx},   {image_x - 1}); 
               const int x14  = min(x+{dx}-{rx}-1, {image_x - 1});
               const int x15  = min(x+{dx}+{rx},   {image_x - 1});    
    
               const int y0   = min(y+{dy}-{ry}-1, {image_y - 1});
               const int y1   = min(y+{dy}-{ry}-1, {image_y - 1}); 
               const int y2   = min(y+{dy}+{ry},   {image_y - 1});
               const int y3   = min(y+{dy}+{ry},   {image_y - 1}); 
               const int y4   = min(y+{dy}-{ry}-1, {image_y - 1});
               const int y5   = min(y+{dy}-{ry}-1, {image_y - 1}); 
               const int y6   = min(y+{dy}+{ry},   {image_y - 1});
               const int y7   = min(y+{dy}+{ry},   {image_y - 1}); 
               const int y8   = min(y+{dy}-{ry}-1, {image_y - 1});
               const int y9   = min(y+{dy}-{ry}-1, {image_y - 1}); 
               const int y10  = min(y+{dy}+{ry},   {image_y - 1});
               const int y11  = min(y+{dy}+{ry},   {image_y - 1}); 
               const int y12  = min(y+{dy}-{ry}-1, {image_y - 1});
               const int y13  = min(y+{dy}-{ry}-1, {image_y - 1}); 
               const int y14  = min(y+{dy}+{ry},   {image_y - 1});
               const int y15  = min(y+{dy}+{ry},   {image_y - 1}); 
    
               const int z0   = min(z+{dz}-{rz}-1, {image_z - 1});
               const int z1   = min(z+{dz}-{rz}-1, {image_z - 1}); 
               const int z2   = min(z+{dz}-{rz}-1, {image_z - 1});
               const int z3   = min(z+{dz}-{rz}-1, {image_z - 1}); 
               const int z4   = min(z+{dz}+{rz},   {image_z - 1});
               const int z5   = min(z+{dz}+{rz},   {image_z - 1}); 
               const int z6   = min(z+{dz}+{rz},   {image_z - 1});
               const int z7   = min(z+{dz}+{rz},   {image_z - 1}); 
               const int z8   = min(z+{dz}-{rz}-1, {image_z - 1});
               const int z9   = min(z+{dz}-{rz}-1, {image_z - 1}); 
               const int z10  = min(z+{dz}-{rz}-1, {image_z - 1});
               const int z11  = min(z+{dz}-{rz}-1, {image_z - 1}); 
               const int z12  = min(z+{dz}+{rz},   {image_z - 1});
               const int z13  = min(z+{dz}+{rz},   {image_z - 1}); 
               const int z14  = min(z+{dz}+{rz},   {image_z - 1});
               const int z15  = min(z+{dz}+{rz},   {image_z - 1}); 
               
               const int w0   = min(w+{dw}-{rw}-1, {image_w - 1});
               const int w1   = min(w+{dw}-{rw}-1, {image_w - 1}); 
               const int w2   = min(w+{dw}-{rw}-1, {image_w - 1});
               const int w3   = min(w+{dw}-{rw}-1, {image_w - 1}); 
               const int w4   = min(w+{dw}-{rw}-1, {image_w - 1});
               const int w5   = min(w+{dw}-{rw}-1, {image_w - 1}); 
               const int w6   = min(w+{dw}-{rw}-1, {image_w - 1});
               const int w7   = min(w+{dw}-{rw}-1, {image_w - 1}); 
               const int w8   = min(w+{dw}+{rw},   {image_w - 1});
               const int w9   = min(w+{dw}+{rw},   {image_w - 1}); 
               const int w10  = min(w+{dw}+{rw},   {image_w - 1});
               const int w11  = min(w+{dw}+{rw},   {image_w - 1}); 
               const int w12  = min(w+{dw}+{rw},   {image_w - 1});
               const int w13  = min(w+{dw}+{rw},   {image_w - 1}); 
               const int w14  = min(w+{dw}+{rw},   {image_w - 1});
               const int w15  = min(w+{dw}+{rw},   {image_w - 1}); 
    
               const uint i0  = x0  + {image_x} * y0  + {image_x * image_y} * z0  + {image_x * image_y * image_z} * w0;
               const uint i1  = x1  + {image_x} * y1  + {image_x * image_y} * z1  + {image_x * image_y * image_z} * w1;
               const uint i2  = x2  + {image_x} * y2  + {image_x * image_y} * z2  + {image_x * image_y * image_z} * w2;
               const uint i3  = x3  + {image_x} * y3  + {image_x * image_y} * z3  + {image_x * image_y * image_z} * w3;
               const uint i4  = x4  + {image_x} * y4  + {image_x * image_y} * z4  + {image_x * image_y * image_z} * w4;
               const uint i5  = x5  + {image_x} * y5  + {image_x * image_y} * z5  + {image_x * image_y * image_z} * w5;
               const uint i6  = x6  + {image_x} * y6  + {image_x * image_y} * z6  + {image_x * image_y * image_z} * w6;
               const uint i7  = x7  + {image_x} * y7  + {image_x * image_y} * z7  + {image_x * image_y * image_z} * w7;
               const uint i8  = x8  + {image_x} * y8  + {image_x * image_y} * z8  + {image_x * image_y * image_z} * w8;
               const uint i9  = x9  + {image_x} * y9  + {image_x * image_y} * z9  + {image_x * image_y * image_z} * w9;
               const uint i10 = x10 + {image_x} * y10 + {image_x * image_y} * z10 + {image_x * image_y * image_z} * w10;
               const uint i11 = x11 + {image_x} * y11 + {image_x * image_y} * z11 + {image_x * image_y * image_z} * w11;
               const uint i12 = x12 + {image_x} * y12 + {image_x * image_y} * z12 + {image_x * image_y * image_z} * w12;
               const uint i13 = x13 + {image_x} * y13 + {image_x * image_y} * z13 + {image_x * image_y * image_z} * w13;
               const uint i14 = x14 + {image_x} * y14 + {image_x * image_y} * z14 + {image_x * image_y * image_z} * w14;
               const uint i15 = x15 + {image_x} * y15 + {image_x * image_y} * z15 + {image_x * image_y * image_z} * w15;
    
               const int i = x + {image_x}*y + {image_x * image_y}*z + {image_x * image_y * image_z}*w;
    
               const float value0  = x0<0 ||y0<0 ||z0<0 ||w0<0  ? 0.0f : integral[i0];
               const float value1  = x1<0 ||y1<0 ||z1<0 ||w1<0  ? 0.0f : integral[i1];
               const float value2  = x2<0 ||y2<0 ||z2<0 ||w2<0  ? 0.0f : integral[i2];
               const float value3  = x3<0 ||y3<0 ||z3<0 ||w3<0  ? 0.0f : integral[i3];
               const float value4  = x4<0 ||y4<0 ||z4<0 ||w4<0  ? 0.0f : integral[i4];
               const float value5  = x5<0 ||y5<0 ||z5<0 ||w5<0  ? 0.0f : integral[i5];
               const float value6  = x6<0 ||y6<0 ||z6<0 ||w6<0  ? 0.0f : integral[i6];
               const float value7  = x7<0 ||y7<0 ||z7<0 ||w7<0  ? 0.0f : integral[i7];
               const float value8  = x8<0 ||y8<0 ||z8<0 ||w8<0  ? 0.0f : integral[i8];
               const float value9  = x9<0 ||y9<0 ||z9<0 ||w9<0  ? 0.0f : integral[i9];
               const float value10 = x10<0||y10<0||z10<0||w10<0 ? 0.0f : integral[i10];
               const float value11 = x11<0||y11<0||z11<0||w11<0 ? 0.0f : integral[i11];
               const float value12 = x12<0||y12<0||z12<0||w12<0 ? 0.0f : integral[i12];
               const float value13 = x13<0||y13<0||z13<0||w13<0 ? 0.0f : integral[i13];
               const float value14 = x14<0||y14<0||z14<0||w14<0 ? 0.0f : integral[i14];
               const float value15 = x15<0||y15<0||z15<0||w15<0 ? 0.0f : integral[i15];
               const float value16 = {"image[i]" if exclude_center else "0.0f"};
    
               const float value = (+1*(value15)
                                    -1*(value7+value11+value13+value14)
                                    +1*(value3+value5+value6+value9+value10+value12)
                                    -1*(value1+value2+value4+value8)
                                    +1*value0
                                    -value16) * {1.0 / (lx * ly * lz * lw)};
    

               feature[i] = value;
           }}
         }}
         """
    print(program_code)

    program = opencl_provider.build(program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(
        opencl_provider.queue,
        feature_gpu.shape[1:4],
        None,
        image_gpu.data,
        integral_image_gpu.data,
        feature_gpu.data,
    )

    pass
