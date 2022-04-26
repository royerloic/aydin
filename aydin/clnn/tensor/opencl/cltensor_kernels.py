import warnings
from typing import Tuple

import numpy
import pyopencl
from pyopencl import cltypes
from pyopencl.array import Array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

from aydin.providers.opencl.opencl_provider import OpenCLProvider


def _plus_if(additive: bool) -> str:
    return '+' if additive else ''


_g_ = "__global "

wang_hash = """
    // Wang Hash function: https://riptutorial.com/opencl/example/20715/using-thomas-wang-s-integer-hash-function
    inline uint wang_hash(uint input, uint seed)
    {
        input += seed;
        input = (input ^ 61) ^ (input >> 16);
        input *= 9;
        input = input ^ (input >> 4);
        input *= 0x27d4eb2d;
        input = input ^ (input >> 15);
        return input;
    }
"""

wang_float = """
    // random number generator for dithering
    inline float wang_float(uint seed)
    {
        uint x = seed;
        x = (x ^ 61) ^ (x >> 16);
        x *= 9;
        x = x ^ (x >> 4);
        x *= 0x27d4eb2d;
        x = x ^ (x >> 15);
      
        float rnd = (x*1.0f)/4294967295.0f;
        
        return 2.0f*(rnd-0.5f);
    }
"""

fast_mod = """
    // Alternative to integer mod from here: https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    inline uint reduce(uint x, uint N) 
    {
        return uint(((ulong) x * (ulong) N) >> 32) ;
    }
"""


class CLTensorKernels:
    def __init__(self, provider: OpenCLProvider):
        """
        Constructs a tensor.
        """
        super().__init__()
        self.context = provider.context
        self.provider = provider

    def mul_add(self, s: Array, a: Array, alpha: float, beta: float, additive: bool):
        mul_add_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a, float alpha, float beta",
            f"s[i] {_plus_if(additive)}= alpha*a[i]+beta",
        )
        mul_add_kernel(s, a, alpha, beta)

    def diff(self, s: Array, a: Array, b: Array, alpha: float, additive: bool):
        diff_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a, {_g_}float *b, float alpha",
            f"s[i] {_plus_if(additive)}= alpha*(a[i]-b[i])",
        )
        diff_kernel(s, a, b, alpha)

    def diff_sign(self, s: Array, a: Array, b: Array, alpha: float, additive: bool):
        diff_sign_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a, {_g_}float *b, float alpha",
            f"s[i] {_plus_if(additive)}= alpha*sign(a[i]-b[i])",
        )
        diff_sign_kernel(s, a, b, alpha)

    def absolute_diff(self, s: Array, a: Array, b: Array, alpha: float, additive: bool):
        absolute_diff_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a, {_g_}float *b, float alpha",
            f"s[i] {_plus_if(additive)}= alpha*fabs(a[i]-b[i])",
        )
        absolute_diff_kernel(s, a, b, alpha)

    def squared_diff(
        self,
        s: Array,
        a: Array,
        b: Array,
        retain_sign: bool,
        alpha: float,
        additive: bool,
    ):
        self.power_diff(s, a, b, 2, retain_sign, alpha, additive)

    def power_diff(
        self,
        s: Array,
        a: Array,
        b: Array,
        p,
        retain_sign: bool,
        alpha: float,
        additive: bool,
    ):
        power_diff_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a, {_g_}float *b, float p, float alpha",
            f"s[i] {_plus_if(additive)}= alpha*pow(fabs(a[i]-b[i]), p){'*sign(a[i]-b[i])' if retain_sign else ''}",
        )
        power_diff_kernel(s, a, b, p, alpha)

    def signum_select(
        self,
        s: Array,
        a: Array,
        b: Array,
        c: Array,
        sb: float,
        sc: float,
        alpha: float,
        additive: bool,
    ):
        signum_select_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a, {_g_}float *b, {_g_}float *c, float sb, float sc, float alpha",
            f"s[i] {_plus_if(additive)}= alpha*( a[i]>0.0f ? sb*b[i] : sc*c[i] )",
        )
        signum_select_kernel(s, a, b, c, sb, sc, alpha)

    def relu(self, s: Array, a: Array):
        relu_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a", "s[i] = max(0.0f, a[i])"
        )
        relu_kernel(s, a)

    def abs(self, s: Array, a: Array):
        abs_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a", "s[i] = fabs(a[i])"
        )
        abs_kernel(s, a)

    def clip(self, s: Array, a: Array, a_min: float, a_max: float):
        clip_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a", f"s[i] = clamp(a[i], {a_min}f, {a_max}f)"
        )
        clip_kernel(s, a)

    def generalised_sum(
        self,
        s: Array,
        a: Array,
        b: Array,
        sa: float,
        sb: float,
        pa: float,
        pb: float,
        alpha: float,
        additive: bool,
    ):
        # 's+alpha*(sa*(a**pa)+sb*(b**pb)))'

        generalised_sum_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a, {_g_}float *b, float sa, float sb, float alpha",
            f"s[i] {_plus_if(additive)}= alpha*( sa*pow(a[i],{pa}f)+sb*pow(b[i],{pb}f) )",
        )
        generalised_sum_kernel(s, a, b, sa, sb, alpha)

    def generalised_product(
        self,
        s: Array,
        a: Array,
        b: Array,
        sa: float,
        sb: float,
        pa: float,
        pb: float,
        oa: float,
        ob: float,
        multiply: bool,
        alpha: float,
        additive: bool,
    ):
        # 's+alpha*( (sa*(a**pa) + oa) {operation} (sb*(b**pb) + ob) ))'

        generalised_product_kernel = self.provider.get_elwise_kernel(
            f"{_g_}float *s, {_g_}float *a, {_g_}float *b, float sa, float sb, float oa, float ob, float alpha",
            f"s[i] {_plus_if(additive)}= alpha*( (sa*pow(a[i],{pa}f)+oa) {'*' if multiply else '/'} (sb*pow(b[i],{pb}f)+ob) )",
        )
        generalised_product_kernel(s, a, b, sa, sb, oa, ob, alpha)

    def l2norm(self, s: Array) -> float:
        l2_kernel = self.provider.get_reduction_kernel(
            neutral=0,
            map_expression="pown(x[i],2)",
            reduce_expression="a+b",
            arguments="float *x",
        )

        sum_of_squares = l2_kernel(s).get().item(0)
        l2_norm = numpy.math.sqrt(sum_of_squares)
        return l2_norm

    def sum(self, s: Array, a: Array, axis: int = None) -> float:
        dim = len(a.shape)
        if axis is None:
            sum = pyopencl.array.sum(a)
            if len(s.shape) == 1:
                s[0] = sum
                return
            elif len(s.shape) == 2:
                s[0, 0] = sum
                return

        elif axis < 0:
            axis += len(a.shape)

        if dim == 1:
            if len(s.shape) == 1:
                s[0] = sum
                return
            elif len(s.shape) == 2:
                s[0, 0] = sum
                return

        elif dim == 2:

            def get_program_code(length_v: int, stride_u: int, stride_v: int) -> str:
                return f"""
                        __kernel void custom_kernel({_g_}float *s, {_g_}float *a)
                         {{
                            const uint u = get_global_id(0);
                            
                            float sum = 0.0f;
                            for(uint v=0; v<{length_v}; v++)
                            {{ 
                                const uint i = {stride_u}*u + {stride_v}*v;
                                sum += a[i];
                            }}
                            s[u] = sum;
                         }}
                        """

            a_y, a_x = a.shape
            if axis == 0:
                stride_u = 1
                stride_v = a_x
                length_u = a_x
                length_v = a_y
            else:  # must be axis==1 for dimension 2:
                stride_u = a_x
                stride_v = 1
                length_u = a_y
                length_v = a_x

            program_code = get_program_code(length_v, stride_u, stride_v)

            sum_kernel = self.provider.get_kernel('sum_kernel', program_code)

            sum_kernel(self.provider.queue, (length_u,), None, s.data, a.data)
            return

        else:
            raise NotImplemented()

    def mean(self, s: Array, a: Array, axis: int = None) -> float:
        dim = len(a.shape)
        if axis is None:
            mean = pyopencl.array.sum(a) / a.size
            if len(s.shape) == 1:
                s[0] = mean
                return
            elif len(s.shape) == 2:
                s[0, 0] = mean
                return

        elif axis < 0:
            axis += len(a.shape)

        if dim == 1:
            mean = pyopencl.array.sum(a) / a.size
            if len(s.shape) == 1:
                s[0] = mean
                return
            elif len(s.shape) == 2:
                s[0, 0] = mean
                return

        elif dim == 2:

            def get_program_code(length_v: int, stride_u: int, stride_v: int) -> str:
                return f"""
                        __kernel void custom_kernel({_g_}float *s, {_g_}float *a)
                         {{
                            const uint u = get_global_id(0);

                            float sum = 0.0f;
                            for(int v=0; v<{length_v}; v++)
                            {{ 
                                const uint i = {stride_u}*u + {stride_v}*v;
                                sum += a[i];
                            }}
                            s[u] = sum/{length_v};
                         }}
                        """

            a_y, a_x = a.shape
            if axis == 0:
                stride_u = 1
                stride_v = a_x
                length_u = a_x
                length_v = a_y
            else:  # must be axis==1 for dimension 2:
                stride_u = a_x
                stride_v = 1
                length_u = a_y
                length_v = a_x

            program_code = get_program_code(length_v, stride_u, stride_v)

            mean_kernel = self.provider.get_kernel('mean_kernel', program_code)

            mean_kernel(self.provider.queue, (length_u,), None, s.data, a.data)
            return

        else:
            raise NotImplemented()

    def dot(
        self,
        s: Array,
        a: Array,
        b: Array,
        ta: bool = False,
        tb: bool = False,
        additive: bool = False,
    ):
        self.affine(s, a, b, None, ta, tb, additive)

    def affine(
        self,
        s: Array,
        a: Array,
        b: Array,
        c: Array,
        ta: bool = False,
        tb: bool = False,
        additive: bool = False,
    ):

        dim_a = len(a.shape)
        dim_b = len(b.shape)
        dim_c = len(c.shape) if not c is None else 0

        just_dot: bool = c is None

        def get_program_code(
            length_k: int,
            stride_a: Tuple[int],
            stride_b: Tuple[int],
            stride_c: Tuple[int],
            stride_s: Tuple[int],
        ) -> str:
            return f"""
                    __kernel void custom_kernel({_g_}float *s, {_g_}float *a, {_g_}float *b, {_g_}float *c)
                     {{
                        const uint u = get_global_id(1);
                        const uint v = get_global_id(0);

                        float sum = 0.0f;
                        for(uint k=0; k<{length_k}; k++)
                        {{
                            const uint ia = {stride_a[0]}*k + {stride_a[1]}*v;
                            float value_a = a[ia];
                            const uint ib = {stride_b[0]}*u + {stride_b[1]}*k;
                            float value_b = b[ib];
                            sum += value_a * value_b;
                        }}

                        const uint is = {stride_s[0]}*u + {stride_s[1]}*v;
                        const uint ic = {stride_c[0]}*u + {stride_c[1]}*v;
                        s[is] {_plus_if(additive)}= sum {'' if just_dot else '+c[ic]'};
                     }}
                    """

        if dim_a == 1 and dim_b == 1:
            assert a.shape == b.shape

            dor_two_vectors_kernel = self.provider.get_reduction_kernel(
                neutral=0,
                map_expression="x[i]*y[i]",
                reduce_expression="a+b",
                arguments="float *x, float *y",
            )

            result = dor_two_vectors_kernel(a, b)
            if not just_dot:
                result += c[0]
            s.ravel()[0] = result

        elif dim_a == 2 or dim_b == 2:

            length_k = a.shape[0] if ta else a.shape[-1]
            stride_a = (a.shape[-1], 1) if ta else (1, a.shape[-1])
            stride_b = (b.shape[-1], 1) if tb else (1, b.shape[-1])
            stride_s = (1, s.shape[-1])
            stride_c = (1, stride_s if dim_c == 2 else 0)

            # get_program_code(length_k: int, stride_u: int, stride_v: int, stride_a: int, stride_b: int) -> str:
            program_code = get_program_code(
                length_k, stride_a, stride_b, stride_c, stride_s
            )
            affine_kernel = self.provider.get_kernel('affine_kernel', program_code)
            if just_dot:
                c = a
            affine_kernel(
                self.provider.queue, s.shape, None, s.data, a.data, b.data, c.data
            )
            return

        else:
            raise NotImplemented()

    def sample(self, s: Array, a: Array, seed: int):
        def get_program_code(
            size: int, length_u: int, stride_s: int, stride_a: int
        ) -> str:
            return f"""

                    {wang_hash}
                    {fast_mod}
            
                    __kernel void custom_kernel({_g_}float *s, {_g_}float *a, uint seed)
                     {{
                        const uint v = get_global_id(0);
                        const uint w = reduce(wang_hash(v, seed), {size});

                        float sum = 0.0f;
                        for(uint u=0; u<{length_u}; u++)
                        {{ 
                            const uint i = u + {stride_s}*v;
                            const uint j = u + {stride_a}*w;
                            s[i] = a[j];
                        }}
                     }}
                    """

        assert s.shape[1] == a.shape[1]
        length = min(s.shape[1], a.shape[1])
        program_code = get_program_code(a.shape[0], length, s.shape[1], a.shape[1])
        sample_kernel = self.provider.get_kernel('sample_kernel', program_code)
        sample_kernel(
            self.provider.queue, (s.shape[0],), None, s.data, a.data, cltypes.uint(seed)
        )

    def copy_from(self, s: Array, a: Array, begin: int, end: int):
        def get_program_code(dst_stride: int, src_stride: int) -> str:
            return f"""
                        __kernel void custom_kernel({_g_}float *s, {_g_}float *a)
                         {{
                            const uint v = get_global_id(0);
                            for(uint u={begin}; u<{end}; u++)
                            {{ 
                                const uint i = u-{begin} + {dst_stride}*v;
                                const uint j = u         + {src_stride}*v;
                                s[i] = a[j];
                            }}
                         }}
                        """

        program_code = get_program_code(s.shape[1], a.shape[1])
        copy_from_kernel = self.provider.get_kernel('copy_from_kernel', program_code)
        length = min(s.shape[0], a.shape[0])
        copy_from_kernel(self.provider.queue, (length,), None, s.data, a.data)

    def uniform_noise(self, s: Array, a: Array, noise_level: float, seed: int):
        def get_program_code(length_u: int, dst_stride: int, src_stride: int) -> str:
            return f"""
                    {wang_hash}
                    {wang_float}
                    __kernel void custom_kernel({_g_}float *s, {_g_}float *a, uint seed)
                     {{
                        const uint v = get_global_id(0);
                        for(uint u=0; u<{length_u}; u++)
                        {{ 
                            const uint i = u + {dst_stride}*v;
                            const float noise  = {noise_level}f*wang_float(seed+{length_u}+wang_hash(u,seed)+wang_hash(v,seed)) ;
                            //printf("noise=%f \\n", noise);
                            s[i] = a[i] + noise;
                        }}
                     }}
                    """

        length_u = s.shape[1]
        length_v = s.shape[0]
        program_code = get_program_code(length_u, s.shape[1], a.shape[1])
        uniform_noise_kernel = self.provider.get_kernel(
            'uniform_noise_kernel', program_code
        )

        uniform_noise_kernel(
            self.provider.queue, (length_v,), None, s.data, a.data, cltypes.uint(seed)
        )
