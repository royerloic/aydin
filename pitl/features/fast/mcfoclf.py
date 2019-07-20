import tempfile

import numpy
import numpy as np
import psutil
import pyopencl as cl
from pyopencl.array import to_device, Array

from pitl.features.fast.features_1d import collect_feature_1d
from pitl.features.fast.features_2d import collect_feature_2d
from pitl.features.fast.features_3d import collect_feature_3d
from pitl.features.fast.features_4d import collect_feature_4d
from pitl.features.fast.integral import (
    integral_1d,
    integral_2d,
    integral_3d,
    integral_4d,
)
from pitl.opencl.opencl_provider import OpenCLProvider
from pitl.util.nd import nd_range


class FastMultiscaleConvolutionalFeatures:
    """
    Multiscale convolutional feature generator.
    Uses OpenCL to acheive very fast integral image based feature generation.

    #TODO: There is some residual non-J-invariance on the borders...
    #TODO: There is still a weird issue in the 4D case... numerical overflow?

    """

    def __init__(
        self,
        opencl_provider=OpenCLProvider(),
        kernel_widths=[3, 3, 3, 3, 3, 3, 3, 3],
        kernel_scales=[1, 3, 7, 15, 31, 63, 127, 255],
        kernel_shapes=None,
        kernel_reductions=None,
        exclude_center=False,
    ):
        """
        Constructs a multiscale convolutional feature generator that uses OpenCL.


        :param opencl_provider:
        :type opencl_provider:
        :param kernel_widths:
        :type kernel_widths:
        :param kernel_scales:
        :type kernel_scales:
        :param kernel_shapes:
        :type kernel_shapes:
        :param exclude_center:
        :type exclude_center:

        """

        self.check_nans = False
        self.debug_log = True
        self.debug_force_memmap = False

        self.opencl_provider = opencl_provider

        self.kernel_widths = kernel_widths
        self.kernel_scales = kernel_scales
        self.kernel_shapes = (
            ['l2'] * len(kernel_widths) if kernel_shapes is None else kernel_shapes
        )
        self.kernel_reductions = (
            ['sum'] * len(kernel_widths)
            if kernel_reductions is None
            else kernel_reductions
        )
        self.exclude_center = exclude_center

    def get_free_mem(self):
        return self.opencl_provider.device.global_mem_size

    def get_needed_mem(self, num_elements):
        # We keep a 20% buffer to be confortable...
        return int(1.2 * 4 * num_elements)

    def is_enough_memory(self, num_elements):
        return (
            self.get_needed_mem(num_elements)
            < self.opencl_provider.device.global_mem_size
        )

    def get_receptive_field_radius(self):

        radii = max(
            [
                width * scale // 2
                for width, scale in zip(self.kernel_widths, self.kernel_scales)
            ]
        )
        return radii

    def compute(self, image, batch_dims=None, features=None):
        """
        Computes the features given an image. If the input image is of shape (d,h,w),
        resulting features are of shape (d,h,w,n) where n is the number of features.
        :param image: image for which features are computed
        :type image:
        :return: feature array
        :rtype:
        """

        # Checking NaNs just in case:
        if self.check_nans and np.isnan(np.sum(image)):
            raise Exception(f'NaN values occur in image!')

        image_dimension = len(image.shape)

        # set default batch_dim value:
        if batch_dims is None:
            batch_dims = (False,) * image_dimension

        # permutate dimensions in image to consolidate batch dimensions at the front:
        batch_dim_axes = [i for i in range(0, image_dimension) if batch_dims[i]]
        non_batch_dim_axes = [i for i in range(0, image_dimension) if not batch_dims[i]]
        axes_permutation = batch_dim_axes + non_batch_dim_axes
        image = numpy.transpose(image, axes=axes_permutation)
        nb_batch_dim = sum([(1 if i else 0) for i in batch_dims])
        nb_non_batch_dim = image_dimension - nb_batch_dim

        image_batch = None
        image_batch_gpu = None
        image_integral_gpu = None
        image_integral_gpu_temp = None
        features = None
        feature_gpu = None

        # We iterate over batches:
        for index in np.ndindex(*(image.shape[0:nb_batch_dim])):

            image_batch_slice = (*index, *(slice(None),) * nb_non_batch_dim)
            feature_batch_slice = (
                slice(None),
                *index,
                *(slice(None),) * nb_non_batch_dim,
            )
            # print(image_batch_slice)

            if image_batch_slice == (slice(None, None, None), slice(None, None, None)):
                # if there is only one batch, no need to do anything...
                image_batch = np.array(image, copy=True, dtype=numpy.float32)
            else:
                # Copy because image[image_batch_slice] is not necessarily contiguous and pyOpenCL does not like discontiguous arrays:
                if image_batch is None:
                    image_batch = np.array(
                        image[image_batch_slice], copy=True, dtype=numpy.float32
                    )
                else:
                    # Here we need to explicitly transfer values without creating an instance!
                    numpy.copyto(
                        image_batch, image[image_batch_slice], casting='unsafe'
                    )

            # We move the image to the GPU. Needs to fit entirely, could be a problem for very very large images.
            if image_batch_gpu is None:
                image_batch_gpu = to_device(self.opencl_provider.queue, image_batch)
            else:
                image_batch_gpu.set(image_batch, self.opencl_provider.queue)

            # This array on the GPU will host a single feature.
            # We will use that as temp destination for each feature generated on the GPU.
            if feature_gpu is None:
                feature_gpu = Array(
                    self.opencl_provider.queue, image_batch_gpu.shape, np.float32
                )

                # we also allocate the integral images:
                image_integral_gpu_1 = Array(
                    self.opencl_provider.queue, image_batch_gpu.shape, np.float32
                )
                image_integral_gpu_2 = Array(
                    self.opencl_provider.queue, image_batch_gpu.shape, np.float32
                )

            # Checking that the number of dimensions is within the bounds of what we can do:
            if nb_non_batch_dim <= 4:

                # We also compute the rest of pyramid for the present batch image:
                image_integral_gpu = self.compute_integral_image(
                    image_batch_gpu, image_integral_gpu_1, image_integral_gpu_2
                )

                # with napari.gui_qt():
                #      viewer = Viewer()
                #      viewer.add_image(image_batch_gpu.get(), name='image')
                #      viewer.add_image(rescale_intensity(image_integral_gpu.get(), in_range='image', out_range=(0, 1)), name='integral')

                # We first do a dry run to compute the number of features:
                nb_features = self.collect_features_nD(
                    image_batch_gpu, image_integral_gpu, nb_non_batch_dim
                )
                if self.debug_log:
                    print(f'Number of features:  {nb_features}')

                if features is None:
                    # At this point we know how big is the whole feature array, so we create it (encompasses all batches)
                    # This happens only once, the first time:
                    features = self.create_feature_array(image, nb_features)

                # we compute one batch of features:
                self.collect_features_nD(
                    image_batch_gpu,
                    image_integral_gpu,
                    nb_non_batch_dim,
                    feature_gpu,
                    features[feature_batch_slice],
                )

            else:  # We only support 1D, 2D, 3D, and 4D.
                raise Exception(
                    f'dimension above {image_dimension} for non nbatch dimensions not yet implemented!'
                )

        del feature_gpu
        del image_batch_gpu

        # Creates a view of the array in which the features are indexed by the last dimension:
        features = np.moveaxis(features, 0, -1)

        # permutate back axes:
        axes_inverse_permutation = [
            axes_permutation.index(l) for l in range(len(axes_permutation))
        ] + [image_dimension]
        features = numpy.transpose(features, axes=axes_inverse_permutation)

        return features

    def create_feature_array(self, image, nb_features):
        """
        Creates a feature array of the right size and possibly in a 'lazy' way using memory mapping.


        :param image: image for which features are created
        :type image:
        :param nb_features: number of features needed
        :type nb_features:
        :return: feature array
        :rtype:
        """
        if self.debug_log:
            print(f'Creating feature array...')

        size_in_bytes = nb_features * image.size * image.itemsize
        free_mem_in_bytes = psutil.virtual_memory().free
        if self.debug_log:
            print(f'There is {int(free_mem_in_bytes/1E6)} MB of free memory')
            print(f'Feature array is {(size_in_bytes/1E6)} MB.')

        # We take the heuristic that we need twice the amount of memory available to be confortable:
        is_enough_memory = 2 * size_in_bytes < free_mem_in_bytes

        # That's the shape we need:
        shape = (nb_features,) + image.shape

        if not self.debug_force_memmap and is_enough_memory:
            if self.debug_log:
                print(
                    f'There is enough memory -- we do not need to use a mem mapped array.'
                )
            array = np.zeros(shape, dtype=np.float32)

        else:
            if self.debug_log:
                print(f'There is not enough memory -- we will use a mem mapped array.')
            temp_file = tempfile.TemporaryFile()
            array = np.memmap(temp_file, dtype=np.float32, mode='w+', shape=shape)

        return array

    def compute_integral_image(
        self, image_gpu, image_integral_gpu_1, image_integral_gpu_2
    ):

        dim = len(image_gpu.shape)

        if dim == 1:
            return integral_1d(self.opencl_provider, image_gpu, image_integral_gpu_1)

        elif dim == 2:
            return integral_2d(
                self.opencl_provider,
                image_gpu,
                image_integral_gpu_1,
                image_integral_gpu_2,
            )

        elif dim == 3:
            return integral_3d(
                self.opencl_provider,
                image_gpu,
                image_integral_gpu_1,
                image_integral_gpu_2,
            )

        elif dim == 4:
            return integral_4d(
                self.opencl_provider,
                image_gpu,
                image_integral_gpu_1,
                image_integral_gpu_2,
            )

    def collect_features_nD(
        self, image_gpu, image_integral_gpu, ndim, feature_gpu=None, features=None
    ):
        """
        Collects nD features, one by one.
        If features is None, it just  counts the number of features so that the right size array
        can be allocated externally and then this method is called again this time with features != None


        :param image_gpu: gpu array of image to collect features from
        :param image_integral_gpu: gpu array to collect features from  (corresponding integral image)
        :param feature_gpu:  gpu array to use as temporary receptacle
        :param features: cpu features array to store all features to
        :return: number of features or the features themselves depending on the value of features (None or not None)
        """

        if self.debug_log:
            if features is None:
                print(f"Counting the number of features...")
            else:
                print(f"Computing features...")

        feature = None

        feature_index = 0
        for width, scale, shape, reduction in zip(
            self.kernel_widths,
            self.kernel_scales,
            self.kernel_shapes,
            self.kernel_reductions,
        ):
            radius = width // 2

            features_shifts = list(nd_range(-radius, +radius + 1, ndim))

            # print(f'Feature shifts: {features_shifts}')

            for shift in features_shifts:

                if self.exclude_center and scale == 1 and shift == (0,) * ndim:
                    continue

                if shape == 'l1' and sum([abs(i) for i in shift]) > radius:
                    continue
                elif shape == 'l2' and sum([i * i for i in shift]) > radius * radius:
                    continue
                elif shape == 'li':
                    pass

                if features is not None:
                    if self.debug_log:
                        print(
                            f"(width={width}, scale={scale}, shift={shift}, shape={shape}, reduction={reduction})"
                        )

                    params = (
                        self.opencl_provider,
                        image_gpu,
                        image_integral_gpu,
                        feature_gpu,
                        *[i * scale for i in shift],
                        *(scale,) * ndim,
                        self.exclude_center,
                    )
                    if ndim == 1:
                        collect_feature_1d(*params)
                    elif ndim == 2:
                        collect_feature_2d(*params)
                    elif ndim == 3:
                        collect_feature_3d(*params)
                    elif ndim == 4:
                        collect_feature_4d(*params)

                    if feature is None:
                        feature = numpy.zeros(features.shape[1:], dtype=numpy.float32)

                    cl.enqueue_copy(
                        self.opencl_provider.queue, feature, feature_gpu.data
                    )

                    features[feature_index] = feature

                    # with napari.gui_qt():
                    #     viewer = Viewer()
                    #     viewer.add_image(image_gpu.get(), name='image')
                    #     viewer.add_image(feature, name='feature')

                    if self.check_nans and np.isnan(np.sum(features[feature_index])):
                        raise Exception(f'NaN values occur in features!')

                feature_index += 1

        if features is not None:
            return features
        else:
            return feature_index
