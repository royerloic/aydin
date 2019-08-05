import math
import tempfile

import napari
import numpy
import numpy as np
import psutil
import pyopencl
import pyopencl as cl
from napari import Viewer
from pyopencl.array import to_device, Array

from aydin.features.fast.features_1d import collect_feature_1d
from aydin.features.fast.features_2d import collect_feature_2d
from aydin.features.fast.features_3d import collect_feature_3d
from aydin.features.fast.features_4d import collect_feature_4d
from aydin.features.fast.integral import (
    integral_1d,
    integral_2d,
    integral_3d,
    integral_4d,
)
from aydin.features.features_base import FeatureGeneratorBase
from aydin.offcore.offcore import offcore_array
from aydin.opencl.opencl_provider import OpenCLProvider
from aydin.util.nd import nd_range, nd_range_radii


class FastMultiscaleConvolutionalFeatures(FeatureGeneratorBase):
    """
    Multiscale convolutional feature generator.
    Uses OpenCL to acheive very fast integral image based feature generation.

    #TODO: There is some residual non-J-invariance on the borders...
    #TODO: There is still a weird issue in the 4D case... numerical overflow?

    """

    def __init__(
        self,
        opencl_provider=OpenCLProvider(),
        kernel_widths=[3] * 10,
        kernel_scales=[2 ** i - 1 for i in range(1, 11)],
        kernel_shapes=None,
        max_level=10,
        exclude_center=False,
        dtype=numpy.float32,
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
        self.max_level = max_level

        self.exclude_center = exclude_center
        self.dtype = dtype

        assert dtype == numpy.float32 or dtype == numpy.uint8 or dtype == numpy.uint16

    def is_enough_memory(self, array):

        # 20% buffer:
        buffer = 1.2

        available_cpu_mem = psutil.virtual_memory().free
        available_gpu_mem = self.opencl_provider.device.global_mem_size
        print(f"Available cpu mem: {available_cpu_mem / 1E6} MB")
        print(f"Available gpu mem: {available_gpu_mem / 1E6} MB")

        # This is what we need on the GPU to store the image and one feature:
        needed_gpu_mem = 2 * int(buffer * 4 * array.size * 1)
        print(
            f"Memory needed on the gpu: {needed_gpu_mem / 1E6} MB (image + 1 feature)"
        )

        nb_dim = len(array.shape)
        approximate_num_features = len(self.get_feature_descriptions(nb_dim))
        print(f"Approximate number of features: {approximate_num_features}")

        # This is what we need on the CPU:
        needed_cpu_mem = int(
            buffer
            * np.dtype(self.dtype).itemsize
            * array.size
            * approximate_num_features
        )
        print(f"Memory needed on the cpu: {needed_cpu_mem/ 1E6} MB")

        min_nb_batches_cpu = math.ceil(needed_cpu_mem / available_cpu_mem)
        min_nb_batches_gpu = math.ceil(needed_gpu_mem / available_gpu_mem)
        min_nb_batches = max(min_nb_batches_cpu, min_nb_batches_gpu)
        print(
            f"Minimum number of batches: {min_nb_batches} ( cpu:{min_nb_batches_cpu}, gpu:{min_nb_batches_gpu} )"
        )

        max_batch_size = (array.itemsize * array.size) / min_nb_batches
        is_enough_memory = (
            needed_cpu_mem < available_cpu_mem and needed_gpu_mem < available_gpu_mem
        )
        print(f"Maximum batch size: {max_batch_size} ")
        print(f"Is enough memory: {is_enough_memory} ")

        return is_enough_memory, max_batch_size

    def get_receptive_field_radius(self, nb_dim):

        radius = 0
        counter = 0
        for width, scale in zip(self.kernel_widths, self.kernel_scales):

            if counter > self.max_level:
                break

            radius = max(radius, width * scale // 2)
            counter += 1

        return radius

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

        if self.debug_log:
            print(
                f'Axis permutation for batch-dim consolidation during feature gen: {axes_permutation}'
            )
            print(
                f'Number of batch dim: {nb_batch_dim}, number of non batch dim: {nb_non_batch_dim}'
            )

        # # Set default aspect ratio based on image dimension:
        # if features_aspect_ratio is None:
        #     features_aspect_ratio = (1,) * image_dimension
        # assert len(features_aspect_ratio) == image_dimension
        # features_aspect_ratio = list(features_aspect_ratio)
        #
        # if self.debug_log:
        #     print(f'Feature aspect ratio: {features_aspect_ratio}')
        #
        # # Permutate aspect ratio:
        # features_aspect_ratio = tuple(
        #     features_aspect_ratio[axis] for axis in axes_permutation
        # )
        #
        # # and we only keep the non-batch dimensions aspect ratio values:
        # features_aspect_ratio = features_aspect_ratio[-nb_non_batch_dim:]
        #
        # if self.debug_log:
        #     print(
        #         f'Feature aspect ratio after permutation and selection: {features_aspect_ratio}'
        #     )

        # Initialise some variables:
        image_batch = None
        image_batch_gpu = None
        image_integral_gpu_1 = None
        image_integral_gpu_2 = None
        feature_gpu = None

        # We iterate over batches:
        for index in np.ndindex(*(image.shape[0:nb_batch_dim])):

            # Image batch slice:
            image_batch_slice = (*index, *(slice(None),) * nb_non_batch_dim)

            # Feature batch slice:
            feature_batch_slice = (
                slice(None),
                *index,
                *(slice(None),) * nb_non_batch_dim,
            )

            if self.debug_log:
                print(f'Index: {index}')
                print(f'Image batch slice: {image_batch_slice}')
                print(f'Feature batch slice: {feature_batch_slice}')

            # Little simplification:
            if image_batch_slice == (slice(None, None, None),) * nb_non_batch_dim:
                image_batch_slice = numpy.s_[...]

            # Copy because image[image_batch_slice] is not necessarily contiguous and pyOpenCL does not like discontiguous arrays:
            if image_batch is None:
                image_batch = np.array(
                    image[image_batch_slice], copy=True, dtype=numpy.float32
                )
            else:
                # Here we need to explicitly transfer values without creating an instance!
                numpy.copyto(image_batch, image[image_batch_slice], casting='unsafe')

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

                # we also allocate the needed integral images gpu storage:
                image_integral_gpu_1 = Array(
                    self.opencl_provider.queue, image_batch_gpu.shape, np.float32
                )
                image_integral_gpu_2 = Array(
                    self.opencl_provider.queue, image_batch_gpu.shape, np.float32
                )

            # Checking that the number of dimensions is within the bounds of what we can do:
            if nb_non_batch_dim <= 4:

                # We also compute the integral image for the present batch image:
                image_integral_gpu, mean = self.compute_integral_image(
                    image_batch_gpu, image_integral_gpu_1, image_integral_gpu_2
                )

                # Usefull code snippet for debugging features:
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
                    mean,
                )

            else:  # We only support 1D, 2D, 3D, and 4D.
                raise Exception(
                    f'dimension above {image_dimension} for non nbatch dimensions not yet implemented!'
                )

        del feature_gpu
        del image_batch_gpu

        # 'collect_fesatures_nD' puts the feature vector in axis 0.
        # The following line creates a view of the array
        # in which the features are indexed by the last dimension instead:
        features = np.moveaxis(features, 0, -1)

        # computes the inverse permutation from the permutation use to consolidate batch dimensions:
        axes_inverse_permutation = [
            axes_permutation.index(l) for l in range(len(axes_permutation))
        ] + [image_dimension]

        # permutates dimensions back:
        image = numpy.transpose(image, axes=axes_inverse_permutation[:-1])
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

        size_in_bytes = nb_features * image.size * np.dtype(self.dtype).itemsize
        free_mem_in_bytes = psutil.virtual_memory().free
        if self.debug_log:
            print(f'There is {int(free_mem_in_bytes/1E6)} MB of free memory')
            print(f'Feature array will be {(size_in_bytes/1E6)} MB.')

        # We take the heuristic that we need 20% extra memory available to be confortable:
        is_enough_memory = 1.2 * size_in_bytes < free_mem_in_bytes

        # That's the shape we need:
        shape = (nb_features,) + image.shape

        if not self.debug_force_memmap and is_enough_memory:
            if self.debug_log:
                print(
                    f'There is enough memory -- we do not need to use a mem mapped array.'
                )
            array = np.zeros(shape, dtype=self.dtype)

        else:
            if self.debug_log:
                print(f'There is not enough memory -- we will use a mem mapped array.')
            array = offcore_array(shape=shape, dtype=self.dtype)

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
        self,
        image_gpu,
        image_integral_gpu,
        ndim,
        feature_gpu=None,
        features=None,
        mean=0,
    ):
        """
        Collects nD features, one by one.
        If features is None, it just  counts the number of features so that the right size array
        can be allocated externally and then this method is called again this time with features != None

        :param image_gpu: gpu array of image to collect features from
        :param image_integral_gpu: gpu array to collect features from  (corresponding integral image)
        :param feature_gpu:  gpu array to use as temporary receptacle
        :param features: cpu features array to store all features to
        :param mean: image mean value needed for integral image usage.
        :return: number of features or the features themselves depending on the value of features (None or not None)
        """

        if self.debug_log:
            if features is None:
                print(f"Counting the number of features...")
            else:
                print(f"Computing features...")

        feature_description_list = self.get_feature_descriptions(ndim)

        if features is not None:

            # If feature is not None then we are

            # Temporary numpy array to hold the features after their computation on GPU:
            feature = None

            # We iterate through all features:
            feature_index = 0
            for effective_shift, effective_scale in feature_description_list:

                if self.debug_log:
                    print(
                        #    f"(width={width}, scale={scale}, shift={shift}, shape={shape}, reduction={reduction})"
                        f"(shift={effective_shift}, scale={effective_scale} "
                    )

                # We assemble the call:
                params = (
                    self.opencl_provider,
                    image_gpu,
                    image_integral_gpu,
                    feature_gpu,
                    *effective_shift,
                    *effective_scale,
                    self.exclude_center,
                    mean,
                )

                # We dispatch the call to the appropriate OpenCL feature collection code:
                if ndim == 1:
                    collect_feature_1d(*params)
                elif ndim == 2:
                    collect_feature_2d(*params)
                elif ndim == 3:
                    collect_feature_3d(*params)
                elif ndim == 4:
                    collect_feature_4d(*params)

                # Just-in-time allocation of the temp CPU feature array:
                if feature is None:
                    feature = numpy.zeros(features.shape[1:], dtype=numpy.float32)

                # We copy the GPU computed feature back into CPU memory:
                cl.enqueue_copy(self.opencl_provider.queue, feature, feature_gpu.data)

                # We multiply to prepare potential casting,
                # Let's also check that we don't have overflows:
                if self.dtype == numpy.uint8:
                    feature *= 255
                    # assert numpy.any(feature < 255)
                elif self.dtype == numpy.uint16:
                    feature *= 255 * 255
                    # assert numpy.any(feature < 255*255)

                # We put the computed feature into the main array that holds all features:
                # casting is done as needed:
                features[feature_index] = feature.astype(self.dtype, copy=False)

                # with napari.gui_qt():
                #     viewer = Viewer()
                #     viewer.add_image(image_gpu.get(), name='image')
                #     viewer.add_image(feature, name='feature')

                # optional sanity check:
                if self.check_nans and np.isnan(np.sum(features[feature_index])):
                    raise Exception(f'NaN values occur in features!')

                # Increment feature counter:
                feature_index += 1

            # We return the array holding all computed features:
            return features

        else:
            # In this case we are just here to count the _number_ of features, and return that count:
            return len(feature_description_list)

    def get_feature_descriptions(self, ndim):
        feature_description_list = []

        level = 0
        for width, scale, shape in zip(
            self.kernel_widths, self.kernel_scales, self.kernel_shapes
        ):
            # Check if we have passed the max number of features already:
            # Important: We might overshoot a bit, but that's ok to make sure we get all features at a given scale...
            if level > self.max_level:
                break
            level += 1

            # Computing the radius:
            radius = width // 2

            # We computw the radii along the different dimensions taking into account the aspect ratio:
            radii = list((max(1, radius),) * ndim)

            # We generate all feature shift vectors:
            features_shifts = list(nd_range_radii(radii))

            # print(f'Feature shifts: {features_shifts}')

            # For each feature shift we append to the feature description list:
            for shift in features_shifts:

                # There is no point in collecting features bigger than the image itself:
                # Note: this is not a good idea: number of features varies image size which causes trouble.
                # if (width * scale // 2) > max(image_gpu.shape):
                #    break

                # Excluding the center pixel/feature:
                if self.exclude_center and scale == 1 and shift == (0,) * ndim:
                    continue

                # Different 'shapes' of feature  distributions:
                if shape == 'l1' and sum([abs(i) for i in shift]) > radius:
                    continue
                elif shape == 'l2' and sum([i * i for i in shift]) > radius * radius:
                    continue
                elif shape == 'li':
                    pass

                # effective shift and scale after taking into account the aspect ratio:
                effective_shift = tuple(i * scale for i in shift)
                effective_scale = (max(1, scale),) * ndim

                #  We append the feature description:
                feature_description_list.append((effective_shift, effective_scale))
        # Some features might be identical due to the aspect ratio, we eliminate duplicates:
        no_duplicate_feature_description_list = remove_duplicates(
            feature_description_list
        )
        # We save the last computed feature description list for debug purposes:
        self.debug_feature_description_list = feature_description_list
        # We check and report how many duplicates were eliminated:
        number_of_duplicates = len(feature_description_list) - len(
            no_duplicate_feature_description_list
        )
        if self.debug_log:
            print(f"Number of duplicate features: {number_of_duplicates}")
        feature_description_list = no_duplicate_feature_description_list
        return feature_description_list


# Removes duplicates without chaning list's order:
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
