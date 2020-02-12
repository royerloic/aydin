import math

import numpy
import numpy as np
import psutil
import pyopencl as cl
import scipy
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
from aydin.features.base import FeatureGeneratorBase
from aydin.providers.opencl.opencl_provider import OpenCLProvider
from aydin.util.log.log import lsection, lprint
from aydin.util.nd import nd_range_radii
from aydin.util.offcore.offcore import offcore_array

# TODO:  rename to FastMultiscaleConvolutionalFeatureGenerator
class FastMultiscaleConvolutionalFeatures(FeatureGeneratorBase):
    """
    Multiscale convolutional feature generator.
    Uses OpenCL and integral images to acheive very fast feature generation.

    However, for images that have more than 256^3 voxels problems start to appear
    because of our usage of integral images. For such large images it is better to
    use a tiled feature generator.

    """

    def __init__(
        self,
        #  0,  1,  2,  3,  4,  5,  6,   7,   8,    9,   10,   11
        #  3,  1,  3,  3,  3,  3,  3,   3,   3,    3,    3,    3
        #  1,  2,  3,  7, 15, 31, 63, 127, 255,  511,  1023, 2047
        # li, li, l2, l2, l3, l1, l1,  l1,  l1,   l1,   l1,   l1
        kernel_widths=[3] + [1] + [3] * 10,
        kernel_scales=[1] + [2] + [2 ** i - 1 for i in range(2, 12)],
        kernel_shapes=['li'] * 2 + ['l2'] * 3 + ['l1+nc'] * 5 + ['l1+oc'] * 2,
        max_level=12,
        exclude_scale_one=True,
        include_median_features=False,
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

        super().__init__()

        self.kernel_widths = kernel_widths
        self.kernel_scales = kernel_scales
        self.kernel_shapes = kernel_shapes
        self.max_level = max_level
        self.exclude_scale_one = exclude_scale_one
        self.include_median_features = include_median_features

        self.dtype = dtype

        assert (
            dtype == numpy.float32
            or dtype == numpy.uint8
            or dtype == numpy.uint16
            or dtype == numpy.float16
        )

    def _ensure_opencl_prodider_initialised(self):
        if not hasattr(self, 'opencl_provider') or self.opencl_provider is None:
            self.opencl_provider = OpenCLProvider()

    def _load_internals(self, path: str):

        # We can't use the loaded provider,
        # we need a new one because of the
        # native ressources associated:
        self.opencl_provider = None

    ## We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'opencl_provider' in state:
            del state['opencl_provider']
        return state

    def save(self, path: str):
        """
        Saves a 'all-batteries-inlcuded' image translation model at a given path (can be file or folder).
        :param path: path to save to
        """

        return super().save(path)

    @staticmethod
    def load(path: str):
        """
        Loads all parameters for the image translation
        :param path: path to load from.
        """

        return super().load(path)

    def get_receptive_field_radius(self, nb_dim):

        radius = 0
        counter = 0
        for width, scale in zip(self.kernel_widths, self.kernel_scales):

            if counter > self.max_level:
                break

            radius = max(radius, width * scale // 2)
            counter += 1

        return radius

    def max_non_batch_dims(self):
        return 4

    def max_voxels(self):
        return 512 ** 3

    def compute(
        self,
        image,
        batch_dims=None,
        exclude_center_feature=False,
        exclude_center_value=False,
        features=None,
        feature_last_dim=True,
    ):

        with lsection('Computing multi-scale convolutional features'):

            # Ensure we have an openCL provider initialised:
            self._ensure_opencl_prodider_initialised()

            # Checking NaNs just in case:
            if self.check_nans and np.isnan(np.sum(image)):
                raise Exception(f'NaN values occur in image!')

            image_dimension = len(image.shape)

            # set default batch_dim value:
            if batch_dims is None:
                batch_dims = (False,) * image_dimension

            # permutate dimensions in image to consolidate batch dimensions at the front:
            batch_dim_axes = [i for i in range(0, image_dimension) if batch_dims[i]]
            non_batch_dim_axes = [
                i for i in range(0, image_dimension) if not batch_dims[i]
            ]
            axes_permutation = batch_dim_axes + non_batch_dim_axes
            image = numpy.transpose(image, axes=axes_permutation)
            nb_batch_dim = sum([(1 if i else 0) for i in batch_dims])
            nb_non_batch_dim = image_dimension - nb_batch_dim

            lprint(
                f'Axis permutation for batch-dim consolidation during feature gen: {axes_permutation}'
            )
            lprint(
                f'Number of batch dim: {nb_batch_dim}, number of non batch dim: {nb_non_batch_dim}'
            )

            #### PLEASE DO NOT DELETE THIS CODE !!!!!!!!
            ###
            ##
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

                lprint(f'Index: {index}')
                lprint(f'Image batch slice: {image_batch_slice}')
                lprint(f'Feature batch slice: {feature_batch_slice}')

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
                    numpy.copyto(
                        image_batch, image[image_batch_slice], casting='unsafe'
                    )

                # We move the image to the GPU. Needs to fit entirely, could be a problem for very very large images.
                lprint(
                    f'Uploading image to GPU (size={image_batch.size*image_batch.itemsize / 1e6} MB)'
                )
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
                        image_batch,
                        image_batch_gpu,
                        image_integral_gpu,
                        nb_non_batch_dim,
                        exclude_center_feature,
                        exclude_center_value,
                    )
                    lprint(f'Number of features:  {nb_features}')

                    if features is None:
                        # At this point we know how big is the whole feature array, so we create it (encompasses all batches)
                        # This happens only once, the first time:
                        lprint(f'Creating feature array the first time...')
                        features = self.create_feature_array(image, nb_features)

                    # we compute one batch of features:
                    self.collect_features_nD(
                        image_batch,
                        image_batch_gpu,
                        image_integral_gpu,
                        nb_non_batch_dim,
                        exclude_center_feature,
                        exclude_center_value,
                        feature_gpu,
                        features[feature_batch_slice],
                        mean,
                    )

                    # #Usefull code snippet for debugging features:
                    # with napari.gui_qt():
                    #      viewer = Viewer()
                    #      viewer.add_image(features, name='features')

                else:  # We only support 1D, 2D, 3D, and 4D.
                    message = f'dimension above {image_dimension} for non nbatch dimensions not yet implemented!'
                    lprint(message)
                    raise Exception(message)

            del feature_gpu
            del image_batch_gpu

            # 'collect_fesatures_nD' puts the feature vector in axis 0.
            # The following line creates a view of the array
            # in which the features are indexed by the last dimension instead:
            if feature_last_dim:
                features = np.moveaxis(features, 0, -1)

            # computes the inverse permutation from the permutation use to consolidate batch dimensions:
            axes_inverse_permutation = [
                axes_permutation.index(l) for l in range(len(axes_permutation))
            ] + [image_dimension]

            # permutates dimensions back:
            # image = numpy.transpose(image, axes=axes_inverse_permutation[:-1])
            features = numpy.transpose(features, axes=axes_inverse_permutation)

            return features

    def compute_integral_image(
        self, image_gpu, image_integral_gpu_1, image_integral_gpu_2
    ):

        with lsection(f'Computing integral image...'):

            # Ensure we have an openCL provider initialised:
            self._ensure_opencl_prodider_initialised()

            dim = len(image_gpu.shape)

            if dim == 1:
                return integral_1d(
                    self.opencl_provider, image_gpu, image_integral_gpu_1
                )

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
            lprint("Done.")

    def collect_features_nD(
        self,
        image,
        image_gpu,
        image_integral_gpu,
        ndim,
        exclude_center_feature,
        exclude_center_value,
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

        with lsection(
            f'{ "Counting" if features is None else "Collecting"} features of dimension {len(image_gpu.shape)}'
        ):
            if not features is None:
                lprint(f"Features array provided has shape: {features.shape} ")

            # Ensure we have an openCL provider initialised:
            self._ensure_opencl_prodider_initialised()

            if features is None:
                lprint(f"Counting the number of features...")
            else:
                lprint(f"Computing features...")

            feature_description_list = self.get_feature_descriptions(
                ndim, exclude_center_feature
            )

            # Number of median features:
            nb_median_features = 2 ** ndim + 1

            if features is not None:

                # If feature is not None then we are actually computing features and not just counting:

                # Standard features:
                with lsection(f"Computing standard features:"):
                    # Temporary numpy array to hold the features after their computation on GPU:
                    feature = None

                    # We iterate through all features:
                    feature_index = 0
                    for (
                        effective_shift,
                        negative_extent,
                        positive_extent,
                        shape,
                    ) in feature_description_list:

                        lprint(
                            f"standard (shape={shape}, shift={effective_shift}, negext={negative_extent}, posext={positive_extent} "
                        )

                        # We assemble the call:
                        params = (
                            self.opencl_provider,
                            image_gpu,
                            image_integral_gpu,
                            feature_gpu,
                            *effective_shift,
                            *negative_extent,
                            *positive_extent,
                            exclude_center_value,
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
                            feature = numpy.zeros(
                                features.shape[1:], dtype=numpy.float32
                            )

                        # We copy the GPU computed feature back into CPU memory:
                        # lprint(f"Copying from GPU VRAM {feature_gpu.shape} to CPU RAM {feature.shape}")
                        cl.enqueue_copy(
                            self.opencl_provider.queue, feature, feature_gpu.data
                        )

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
                        if self.check_nans and np.isnan(
                            np.sum(features[feature_index])
                        ):
                            raise Exception(f'NaN values occur in features!')

                        # Increment feature counter:
                        feature_index += 1

                # Median features:
                if self.include_median_features:
                    with lsection(f"Computing scale 2 features:"):
                        pass
                        # with napari.gui_qt():
                        #     viewer = Viewer()
                        #     viewer.add_image(features, name='features')

                # We return the array holding all computed features:
                return features

            else:
                # In this case we are just here to count the _number_ of features, and return that count:
                return len(feature_description_list) + nb_median_features

    def get_feature_descriptions(self, ndim, exclude_center):
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

            # We compute the radii along the different dimensions taking into account the aspect ratio:
            radii = list((max(1, radius),) * ndim)

            # We generate all feature shift vectors:
            features_shifts = list(nd_range_radii(radii))

            if scale == 2 and width == 1:
                # This is the special case of scale-two-features:
                features_shifts = list(
                    [shift for shift in features_shifts if not 1 in shift]
                )

            # print(f'Feature shifts: {features_shifts}')

            # For each feature shift we append to the feature description list:
            for shift in features_shifts:

                # There is no point in collecting features bigger than the image itself:
                # Note: this is not a good idea: number of features varies image size which causes trouble.
                # if (width * scale // 2) > max(image_gpu.shape):
                #    break

                # Excluding the center pixel/feature:
                if exclude_center and scale == 1 and shift == (0,) * ndim:
                    continue

                # Exclude scale one features:
                if self.exclude_scale_one and scale == 1:
                    continue

                if scale == 2 and width == 1:
                    effective_shift = shift
                    negative_extent = (0,) * ndim
                    positive_extent = (2,) * ndim
                else:
                    # Different 'shapes' of feature  distributions:
                    if 'l1' in shape and sum([abs(i) for i in shift]) > radius:
                        continue
                    elif (
                        'l2' in shape and sum([i * i for i in shift]) > radius * radius
                    ):
                        continue
                    elif 'li' in shape:
                        pass

                    # keep only center (oc) or remove all that are not center (nc)
                    if 'oc' in shape and sum([abs(i) for i in shift]) > 0:
                        continue
                    elif 'nc' in shape and sum([abs(i) for i in shift]) == 0:
                        continue

                    # effective shift and scale after taking into account the aspect ratio:
                    effective_shift = tuple(i * scale for i in shift)
                    negative_extent = (max(1, scale // 2),) * ndim
                    positive_extent = (max(1, scale // 2),) * ndim

                #  We append the feature description:
                feature_description_list.append(
                    (effective_shift, negative_extent, positive_extent, shape)
                )
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

        lprint(f"Number of duplicate features: {number_of_duplicates}")
        feature_description_list = no_duplicate_feature_description_list
        return feature_description_list


# Removes duplicates without chaning list's order:
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
