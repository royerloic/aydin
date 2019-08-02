import math
from abc import ABC, abstractmethod
from functools import reduce
from operator import mul

import numpy
import psutil

from aydin.analysis.correlation import correlation_distance
from aydin.normaliser.identity import IdentityNormaliser
from aydin.normaliser.minmax import MinMaxNormaliser
from aydin.normaliser.percentile import PercentileNormaliser
from aydin.offcore.offcore import offcore_array
from aydin.util.combinatorics import closest_product
from aydin.util.nd import nd_split_slices, remove_margin_slice


class ImageTranslatorBase(ABC):
    """
        Image Translator base class

    """

    def __init__(
        self, normaliser='percentile', analyse_correlation=False, monitor=None
    ):
        """

        """

        self.normaliser_type = normaliser
        self.debug = True
        self.models = []
        self.self_supervised = None
        self.analyse_correlation = analyse_correlation
        self.monitor = monitor

    @abstractmethod
    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        train_test_ratio,
        batch=False,
        monitoring_images=None,
        callback_period=3,
    ):
        pass

    @abstractmethod
    def _translate(self, input_image, batch_dims=None):
        """
        Translates an input image into an output image according to the learned function
        :param input_image: input image
        :param batch_dims: batch dimensions

        """
        pass

    @abstractmethod
    def get_receptive_field_radius(self):
        """
        Returns the receptive field radoius in voxels
        """
        pass

    @abstractmethod
    def _get_needed_mem(self, size):
        """
        Returns the memory needed (can be CPU, GPU or other) given an image of given size.
        :param size: size of image in voxels
        """
        pass

    @abstractmethod
    def _get_available_mem(self):
        """
        Returns the memory available (can be CPU, GPU or other).
        """
        pass

    def train(
        self,
        input_image,
        target_image,
        train_test_ratio=0.1,
        batch_dims=None,
        batch_size=None,
        batch_shuffle=False,
        monitoring_images=None,
        callback_period=3,
    ):
        """
            Train to translate a given input image to a given output image
            This has a lot of the machinery for batching and more...
        """

        if self.debug:
            print(f"Training on image of dimension {str(input_image.shape)} .")

        # If we use the same image for input and ouput then we are in a self-supervised setting:
        self.self_supervised = input_image is target_image

        if self.debug:
            print(f'Training is self-supervised.')

        # Analyse the input image correlation structure:
        if self.analyse_correlation:
            self.correlation = correlation_distance(input_image, target_image)

            if self.debug:
                print(f'Correlation structure of the image: {self.correlation}.')
        else:
            self.correlation = None

        # Instanciates normaliser(s):
        if self.normaliser_type == 'identity':
            self.input_normaliser = IdentityNormaliser()
            self.target_normaliser = (
                self.input_normaliser
                if self.self_supervised
                else PercentileNormaliser()
            )
        elif self.normaliser_type == 'percentile':
            self.input_normaliser = PercentileNormaliser()
            self.target_normaliser = (
                self.input_normaliser
                if self.self_supervised
                else PercentileNormaliser()
            )
        elif self.normaliser_type == 'minmax':
            self.input_normaliser = MinMaxNormaliser()
            self.target_normaliser = (
                self.input_normaliser if self.self_supervised else MinMaxNormaliser()
            )

        # Calibrates normaliser(s):

        self.input_normaliser.calibrate(input_image)
        if not self.self_supervised:
            self.target_normaliser.calibrate(target_image)

        # image shape:
        shape = input_image.shape

        # set default batch_dim value:
        if batch_dims is None:
            batch_dims = (False,) * len(input_image.shape)

        # Sanity check when not default batch dims:
        assert len(batch_dims) == len(input_image.shape)

        # We compute the ideal number of batches and whether there is enough memory:
        ideal_number_of_batches, is_enough_memory = self._get_number_of_batches(
            batch_size, input_image
        )

        if (batch_size is None) and is_enough_memory:

            # If there is enough memory, we don' need to batch:

            if self.debug:
                print(
                    f'There is enough memory -- we do full training (no batches, i.e. single batch).'
                )
            # array = np.zeros(shape, dtype=np.float32)

            # In that case the batch strategy is None:
            batch_strategy = None

            # 'Last minute' normalisation:
            normalised_input_image = self.input_normaliser.normalise(input_image)
            if not self.self_supervised:
                normalised_target_image = normalised_input_image
            else:
                normalised_target_image = self.target_normaliser.normalise(target_image)

            # We do one batch training:
            inferred_image = self._train(
                normalised_input_image,
                normalised_target_image,
                batch_dims,
                train_test_ratio,
                batch=False,
                monitoring_images=monitoring_images,
                callback_period=callback_period,
            )

            denormalised_inferred_image = self.input_normaliser.denormalise(
                inferred_image
            )

            return denormalised_inferred_image

        else:
            # There is not enough memory or a small batch size was requested:

            if self.debug:
                print(
                    f'There is not enough memory (CPU or GPU) -- we have to do batch training.'
                )

            # Ok, now we can start iterating through the batches:
            # strategy is a list in which each integer is the number of chunks per dimension.

            batch_strategy = self._get_batching_strategy(
                batch_dims, ideal_number_of_batches, shape
            )

            # How large the margins need to be to account for the receptive field:
            margins = self._get_margins(shape, batch_strategy)
            if self.debug:
                print(f'We will use these margins around batches: {margins}.')

            # We compute the slices objects to cut the input and target images into batches:
            batch_slices = nd_split_slices(
                shape, batch_strategy, do_shuffle=batch_shuffle, margins=margins
            )

            # We iterate through each batch and do training:
            for slice_tuple in batch_slices:

                if self.debug:
                    print(f"Current batch slice: {slice_tuple}")

                input_image_batch = input_image[slice_tuple]

                if self.self_supervised:
                    target_image_batch = input_image_batch
                else:
                    target_image_batch = target_image[slice_tuple]

                # 'Last minute' normalisation:
                normalised_input_image_batch = self.input_normaliser.normalise(
                    input_image_batch
                )
                if self.self_supervised:
                    normalised_target_image_batch = normalised_input_image_batch
                else:
                    normalised_target_image_batch = self.target_normaliser.normalise(
                        target_image_batch
                    )

                self._train(
                    normalised_input_image_batch,
                    normalised_target_image_batch,
                    batch_dims,
                    train_test_ratio,
                    batch=True,
                    monitoring_images=monitoring_images,
                    callback_period=callback_period,
                )

            # TODO: returned inferred image even in the batch case. Reason not to do it: might be a lot of data...

            return None

    def _get_number_of_batches(self, batch_size, input_image):

        # Compute the data and auxiliary data sizes:
        dataset_size_in_bytes = (input_image.size * input_image.itemsize) * (
            1 if self.self_supervised else 2
        )
        auxiliary_data_size_in_bytes = self._get_needed_mem(input_image.size)

        # Obtain free memory:
        free_cpu_mem_in_bytes = psutil.virtual_memory().free
        free_aux_mem_in_bytes = self._get_available_mem()
        if self.debug:
            print(f'Dataset is {(dataset_size_in_bytes / 1E6)} MB.')
            print(
                f'  and {(auxiliary_data_size_in_bytes / 1E6)} MB of auxiliary storage needed.'
            )
            print(f'There is {int(free_cpu_mem_in_bytes / 1E6)} MB of free CPU memory')
            print(
                f'There is {int(free_aux_mem_in_bytes / 1E6)} MB of free aux memory (typ. GPU)'
            )
        # We specify how much more free memory we need compared to the size of the data we need to allocate:
        # This is to give us some room. In the case of feature generation, it seems that GPU mem fragmentation prevents
        # us from allocating really big chunks...
        cpu_loading_factor = 1.2
        aux_loading_factor = 10.0
        # We compute batch sizes based on available, free and load factors:
        cpu_batch_size = free_cpu_mem_in_bytes // cpu_loading_factor
        aux_batch_size = free_aux_mem_in_bytes // aux_loading_factor
        max_batch_size = min(cpu_batch_size, aux_batch_size)
        if batch_size is None:
            # No forcing of batch size requested, so we proceed with the calculated value:
            effective_batch_size = max_batch_size
            if self.debug:
                print(f'Batch size is: {(effective_batch_size / 1E6)} MB.')
        else:
            # If the batch size has been specified then we use that
            if self.debug:
                print(f'Batch size has been forced: {(batch_size / 1E6)} MB.')
            effective_batch_size = min(batch_size, max_batch_size)
        # Do we have enough memory overall?
        is_enough_memory = (
            dataset_size_in_bytes <= effective_batch_size
            and auxiliary_data_size_in_bytes <= effective_batch_size
        )

        # This is the ideal number of batches so that we partition just enough:
        ideal_number_of_batches = max(
            2, int(math.ceil(dataset_size_in_bytes // effective_batch_size))
        )

        return ideal_number_of_batches, is_enough_memory

    def translate(
        self, input_image, translated_image=None, batch_dims=None, batch_size=None
    ):
        """
        Translates an input image into an output image according to the learned function.
        :param input_image:
        :type input_image:
        :param clip:
        :type clip:
        :return:
        :rtype:
        """
        if self.debug:
            print(
                "Predicting output image from input image of dimension %s ."
                % str(input_image.shape)
            )

        # set default batch_dim value:
        if batch_dims is None:
            batch_dims = (False,) * len(input_image.shape)

        # Sanity check when not default batch dims:
        assert len(batch_dims) == len(input_image.shape)

        # Input image shape:
        shape = input_image.shape

        # We prepare the translated image:
        if translated_image is None:
            translated_image = offcore_array(
                shape, dtype=self.target_normaliser.original_dtype
            )

        # We compute the parameters for batching (can be different than from training!):
        ideal_number_of_batches, is_enough_memory = self._get_number_of_batches(
            batch_size, input_image
        )
        batch_strategy = self._get_batching_strategy(
            batch_dims, ideal_number_of_batches, shape
        )

        if (batch_size is None) and is_enough_memory:
            # We did not batch during training so we can directly translate:

            # First we normalise the input:
            normalised_input_image = self.input_normaliser.normalise(input_image)

            # We translate:
            translated_image = self._translate(normalised_input_image, batch_dims)

            # Then we denormalise:
            denormalised_translated_image = self.target_normaliser.denormalise(
                translated_image
            )

            return denormalised_translated_image

        else:
            # We do need to do batch training because of a lack of memory or because a small batch size was requested:

            # First we compute the margins:
            margins = self._get_margins(shape, batch_strategy)
            if self.debug:
                print(f"Margins for batches: {margins} .")

            # batch slice objects (with and without margins):

            batch_slices_margins = nd_split_slices(
                shape, batch_strategy, margins=margins
            )
            batch_slices = nd_split_slices(shape, batch_strategy)

            for slice_margin_tuple, slice_tuple in zip(
                batch_slices_margins, batch_slices
            ):
                if self.debug:
                    print(f"Current batch slice: {slice_tuple}")

                # We first extract the batch image:
                input_image_batch = input_image[slice_margin_tuple]

                # Then we normalise this batch:
                normalised_input_image_batch = self.input_normaliser.normalise(
                    input_image_batch
                )

                # We do the actual translation:
                inferred_image_batch = self._translate(
                    normalised_input_image_batch, batch_dims
                )

                # We normalise that result:
                denormalised_inferred_image_batch = self.target_normaliser.denormalise(
                    inferred_image_batch
                )

                # We compute the slice needed to cut out the margins:
                remove_margin_slice_tuple = remove_margin_slice(
                    shape, slice_margin_tuple, slice_tuple
                )

                # We plug in the batch without margins into the destination image:
                translated_image[slice_tuple] = denormalised_inferred_image_batch[
                    remove_margin_slice_tuple
                ]

            return translated_image

    def _get_batching_strategy(self, batch_dims, ideal_number_of_batches, shape):

        # We will store the batch strategy as a list of integers representing the number of chunks per dimension:

        # This is the total number of batches that we would get if we were to use all batch dimensions:
        num_provided_batches = reduce(
            mul,
            [
                (dim_size if is_batch else 1)
                for dim_size, is_batch in zip(shape, batch_dims)
            ],
            1,
        )
        # We can use that to already determine whether the provided batch dimensions are enough:
        if num_provided_batches >= ideal_number_of_batches:
            # In this case we can make use of these dimensions, but there might be too many batches if we don't chunk...

            # Batch dimensions -- other dimensions are set to 1:
            batch_dimensions_sizes = [
                (dim_size if is_batch else 1)
                for dim_size, is_batch in zip(shape, batch_dims)
            ]

            # Considering the dimensions marked by the client of this method as 'batch dimensions',
            # if possible, what combination of such dimensions is best for creating batches?
            best_batch_dimensions = closest_product(
                batch_dimensions_sizes, ideal_number_of_batches, 1.0, 5.0
            )

            # At this point, the strategy is simply to use the provided batch dimensions:
            batch_strategy = batch_dimensions_sizes

            # But, it might not be possible to do so because of too large batch dimensions...
            if best_batch_dimensions is None:
                # We could ignore this case, but this might lead to very inefficient processing due to excessive batching.
                # Also, regressors perform worse (e.g. lGBM) when there are many batches.
                # Please note that the feature generation and regression will take into account the informastion
                # about w

                # In the following we take the heuristic to split the longest batch dimension.

                # First, we identify the largest batch dimension:
                index_of_largest_batch_dimension = batch_dimensions_sizes.index(
                    max(batch_dimensions_sizes)
                )

                # Then we determine the number of batches excluding that dimension:
                num_batches_without_dimension = (
                    num_provided_batches
                    / batch_dimensions_sizes[index_of_largest_batch_dimension]
                )

                # Then we can determine the optimal batching for that dimension:
                optimum = int(
                    math.ceil(ideal_number_of_batches / num_batches_without_dimension)
                )

                # the strategy is then:
                batch_strategy[index_of_largest_batch_dimension] = optimum

        else:
            # In this case we have too few batches provided, so we need to further split the dataset:

            # This is the amount of batching still required beyond what batching dimensions provide:
            extra_batching = int(
                math.ceil(ideal_number_of_batches / num_provided_batches)
            )

            # Now the question is how to distribute this among the other dimensions (non-batch).

            # First, what is the number of non-batch dimensions?
            num_non_batch_dimensions = sum(
                [(0 if is_batch else 1) for is_batch in batch_dims]
            )

            # Then, we distribute batching fairly based on the length of each dimension:

            # This is the product of non-batch dimensions:
            product_non_batch_dim = numpy.prod(
                [
                    dim_size
                    for dim_size, is_batch in zip(shape, batch_dims)
                    if not is_batch
                ]
            )

            # A little of logarythmic magic to find the 'geometric' (versus arithmetic) proportionality:
            alpha = math.log2(extra_batching) / math.log2(product_non_batch_dim)

            # The strategy is then:
            batch_strategy = [
                (dim_size if is_batch else int(math.ceil(dim_size ** alpha)))
                for dim_size, is_batch in zip(shape, batch_dims)
            ]
        if self.debug:
            print(f"Batching strategy is: {batch_strategy}")

        return batch_strategy

    def _get_margins(self, shape, batch_strategy):

        # We compute the margin from the receptive field but limit it to 33% of the tile size:
        margins = tuple(
            min(self.get_receptive_field_radius(), (dim // split) // 3)
            for (dim, split) in zip(shape, batch_strategy)
        )
        # We only need margins if we split a dimension:
        margins = tuple(
            (0 if split == 1 else margin)
            for margin, split in zip(margins, batch_strategy)
        )
        return margins
