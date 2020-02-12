import math
import os
from abc import ABC, abstractmethod
from os.path import join
import jsonpickle

from aydin.analysis.correlation import correlation_distance
from aydin.normaliser.identity import IdentityNormaliser
from aydin.normaliser.minmax import MinMaxNormaliser
from aydin.normaliser.normaliser_base import NormaliserBase
from aydin.normaliser.percentile import PercentileNormaliser
from aydin.util.log.log import lprint, lsection
from aydin.util.json import encode_indent
from aydin.util.nd import nd_split_slices, remove_margin_slice
from aydin.util.offcore.offcore import offcore_array


class ImageTranslatorBase(ABC):
    """Image Translator base class

    """

    def __init__(self, normaliser_type='percentile', monitor=None):
        """
        """

        self.normaliser_type = normaliser_type
        self.models = []
        self.self_supervised = None
        self.monitor = monitor

        self.callback_period = 3
        self.last_callback_time_sec = -math.inf

    def save(self, path: str):
        """Saves a 'all-batteries-included' image translation model at a given path (folder).

        :param path: path to save to
        """
        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        lprint(f"Saving image translator to: {path}")
        with open(join(path, "image_translation.json"), "w") as json_file:
            json_file.write(frozen)

        frozen += self.input_normaliser.save(path, 'input') + '\n'
        frozen += self.target_normaliser.save(path, 'target') + '\n'

        return frozen

    @staticmethod
    def load(path: str):
        """Returns a 'all-batteries-included' image translation model at a given path (folder).

        :param model_path: path to load from.
        """

        lprint(f"Loading image translator from: {path}")
        with open(join(path, "image_translation.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        thawed.input_normaliser = NormaliserBase.load(path, 'input')
        thawed.target_normaliser = NormaliserBase.load(path, 'target')

        thawed._load_internals(path)

        return thawed

    @abstractmethod
    def _load_internals(self, path: str):
        raise NotImplementedError()

    ## We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['input_normaliser']
        del state['target_normaliser']
        return state

    @abstractmethod
    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        train_valid_ratio=0.1,
        callback_period=3,
    ):
        raise NotImplemented()

    @abstractmethod
    def stop_training(self):
        raise NotImplemented()

    @abstractmethod
    def _translate(self, input_image, batch_dims=None):
        """
        Translates an input image into an output image according to the learned function
        :param input_image: input image
        :param batch_dims: batch dimensions

        """
        raise NotImplemented()

    @abstractmethod
    def get_receptive_field_radius(self, nb_dim):
        """
        Returns the receptive field radius in voxels
        """
        raise NotImplemented()

    def train(
        self,
        input_image,
        target_image,
        batch_dims=None,
        train_valid_ratio=0.1,
        callback_period=3,
    ):
        """
            Train to translate a given input image to a given output image
            This has a lot of the machinery for batching and more...
        """
        with lsection(f"Training on image of dimension {str(input_image.shape)} ."):

            # Verify that input andtarget images have same shape:
            assert input_image.shape == target_image.shape

            # If we use the same image for input and ouput then we are in a self-supervised setting:
            self.self_supervised = input_image is target_image

            lprint(f'Training is self-supervised.')

            # Analyse the input image correlation structure:
            self.correlation = correlation_distance(input_image, target_image)
            lprint(f'Correlation structure of the image: {self.correlation}.')

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
                    self.input_normaliser
                    if self.self_supervised
                    else MinMaxNormaliser()
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

            # 'Last minute' normalisation:
            normalised_input_image = self.input_normaliser.normalise(input_image)
            if not self.self_supervised:
                normalised_target_image = normalised_input_image
            else:
                normalised_target_image = self.target_normaliser.normalise(target_image)

            # We do one batch training:
            self._train(
                normalised_input_image,
                normalised_target_image,
                batch_dims=batch_dims,
                train_valid_ratio=train_valid_ratio,
                callback_period=callback_period,
            )

    def translate(
        self,
        input_image,
        translated_image=None,
        batch_dims=None,
        tile_size=None,
        max_margin=32,
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

        with lsection(
            f"Predicting output image from input image of dimension {input_image.shape}"
        ):

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

            if tile_size is None:
                # no tilling requested...

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
                # We do need to do tiled inference because of a lack of memory or because a small batch size was requested:

                # Receptive field:
                receptive_field_radius = self.get_receptive_field_radius(len(shape))

                # We get the tilling strategy but adjust for the max margins:
                tilling_strategy = self._get_tilling_strategy(
                    batch_dims, max(1, tile_size - 2 * max_margin), shape
                )
                lprint(f"Tilling strategy: {tilling_strategy}")

                # First we compute the margins:
                margins = self._get_margins(shape, tilling_strategy, max_margin)
                lprint(f"Margins for tiles: {margins} .")

                # tile slice objects (with and without margins):
                tile_slices_margins = list(
                    nd_split_slices(shape, tilling_strategy, margins=margins)
                )
                tile_slices = list(nd_split_slices(shape, tilling_strategy))

                # Number of tiles:
                number_of_tiles = len(tile_slices)
                lprint(f"Number of tiles (slices): {number_of_tiles}")

                # We create slice list:
                slicezip = zip(tile_slices_margins, tile_slices)

                counter = 1
                for slice_margin_tuple, slice_tuple in slicezip:
                    with lsection(
                        f"Current tile: {counter}/{number_of_tiles}, slice: {slice_tuple} "
                    ):

                        # We first extract the tile image:
                        input_image_tile = input_image[slice_margin_tuple].copy()

                        # Then we normalise this tile:
                        lprint(f"Normalising...")
                        normalised_input_image_tile = self.input_normaliser.normalise(
                            input_image_tile
                        )

                        # We do the actual translation:
                        lprint(f"Translating...")
                        inferred_image_tile = self._translate(
                            normalised_input_image_tile, batch_dims
                        )

                        # We denormalise that result:
                        lprint(f"Denormalising...")
                        denormalised_inferred_image_tile = self.target_normaliser.denormalise(
                            inferred_image_tile
                        )

                        # We compute the slice needed to cut out the margins:
                        lprint(f"Removing margins...")
                        remove_margin_slice_tuple = remove_margin_slice(
                            shape, slice_margin_tuple, slice_tuple
                        )

                        # We plug in the batch without margins into the destination image:
                        lprint(f"Inserting translated batch into result image...")
                        translated_image[
                            slice_tuple
                        ] = denormalised_inferred_image_tile[remove_margin_slice_tuple]

                        counter += 1

                return translated_image

    def _get_tilling_strategy(self, batch_dims, tile_size, shape):

        # We will store the batch strategy as a list of integers representing the number of chunks per dimension:
        with lsection(f"Determine tilling strategy:"):

            lprint(f"shape                   = {shape}")
            lprint(f"batch_dims              = {batch_dims}")
            lprint(f"tile_size               = {tile_size}")

            # This is the tile strategy, essentially how to split each dimension...
            tilling_strategy = tuple(
                (1 if is_batch else max(1, int(math.ceil(dim / tile_size))))
                for dim, is_batch in zip(shape, batch_dims)
            )
            lprint(f"Tilling strategy is: {tilling_strategy}")

            return tilling_strategy

    def _get_margins(self, shape, tilling_strategy, max_margin):

        # We compute the margin from the receptive field:
        margin = min(max_margin, self.get_receptive_field_radius(len(shape)))

        # n-d margin:
        margins = (margin,) * len(shape)

        # We only need margins if we split a dimension:
        margins = tuple(
            (0 if split == 1 else margin)
            for margin, split in zip(margins, tilling_strategy)
        )
        return margins
