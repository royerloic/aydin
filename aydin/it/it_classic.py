import time
import numpy

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.features.base import FeatureGeneratorBase
from aydin.it.balancing.datahistogrambalancer import DataHistogramBalancer
from aydin.it.it_base import ImageTranslatorBase
from aydin.regression.gbm import GBMRegressor
from aydin.regression.regressor_base import RegressorBase
from aydin.util.log.log import lprint, lsection


class ImageTranslatorClassic(ImageTranslatorBase):
    """
        aydin Classic Image Translator

        Using classic ML (feature generation + regression)

    """

    feature_generator: FeatureGeneratorBase

    def __init__(
        self,
        feature_generator=FastMultiscaleConvolutionalFeatures(),
        regressor=GBMRegressor(),
        normaliser_type='percentile',
        balance_training_data=None,
        keep_ratio=1,
        max_voxels_for_training=4e6,
        monitor=None,
    ):
        """
        Constructs a 'classic' image translator. Classic image translators use feature generation
        and standard machine learning regressors to acheive image translation.

        :param feature_generator: Feature generator
        :param regressor: regressor
        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(normaliser_type, monitor=monitor)

        self.max_voxels_for_training = max_voxels_for_training
        self.keep_ratio = keep_ratio
        self.balance_training_data = balance_training_data
        self.feature_generator = feature_generator
        self.regressor = regressor

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).
        :param path: path to save to
        """
        with lsection(f"Saving 'classic' image translator to {path}"):
            frozen = super().save(path)
            frozen += self.feature_generator.save(path) + '\n'
            frozen += self.regressor.save(path) + '\n'

        return frozen

    def _load_internals(self, path: str):
        with lsection(f"Loading 'classic' image translator from {path}"):
            self.feature_generator = FeatureGeneratorBase.load(path)
            self.regressor = RegressorBase.load(path)

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['feature_generator']
        del state['regressor']
        return state

    def get_receptive_field_radius(self, nb_dim):
        return self.feature_generator.get_receptive_field_radius(nb_dim)

    def train(
        self,
        input_image,
        target_image,
        batch_dims=None,
        train_valid_ratio=0.1,
        callback_period=3,
        max_epochs=1024,
        patience=3,
        patience_epsilon=0.000001,
    ):

        # Resetting regressor:
        self.regressor.reset()

        super().train(
            input_image,
            target_image,
            batch_dims=batch_dims,
            train_valid_ratio=train_valid_ratio,
            callback_period=callback_period,
        )

    def stop_training(self):
        self.regressor.stop_fit()

    def _compute_features(
        self, image, batch_dims, exclude_center_feature, exclude_center_value
    ):
        """
            Internal function that computes features for a given image.
        :param image: image
        :param batch_dims: batch dimensions
        :param exclude_center_feature: exclude center feature
        :param exclude_center_value: exclude center value
        :return: returns flattened array of features
        """

        with lsection(f"Computing features for image of shape {image.shape}:"):

            lprint(f"exclude_center_feature={exclude_center_feature}")
            lprint(f"exclude_center_value={exclude_center_value}")
            lprint(f"batch_dims={batch_dims}")

            features = self.feature_generator.compute(
                image,
                batch_dims=batch_dims,
                exclude_center_feature=exclude_center_feature,
                exclude_center_value=exclude_center_value,
            )
            x = features.reshape(-1, features.shape[-1])

            return x

    def _predict_from_features(self, x, image_shape):
        """
            Internal function that predicts y from the features x
        :param x: flattened feature array
        :param image_shape: image shape
        :return: inferred image with given shape
        """

        with lsection(f"Predict from feature vector of dimension {x.shape}:"):

            lprint(f"Predicting... ")
            # Predict using regressor:
            yp = self.regressor.predict(x)

            # We make sure that we have the result in float type, but make _sure_ to avoid copying data:
            if yp.dtype != numpy.float32 and yp.dtype != numpy.float64:
                yp = yp.astype(numpy.float32, copy=False)

            # We reshape the array:
            lprint(f"Reshaping array to {image_shape}... ")
            inferred_image = yp.reshape(image_shape)
            return inferred_image

    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        train_valid_ratio=0.1,
        callback_period=3,
    ):

        with lsection(
            f"Training image translator from image of shape {input_image.shape} to image of shape {target_image.shape}:"
        ):

            # Compute features on main training data:
            x = self._compute_features(
                input_image, batch_dims, self.self_supervised, self.self_supervised
            )
            y = target_image.reshape(-1)
            # if self.debug:
            #   assert numpy.may_share_memory(target_image, y)

            # Compute features for monitoring images:
            if self.monitor is not None and self.monitor.monitoring_images is not None:
                # Normalise monitoring images:
                normalised_monitoring_images = [
                    self.input_normaliser.normalise(monitoring_image)
                    for monitoring_image in self.monitor.monitoring_images
                ]
                # compute features proper:
                monitoring_images_features = [
                    self._compute_features(
                        monitoring_image, batch_dims, self.self_supervised, False
                    )
                    for monitoring_image in normalised_monitoring_images
                ]
            else:
                monitoring_images_features = None

            # We keep these features handy...
            self.monitoring_datasets = monitoring_images_features

            # Regressor callback:
            def regressor_callback(iteration, val_loss, model):

                if val_loss is None:
                    return

                current_time_sec = time.time()

                # Correct for dtype range:
                if self.feature_generator.dtype == numpy.uint8:
                    val_loss /= 255
                elif self.feature_generator.dtype == numpy.uint16:
                    val_loss /= 255 * 255

                if (
                    current_time_sec
                    > self.last_callback_time_sec + self.callback_period
                ):

                    if self.monitoring_datasets and self.monitor:
                        predicted_monitoring_datasets = [
                            self.regressor.predict(x_m, model_to_use=model)
                            for x_m in self.monitoring_datasets
                        ]
                        inferred_images = [
                            y_m.reshape(image.shape)
                            for (image, y_m) in zip(
                                self.monitor.monitoring_images,
                                predicted_monitoring_datasets,
                            )
                        ]

                        denormalised_inferred_images = [
                            self.target_normaliser.denormalise(inferred_image)
                            for inferred_image in inferred_images
                        ]

                        self.monitor.variables = (
                            iteration,
                            val_loss,
                            denormalised_inferred_images,
                        )
                    elif self.monitor:
                        self.monitor.variables = (iteration, val_loss, None)

                    self.last_callback_time_sec = current_time_sec
                else:
                    pass
                    # print(f"Iteration={iteration} metric value: {eval_metric_value} ")

            nb_features = x.shape[-1]
            nb_entries = y.shape[0]
            lprint("Number of entries: %d features: %d" % (nb_entries, nb_features))

            with lsection(
                f"Splitting train and test sets (train_test_ratio={train_valid_ratio}) "
            ):

                if nb_entries <= 1e3:
                    nb_split_batches = min(nb_entries, 128)
                elif nb_entries <= 1e6:
                    nb_split_batches = 1024
                elif nb_entries <= 1e7:
                    nb_split_batches = 8 * 1024
                else:
                    nb_split_batches = 8 * 8 * 1024
                lprint(
                    f"Creating random indices for train/val split (nb_split_batches={nb_split_batches})"
                )

                nb_split_batches_valid = int(train_valid_ratio * nb_split_batches)
                nb_split_batches_train = nb_split_batches - nb_split_batches_valid
                is_train_array = numpy.full(nb_split_batches, False)
                is_train_array[nb_split_batches_valid:] = True
                lprint(
                    f"train/valid bool array created (length={is_train_array.shape[0]})"
                )

                lprint(f"Shuffling train/valid bool array...")
                numpy.random.shuffle(is_train_array)

                lprint(f"Calculating number of entries for train and validation...")
                nb_entries_per_split_batch = max(1, nb_entries // nb_split_batches)
                nb_entries_train = nb_split_batches_train * nb_entries_per_split_batch
                nb_entries_valid = nb_split_batches_valid * nb_entries_per_split_batch

                lprint(
                    f"Number of entries for training: {nb_entries_train} = {nb_split_batches_train}*{nb_entries_per_split_batch}, validation: {nb_entries_valid} = {nb_split_batches_valid} * {nb_entries_per_split_batch}"
                )

                lprint(f"Allocating arrays...")
                x_train = numpy.zeros((nb_entries_train, nb_features), dtype=x.dtype)
                y_train = numpy.zeros((nb_entries_train,), dtype=y.dtype)
                x_valid = numpy.zeros((nb_entries_valid, nb_features), dtype=x.dtype)
                y_valid = numpy.zeros((nb_entries_valid,), dtype=y.dtype)

                with lsection(f"Copying data for training and validation sets..."):

                    num_of_voxels = input_image.size
                    lprint(
                        f"Image has: {num_of_voxels} voxels, at most: {self.max_voxels_for_training} voxels will be used for training or validation."
                    )
                    max_voxels_keep_ratio = (
                        float(self.max_voxels_for_training) / num_of_voxels
                    )
                    lprint(
                        f"Given train ratio is: {self.keep_ratio}, max_voxels induced keep-ratio is: {max_voxels_keep_ratio}"
                    )
                    keep_ratio = min(self.keep_ratio, max_voxels_keep_ratio)
                    lprint(f"Effective keep-ratio is: {keep_ratio}")

                    if self.balance_training_data is None:
                        is_balancer_active = False if num_of_voxels < 5 * 1e6 else True
                    else:
                        is_balancer_active = self.balance_training_data

                    lprint(
                        f"Data histogram balancer is: {'active' if is_balancer_active else 'inactive'}"
                    )

                    balancer = DataHistogramBalancer(
                        total_entries=nb_split_batches,
                        number_of_bins=128,
                        is_active=is_balancer_active,
                        keep_ratio=keep_ratio,
                    )

                    # We use a random permutation to avoid having the balancer drop only from the 'end' of the image
                    permutation = numpy.random.permutation(nb_split_batches)

                    i, jt, jv = 0, 0, 0
                    dst_stop_train = 0
                    dst_stop_valid = 0

                    for is_train in numpy.nditer(is_train_array):
                        if i % 64 == 0:
                            lprint(f"Copying section [{i},{i+64}]")

                        permutated_i = permutation[i]
                        src_start = permutated_i * nb_entries_per_split_batch
                        src_stop = src_start + nb_entries_per_split_batch
                        i += 1

                        xsrc = x[src_start:src_stop]
                        ysrc = y[src_start:src_stop]

                        if balancer.add_entry(ysrc):
                            if is_train:
                                dst_start_train = jt * nb_entries_per_split_batch
                                dst_stop_train = (jt + 1) * nb_entries_per_split_batch

                                jt += 1

                                xdst = x_train[dst_start_train:dst_stop_train]
                                ydst = y_train[dst_start_train:dst_stop_train]

                                numpy.copyto(xdst, xsrc)
                                numpy.copyto(ydst, ysrc)

                            else:
                                dst_start_valid = jv * nb_entries_per_split_batch
                                dst_stop_valid = (jv + 1) * nb_entries_per_split_batch

                                jv += 1

                                xdst = x_valid[dst_start_valid:dst_stop_valid]
                                ydst = y_valid[dst_start_valid:dst_stop_valid]

                                numpy.copyto(xdst, xsrc)
                                numpy.copyto(ydst, ysrc)

                    # Now we actually truncate out all the zeros at the end of the arrays:
                    x_train = x_train[0:dst_stop_train]
                    y_train = y_train[0:dst_stop_train]
                    x_valid = x_valid[0:dst_stop_valid]
                    y_valid = y_valid[0:dst_stop_valid]

                    lprint(
                        f"Histogram all    : {balancer.get_histogram_all_as_string()}"
                    )
                    lprint(
                        f"Histogram kept   : {balancer.get_histogram_kept_as_string()}"
                    )
                    lprint(
                        f"Histogram dropped: {balancer.get_histogram_dropped_as_string()}"
                    )
                    lprint(
                        f"Number of entries kept: {balancer.total_kept()} out of {balancer.total_entries} total"
                    )
                    lprint(
                        f"Percentage of data kept: {100*balancer.percentage_kept():.2f}% (train_data_ratio={keep_ratio}) "
                    )
                    if keep_ratio >= 1 and balancer.percentage_kept() < 1:
                        lprint(
                            f"Note: balancer has dropped entries that fell on over-represented histogram bins"
                        )

            lprint("Training now...")
            self.regressor.fit(
                x_train,
                y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                regressor_callback=regressor_callback if self.monitor else None,
            )

    def _translate(self, input_image, batch_dims=None):
        """
            Internal method that translates an input image on the basis of the trainined model.
        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
        """
        features = self._compute_features(
            input_image, batch_dims, self.self_supervised, False
        )
        inferred_image = self._predict_from_features(
            features, image_shape=input_image.shape
        )
        return inferred_image
