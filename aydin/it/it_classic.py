import math
import time

import numpy

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.features.features_base import FeatureGeneratorBase
from aydin.it.balancing.trainingdatabalancer import TrainingDataBalancer
from aydin.it.it_base import ImageTranslatorBase
from aydin.regression.gbm import GBMRegressor
from aydin.regression.regressor_base import RegressorBase
from aydin.util.log.logging import lprint, lsection


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
        balance_training_data=False,
        analyse_correlation=False,
        monitor=None,
    ):
        """
        Constructs a 'classic' image translator. Classic image translators use feature generation
        and standard machine learning regressors to acheive image translation.

        :param feature_generator: Feature generator
        :param regressor: regressor
        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param analyse_correlation: analyse correlation?
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(
            normaliser_type, analyse_correlation=analyse_correlation, monitor=monitor
        )

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

    ## We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['feature_generator']
        del state['regressor']
        return state

    def get_receptive_field_radius(self, nb_dim):
        return self.feature_generator.get_receptive_field_radius(nb_dim)

    def is_enough_memory(self, array):
        return self.feature_generator.is_enough_memory(array)

    def limit_epochs(self, max_epochs: int) -> int:
        # If the regressor is not progressive (supports training through multiple epochs) then we limit epochs to 1
        return max_epochs if self.regressor.progressive else 1

    def train(
        self,
        input_image,
        target_image,
        train_test_ratio=0.1,
        batch_dims=None,
        batch_size=None,
        batch_shuffle=False,
        callback_period=3,
        patience=3,
        patience_epsilon=0.000001,
        max_epochs=1024,
    ):

        # Resetting regressor:
        self.regressor.reset()

        if self.regressor.progressive:
            self.regressor.max_epochs = max_epochs
            self.regressor.patience = patience

        return super().train(
            input_image,
            target_image,
            train_test_ratio,
            batch_dims,
            batch_size,
            batch_shuffle,
            callback_period,
            patience,
            patience_epsilon,
        )

    def stop_training(self):
        return self.regressor.stop_fit()

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

            if self.correlation:
                max_length = max(self.correlation)
                features_aspect_ratio = tuple(
                    length / max_length for length in self.correlation
                )

                lprint(f"Features aspect ratio: {features_aspect_ratio} ")
            else:
                features_aspect_ratio = None

            # image, batch_dims=None, features_aspect_ratio=None, features=None
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
        is_batch=False,
        callback_period=3,
    ):
        """
            Train to translate a given input image to a given output image
            :param input_image: input image
            :param target_image: target image
            :param batch_dims: batch dimensions
            :param train_valid_ratio: ratio between train and validation data
            :param is_batch: is training batched ? (i.e. should we be able to call train again and continue training?)
            :param monitoring_images: images to use for mon itoring progress
            :param callback_period: callback max period
            :return:

        """
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
            if self.monitor.monitoring_images is not None:
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
                # Size of split batch, we assume we use 1024 chunks:
                lprint(f"Creating random indices for train/val split")
                nb_split_batches = 1024

                nb_split_batches_valid = int(train_valid_ratio * nb_split_batches)
                nb_split_batches_train = nb_split_batches - nb_split_batches_valid
                train_indices = numpy.full(nb_split_batches, False)
                train_indices[nb_split_batches_valid:] = True
                numpy.random.shuffle(train_indices)

                lprint(f"Calculating number of entries for train and validation...")
                nb_entries_per_split_batch = max(1, nb_entries // nb_split_batches)
                nb_entries_train = nb_split_batches_train * nb_entries_per_split_batch
                nb_entries_valid = nb_split_batches_valid * nb_entries_per_split_batch
                lprint(
                    f"Number of entries for training: {nb_entries_train}, validation:{nb_entries_valid}"
                )

                lprint(f"Allocating arrays...")
                x_train = numpy.zeros((nb_entries_train, nb_features), dtype=x.dtype)
                y_train = numpy.zeros((nb_entries_train,), dtype=y.dtype)
                x_valid = numpy.zeros((nb_entries_valid, nb_features), dtype=x.dtype)
                y_valid = numpy.zeros((nb_entries_valid,), dtype=y.dtype)

                with lsection(f"Copying data for training and validation sets..."):

                    balancer = TrainingDataBalancer(
                        total_entries=nb_split_batches,
                        number_of_bins=nb_split_batches // 8,
                        is_active=self.balance_training_data,
                    )

                    # We use a random permutation to avoid having the balancer drop only from the 'end' of the image
                    permutation = numpy.random.permutation(nb_split_batches)

                    i, jt, jv = 0, 0, 0
                    for is_train in numpy.nditer(train_indices):
                        if i % 64 == 0:
                            lprint(f"Copying section [{i},{i+64}]")

                        permutated_i = permutation[i]
                        src_start = permutated_i * nb_entries_per_split_batch
                        src_stop = src_start + nb_entries_per_split_batch
                        i += 1

                        xsrc = x[src_start:src_stop]
                        ysrc = y[src_start:src_stop]

                        if is_train:

                            if balancer.add_entry(ysrc):
                                dst_start = jt * nb_entries_per_split_batch
                                dst_stop = (jt + 1) * nb_entries_per_split_batch
                                jt += 1

                                xdst = x_train[dst_start:dst_stop]
                                ydst = y_train[dst_start:dst_stop]

                                numpy.copyto(xdst, xsrc)
                                numpy.copyto(ydst, ysrc)

                        else:

                            dst_start = jv * nb_entries_per_split_batch
                            dst_stop = (jv + 1) * nb_entries_per_split_batch
                            jv += 1

                            xdst = x_valid[dst_start:dst_stop]
                            ydst = y_valid[dst_start:dst_stop]

                            numpy.copyto(xdst, xsrc)
                            numpy.copyto(ydst, ysrc)

                    lprint(f"Histogram: {balancer.get_histogram_as_string()}")
                    lprint(
                        f"Percentage of data kept: {100*balancer.percentage_kept():.2f}%"
                    )

            lprint("Training now...")
            if is_batch:
                validation_loss = self.regressor.fit(
                    x_train,
                    y_train,
                    x_valid=x_valid,
                    y_valid=y_valid,
                    is_batch=True,
                    regressor_callback=regressor_callback if self.monitor else None,
                )
                return validation_loss
            else:
                self.regressor.fit(
                    x_train,
                    y_train,
                    x_valid=x_valid,
                    y_valid=y_valid,
                    is_batch=False,
                    regressor_callback=regressor_callback if self.monitor else None,
                )
                inferred_image = self._predict_from_features(x, input_image.shape)
                return inferred_image

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
