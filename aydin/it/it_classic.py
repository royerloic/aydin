import time

import numpy

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.features.features_base import FeatureGeneratorBase
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
        analyse_correlation=False,
        monitor=None,
    ):
        """

        :param feature_generator:
        :type feature_generator:
        :param regressor:
        :type regressor:
        """
        super().__init__(
            normaliser_type, analyse_correlation=analyse_correlation, monitor=monitor
        )

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
            monitoring_images,
            callback_period,
            patience,
            patience_epsilon,
        )

    def is_enough_memory(self, array):
        return self.feature_generator.is_enough_memory(array)

    def limit_epochs(self, max_epochs: int) -> int:
        # If the regressor is not progressive (supports training through multiple epochs) then we limit epochs to 1
        return max_epochs if self.regressor.progressive else 1

    def _compute_features(
        self, image, batch_dims, exclude_center_feature, exclude_center_value
    ):
        """

        :param image:
        :type image:
        :param exclude_center:
        :type exclude_center:
        :return:
        :rtype:
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
            internal function that predicts y from the features x
        :param x:
        :type x:
        :param image_shape:
        :type image_shape:
        :param clip:
        :type clip:
        :return:
        :rtype:
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
        train_test_ratio=0.1,
        is_batch=False,
        monitoring_images=None,
        callback_period=3,
    ):
        """
            Train to translate a given input image to a given output image

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
            if monitoring_images:
                # Normalise monitoring images:
                normalised_monitoring_images = [
                    self.input_normaliser.normalise(monitoring_image)
                    for monitoring_image in monitoring_images
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
                                monitoring_images, predicted_monitoring_datasets
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
            lprint("Number of entries: %d features: %d ." % (nb_entries, nb_features))
            lprint("Splitting train and test sets.")

            lprint(f"Creating random indices for train/val split")
            val_size = int(train_test_ratio * nb_entries)
            train_indices = numpy.full(nb_entries, False)
            train_indices[val_size:] = True
            numpy.random.shuffle(train_indices)
            valid_indices = numpy.logical_not(train_indices)

            lprint(f"Allocating arrays...")
            x_train = numpy.zeros(((nb_entries - val_size), nb_features), dtype=x.dtype)
            y_train = numpy.zeros((nb_entries - val_size,), dtype=y.dtype)
            x_valid = numpy.zeros((val_size, nb_features), dtype=x.dtype)
            y_valid = numpy.zeros((val_size,), dtype=y.dtype)

            lprint(f"Copying training data...")
            numpy.copyto(x_train, x[train_indices])
            numpy.copyto(y_train, y[train_indices])

            lprint(f"Copying validation data...")
            numpy.copyto(x_valid, x[valid_indices])
            numpy.copyto(y_valid, y[valid_indices])

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

        features = self._compute_features(
            input_image, batch_dims, self.self_supervised, False
        )
        inferred_image = self._predict_from_features(
            features, image_shape=input_image.shape
        )
        return inferred_image
