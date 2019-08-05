import tempfile
import time

import numpy

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.features.features_base import FeatureGeneratorBase
from aydin.it.it_base import ImageTranslatorBase
from aydin.offcore.offcore import offcore_array
from aydin.regression.gbm import GBMRegressor


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
        normaliser='percentile',
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
            normaliser, analyse_correlation=analyse_correlation, monitor=monitor
        )

        self.feature_generator = feature_generator
        self.regressor = regressor

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

        # TODO: clean here
        # print("first train: ")
        # monitoring_variables = monitoring_variables[0], monitoring_variables[1], 3
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

    def _compute_features(self, image, exclude_center, batch_dims):
        """

        :param image:
        :type image:
        :param exclude_center:
        :type exclude_center:
        :return:
        :rtype:
        """
        if self.debug:
            print(f"Computing features ")

        self.feature_generator.exclude_center = exclude_center

        if self.correlation:
            max_length = max(self.correlation)
            features_aspect_ratio = tuple(
                length / max_length for length in self.correlation
            )

            if self.debug:
                print(f"Features aspect ratio: {features_aspect_ratio} ")
        else:
            features_aspect_ratio = None

        # image, batch_dims=None, features_aspect_ratio=None, features=None
        features = self.feature_generator.compute(image, batch_dims=batch_dims)
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

        # Predict using regressor:
        yp = self.regressor.predict(x)

        # We make sure that we have the result in float type, but make _sure_ to avoid copying data:
        if yp.dtype != numpy.float32 and yp.dtype != numpy.float64:
            yp = yp.astype(numpy.float32, copy=False)

        # We reshape the array:
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

        # Compute features on main training data:
        x = self._compute_features(input_image, self.self_supervised, batch_dims)
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
                    monitoring_image, self.self_supervised, batch_dims
                )
                for monitoring_image in normalised_monitoring_images
            ]
        else:
            monitoring_images_features = None

        self.monitoring_datasets = monitoring_images_features

        # TODO: clean regressor/it varialables mixed use
        def regressor_callback(env):

            iteration = env.iteration
            eval_metric_value = env.evaluation_result_list[0][2]
            current_time_sec = time.time()

            if (
                current_time_sec
                > self.regressor.last_callback_time_sec + self.regressor.callback_period
            ):
                model = env.model
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
                        eval_metric_value,
                        denormalised_inferred_images,
                    )
                elif self.monitor:
                    self.monitor.variables = (iteration, eval_metric_value, None)

                self.regressor.last_callback_time_sec = current_time_sec
            else:
                pass
                # print(f"Iteration={iteration} metric value: {eval_metric_value} ")

        nb_features = x.shape[-1]
        nb_entries = y.shape[0]
        if self.debug:
            print("Number of entries: %d features: %d ." % (nb_entries, nb_features))
            print("Splitting train and test sets.")

        # creates random complementary indices for selecting train and test entries:
        test_size = int(train_test_ratio * nb_entries)
        train_indices = numpy.full(nb_entries, False)
        train_indices[test_size:] = True
        numpy.random.shuffle(train_indices)
        test_indices = numpy.logical_not(train_indices)

        # Allocate arrays:
        x_train = numpy.zeros(((nb_entries - test_size), nb_features), dtype=x.dtype)
        y_train = numpy.zeros((nb_entries - test_size,), dtype=y.dtype)
        x_test = numpy.zeros((test_size, nb_features), dtype=x.dtype)
        y_test = numpy.zeros((test_size,), dtype=y.dtype)

        # train data
        numpy.copyto(x_train, x[train_indices])
        numpy.copyto(y_train, y[train_indices])

        # test data:
        numpy.copyto(x_test, x[test_indices])
        numpy.copyto(y_test, y[test_indices])

        if self.debug:
            print("Training...")
        if is_batch:
            validation_loss = self.regressor.fit(
                x_train,
                y_train,
                x_valid=x_test,
                y_valid=y_test,
                is_batch=True,
                regressor_callback=regressor_callback,
            )
            return validation_loss
        else:
            self.regressor.fit(
                x_train,
                y_train,
                x_valid=x_test,
                y_valid=y_test,
                is_batch=False,
                regressor_callback=regressor_callback,
            )
            inferred_image = self._predict_from_features(x, input_image.shape)
            return inferred_image

    def _translate(self, input_image, batch_dims=None):

        features = self._compute_features(input_image, self.self_supervised, batch_dims)
        inferred_image = self._predict_from_features(
            features, image_shape=input_image.shape
        )
        return inferred_image
