import tempfile

import numpy

from pitl.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from pitl.it.it_base import ImageTranslatorBase
from pitl.regression.gbm import GBMRegressor


class ImageTranslatorClassic(ImageTranslatorBase):
    """
        Portable Image Translation Learning (PITL)

        Using classic ML (feature generation + regression)

    """

    def __init__(
        self,
        feature_generator=FastMultiscaleConvolutionalFeatures(),
        regressor=GBMRegressor(),
        normaliser='percentile',
    ):
        """

        :param feature_generator:
        :type feature_generator:
        :param regressor:
        :type regressor:
        """
        super().__init__(normaliser)

        self.feature_generator = feature_generator
        self.regressor = regressor

    def get_receptive_field_radius(self):
        return self.feature_generator.get_receptive_field_radius()

    def train(
        self,
        input_image,
        target_image,
        train_test_ratio=0.1,
        batch_dims=None,
        batch_size=None,
        batch_shuffle=False,
        monitoring_images=None,
        callbacks=None,
    ):

        # Resetting regressor:
        self.regressor.reset()
        return super().train(
            input_image,
            target_image,
            train_test_ratio,
            batch_dims,
            batch_size,
            batch_shuffle,
            monitoring_images,
            callbacks,
        )

    def _get_needed_mem(self, num_elements):
        return self.feature_generator.get_needed_mem(num_elements)

    def _get_available_mem(self):
        return self.feature_generator.get_available_mem()

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
            print("Computing features ")

        self.feature_generator.exclude_center = exclude_center
        features = self.feature_generator.compute(image, batch_dims)
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
        yp = self.regressor.predict(x)
        inferred_image = yp.reshape(image_shape)
        return inferred_image

    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        train_test_ratio=0.1,
        batch=False,
        monitoring_images=None,
        callbacks=None,
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

        # We pass these to the regressor:
        self.regressor.monitoring_datasets = monitoring_images_features

        # Setting monitoring dataset and callbacks:
        regressor_callbacks = []
        if callbacks:
            for callback in callbacks:

                def regressor_callback(
                    iteration, eval_metric_value, inferred_flat_monitoring_images
                ):

                    if inferred_flat_monitoring_images:
                        inferred_images = [
                            y_m.reshape(image.shape)
                            for (image, y_m) in zip(
                                monitoring_images, inferred_flat_monitoring_images
                            )
                        ]

                        denormalised_inferred_images = [
                            self.target_normaliser.denormalise(inferred_image)
                            for inferred_image in inferred_images
                        ]

                        callback(
                            iteration, eval_metric_value, denormalised_inferred_images
                        )
                    else:
                        callback(iteration, eval_metric_value, None)

                regressor_callbacks.append(regressor_callback)

        self.regressor.callbacks = regressor_callbacks

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

        # we allocate memory for the new arrays taking into account that we might need to use memory mapped files
        # in the splitting of train and test sets. The features are the heavy part, so that's what we map:
        # x_train, y_train, x_test, y_test = (None,) * 4

        if isinstance(x, numpy.memmap):
            temp_file = tempfile.TemporaryFile()
            x_train = numpy.memmap(
                temp_file,
                dtype=numpy.float32,
                mode='w+',
                shape=((nb_entries - test_size), nb_features),
            )
        else:
            x_train = numpy.zeros(
                ((nb_entries - test_size), nb_features), dtype=numpy.float
            )
        y_train = numpy.zeros((nb_entries - test_size,), dtype=numpy.float)
        x_test = numpy.zeros((test_size, nb_features), dtype=numpy.float)
        y_test = numpy.zeros((test_size,), dtype=numpy.float)

        # train data
        numpy.copyto(x_train, x[train_indices])
        numpy.copyto(y_train, y[train_indices])

        # test data:
        numpy.copyto(x_test, x[test_indices])
        numpy.copyto(y_test, y[test_indices])

        if self.debug:
            print("Training...")
        if batch:
            self.regressor.fit_batch(x_train, y_train, x_valid=x_test, y_valid=y_test)
            return None
        else:
            self.regressor.fit(x_train, y_train, x_valid=x_test, y_valid=y_test)
            inferred_image = self._predict_from_features(x, input_image.shape)
            return inferred_image

    def _translate(self, input_image, batch_dims=None):

        features = self._compute_features(input_image, self.self_supervised, batch_dims)
        inferred_image = self._predict_from_features(
            features, image_shape=input_image.shape
        )
        return inferred_image
