import glob
import math
import multiprocessing
from os.path import join
from typing import List, Union

import gc
import lightgbm
import numpy
import psutil
from lightgbm import Booster

from aydin.regression.gbm_utils.callbacks import early_stopping
from aydin.regression.regressor_base import RegressorBase
from aydin.util.log.logging import lsection, lprint


class GBMRegressor(RegressorBase):
    """
    Regressor that uses the LightGBM library.

    """

    model: Booster

    def __init__(
        self,
        num_leaves=127,
        n_estimators=2048,
        max_bin=512,
        learning_rate=0.01,
        loss='l1',
        patience=5,
        verbosity=-1,
        compute_load=0.9,
    ):

        """
        Constructs a LightGBM regressor.
        :param num_leaves:
        :param n_estimators:
        :param max_bin:
        :param learning_rate:
        :param loss:
        :param patience:
        :param verbosity:
        :param compute_load:

        """

        super().__init__()

        self.force_verbose_eval = False

        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.max_bin = max_bin
        self.learning_rate = learning_rate
        self.metric = loss
        self.early_stopping_rounds = patience
        self.verbosity = verbosity
        self.compute_load = compute_load

        self.model = None

    def save(self, path: str):
        super().save(path)
        if not self.model is None:
            if isinstance(self.model, (list,)):
                counter = 0
                for model in self.model:
                    lgbm_model_file = join(path, 'lgbm_model_{counter}.txt')
                    model.save_model(lgbm_model_file)
                    counter += 1
            else:
                lgbm_model_file = join(path, 'lgbm_model.txt')
                self.model.save_model(lgbm_model_file)

        return 'lightGBM'

    def _load_internals(self, path: str):

        lgbm_files = glob.glob(join(path, 'lgbm_model_*.txt'))

        if len(lgbm_files) == 0:
            lgbm_model_file = join(path, 'lgbm_model.txt')
            self.model = Booster(model_file=lgbm_model_file)
        else:
            self.model = []
            for lgbm_file in lgbm_files:
                booster = Booster(model_file=lgbm_file)
                self.model.append(booster)

    ## We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state

    def reset(self):
        del self.model
        self.model = []

    def _get_params(self, num_samples, dtype=numpy.float32):
        min_data_in_leaf = 20 + int(0.01 * (num_samples / self.num_leaves))
        # print(f'min_data_in_leaf: {min_data_in_leaf}')

        objective = self.metric
        if objective == 'l1':
            objective = 'regression_l1'
        elif objective == 'l2':
            objective = 'regression_l2'

        if dtype == numpy.uint8:
            max_bin = 256
        else:
            max_bin = self.max_bin

        params = {
            "device": "cpu",
            "boosting_type": "gbdt",
            'objective': objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": max(3, int(math.log2(self.num_leaves)) - 1),
            "max_bin": max_bin,
            # "min_data_in_leaf": min_data_in_leaf,
            "subsample_for_bin": 200000,
            "num_threads": max(1, int(self.compute_load * multiprocessing.cpu_count())),
            "metric": self.metric,
            'verbosity': -1,  # self.verbosity,
            "bagging_freq": 1,
            "bagging_fraction": 0.8,
            # "device_type" : 'gpu'
        }

        if self.metric == 'l1':
            params["lambda_l1"] = 0.01
        elif self.metric == 'l2':
            params["lambda_l2"] = 0.01
        else:
            params["lambda_l1"] = 0.01

        return params

    def fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        super().fit(
            x_train, y_train, x_valid, y_valid, regressor_callback=regressor_callback
        )

        with lsection(f"GBM regressor fitting:"):

            nb_data_points = y_train.shape[0]
            self.num_features = x_train.shape[-1]
            has_valid_dataset = x_valid is not None and y_valid is not None

            lprint(f"Number of data points: {nb_data_points}")
            if has_valid_dataset:
                lprint(f"Number of validation data points: {y_valid.shape[0]}")
            lprint(f"Number of features per data point: {self.num_features}")

            train_dataset = lightgbm.Dataset(x_train, y_train, silent=True)
            valid_dataset = (
                lightgbm.Dataset(x_valid, y_valid, silent=True)
                if has_valid_dataset
                else None
            )

            self.__epoch_counter = 0

            # We translate the it classic callback into a lightGBM callback:
            # This avoids propagating annoying 'evaluation_result_list[0][2]'
            # throughout the codebase...
            def lgbm_callback(env):
                try:
                    val_loss = env.evaluation_result_list[0][2]
                except:
                    val_loss = 0
                    lprint("Problem with getting loss from LightGBM 'env' in callback")
                if regressor_callback:
                    regressor_callback(env.iteration, val_loss, env.model)
                else:
                    lprint(f"Epoch {self.__epoch_counter}: Validation loss: {val_loss}")
                    self.__epoch_counter += 1

            evals_result = {}

            verbose_eval = (lgbm_callback is None) or (self.force_verbose_eval)

            self.early_stopping_callback = early_stopping(
                self, self.early_stopping_rounds
            )

            with lsection("GBM regressor fitting now:"):
                model = lightgbm.train(
                    params=self._get_params(nb_data_points, dtype=x_train.dtype),
                    init_model=None,  # self.lgbmr if is_batch else None, <-- not working...
                    train_set=train_dataset,
                    valid_sets=valid_dataset,
                    early_stopping_rounds=None if has_valid_dataset else None,
                    num_boost_round=self.n_estimators,
                    # keep_training_booster= is_batch, <-- not working...
                    callbacks=[lgbm_callback, self.early_stopping_callback]
                    if has_valid_dataset
                    else [lgbm_callback],
                    verbose_eval=verbose_eval,
                    evals_result=evals_result,
                )
                lprint(f"GBM fitting done.")

            self.model = model

            del train_dataset
            del valid_dataset

            gc.collect()

            if has_valid_dataset:
                valid_loss = evals_result['valid_0'][self.metric][-1]
                return valid_loss
            else:
                return None

    def predict(self, x, batch_mode='median', model_to_use: Booster = None):

        with lsection(f"GBM regressor prediction:"):

            # We check that we get the right number of features.
            # If not, most likely the batch_dims are set wrong...
            assert self.num_features == x.shape[-1]

            lprint(f"Number of data points             : {x.shape[0]}")
            lprint(f"Number of features per data points: {x.shape[-1]}")

            if model_to_use is None:
                model_to_use = self.model
            else:
                lprint(f"Using a provided model! (Typical for callbacks)")

            lprint(f"GBM regressor predicting now...")
            return model_to_use.predict(x, num_iteration=model_to_use.best_iteration)
            lprint(f"GBM regressor predicting done!")
