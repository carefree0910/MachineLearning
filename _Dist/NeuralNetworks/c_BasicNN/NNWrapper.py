import numpy as np

import sys
sys.path.append("../../../")
from _Dist.NeuralNetworks.c_BasicNN.NNCore import NNCore


class NNWrapper:
    def __init__(self, name, numerical_idx, features_lists, core=NNCore, **kwargs):
        self._core = core
        self._model = None
        self._name, self._kwargs = name, kwargs
        self._numerical_idx = numerical_idx
        self._is_regression = numerical_idx[-1]
        self._features_lists = features_lists
        self._n_classes = len(features_lists[-1])
        self._train_data = self._test_data = None

        self._n_features = [len(feat) for feat in features_lists]
        self._categorical_columns = [
            (i, value) for i, value in enumerate(self._n_features)
            if not numerical_idx[i] and numerical_idx[i] is not None
        ][:-1]

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    @property
    def core_params(self):
        return {
            "numerical_idx": self._numerical_idx,
            "categorical_columns": self._categorical_columns,
            "n_classes": self._n_classes
        }

    @staticmethod
    def get_np_arrays(*args):
        return [np.asarray(arr, np.float32) for arr in args]

    @staticmethod
    def get_lists(*args):
        return tuple(
            arr if isinstance(arr, list) else np.asarray(arr, np.float32).tolist()
            for arr in args
        )

    # Core

    def fit(self, x, y, x_test, y_test, **kwargs):
        x, y, x_test, y_test = NNWrapper.get_np_arrays(x, y, x_test, y_test)
        print_settings = kwargs.pop("print_settings", True)
        self._model = self._core(**self.core_params, **self._kwargs)
        train_losses, test_losses = self._model.fit(x, y, x_test, y_test, print_settings=print_settings)
        print("Test ", end="")
        self.evaluate(x_test, y_test)
        return train_losses, test_losses

    def predict(self, x, get_raw=False, verbose=False):
        return self._model.predict(x, get_raw, verbose)

    def predict_classes(self, x, verbose=False):
        return self._model.predict(x, True, verbose).argmax(axis=1)

    def predict_target_prob(self, x, target, verbose=False):
        return self._model.predict(x, False, verbose)[..., target]

    def evaluate(self, x, y, metric_name=None, verbose=False):
        self._model.evaluate(x, y, metric_name, verbose)
