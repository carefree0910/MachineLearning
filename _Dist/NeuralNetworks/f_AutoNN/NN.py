import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import math
import random
import pickle
import numpy as np

from _Dist.NeuralNetworks.NNUtil import *
from _Dist.NeuralNetworks.e_AdvancedNN.NN import Advanced


class Auto(Advanced):
    def __init__(self, x=None, y=None, x_cv=None, y_cv=None, name=None, loss=None, metric=None,
                 n_epoch=32, max_epoch=256, n_iter=128, batch_size=128, optimizer="Adam", lr=1e-3,
                 numerical_idx=None, feature_sets=None, is_regression=None,
                 pre_process_settings=None, nan_handler_settings=None, **kwargs):
        if name is None:
            raise ValueError("name should be provided in AutoNN")
        self._name = name

        # Pre-process settings
        if pre_process_settings is None:
            pre_process_settings = {}
        else:
            assert_msg = "Pre-process settings must be a dictionary"
            assert isinstance(pre_process_settings, dict), assert_msg
        self._pre_process_settings = pre_process_settings
        self._pre_processor = None
        self.pre_process_method = self.scale_method = self.refresh_mean_and_std = None

        # Nan handler settings
        if nan_handler_settings is None:
            nan_handler_settings = {}
        else:
            assert_msg = "Nan handler settings must be a dictionary"
            assert isinstance(nan_handler_settings, dict), assert_msg
        self._nan_handler_settings = nan_handler_settings
        self._nan_handler = None
        self.nan_handler_method = self.reuse_nan_handler_values = None

        self._data_loaded = False
        self._whether_redundant = None
        self._numerical_idx = numerical_idx
        self._is_regression = is_regression
        self._feature_sets = feature_sets
        self._is_numeric_label = None
        self._sparsity = kwargs.pop("sparsity", None)
        self._class_prior = kwargs.pop("class_prior", None)
        self._n_features = self._categorical_columns = self._all_num_idx = self._transform_dicts = None
        if feature_sets is not None and numerical_idx is not None:
            self._n_features = [len(feature_set) for feature_set in feature_sets]
            self._categorical_columns = [
                (i, value) for i, value in enumerate(self._n_features)
                if not numerical_idx[i] and numerical_idx[i] is not None
            ][:-1]

        file_type = kwargs.pop("file_type", "txt")
        shuffle = kwargs.pop("shuffle", True)
        restore = kwargs.pop("restore", True)
        test_rate = kwargs.pop("test_rate", 0.1)
        args = (numerical_idx, file_type, shuffle, restore, test_rate)
        if x is None or y is None:
            x, y, x_cv, y_cv = self._load_data(None, *args)
        else:
            data = np.hstack([x, y.reshape([-1, 1])])
            if x_cv is not None and y_cv is not None:
                data = (data, np.hstack([x_cv, y_cv.reshape([-1, 1])]))
            x, y, x_cv, y_cv = self._load_data(data, *args)

        self._kwargs = kwargs
        self._kwargs["numerical_idx"] = self._numerical_idx
        self._kwargs["categorical_columns"] = self._categorical_columns

        super(Auto, self).__init__(
            x, y, x_cv, y_cv, name, loss, metric,
            n_epoch, max_epoch, n_iter, batch_size, optimizer, lr, **kwargs
        )

    @property
    def label2num_dict(self):
        return None if not self._transform_dicts[-1] else self._transform_dicts[-1]

    @property
    def num2label_dict(self):
        label2num_dict = self.label2num_dict
        if label2num_dict is None:
            return
        return {i: c for c, i in label2num_dict.items()}

    @staticmethod
    def remove_redundant(whether_redundant, lst):
        return [elem for elem, redundant in zip(lst, whether_redundant) if not redundant]

    @staticmethod
    def get_np_arrays(*arrs):
        return [None if arr is None else np.asarray(arr, np.float32) for arr in arrs]

    @staticmethod
    def get_lists(*arrs):
        return [
            arr if isinstance(arr, list) else np.asarray(arr, np.float32).tolist()
            for arr in arrs
        ]

    def init_pre_process_settings(self):
        self.pre_process_method = self._pre_process_settings.get("pre_process_method", "normalize")
        self.scale_method = self._pre_process_settings.get("scale_method", "truncate")
        self.refresh_mean_and_std = self._pre_process_settings.get("refresh_mean_and_std", False)
        if self.pre_process_method is not None and self._pre_processor is None:
            self._pre_processor = PreProcessor(
                self.pre_process_method, self.scale_method, self.refresh_mean_and_std
            )

    def init_nan_handler_settings(self):
        self.nan_handler_method = self._nan_handler_settings.get("nan_handler_method", "median")
        self.reuse_nan_handler_values = self._nan_handler_settings.get("reuse_nan_handler_values", True)

    def _transform_data(self, data, include_label=False, refresh_redundant_info=False, refresh_info=False):
        print("Transforming data")
        label_dict = self._transform_dicts[-1]
        if refresh_redundant_info:
            self._whether_redundant = np.array([
                True if local_dict is None else False
                for i, local_dict in enumerate(self._transform_dicts)
            ])
        targets = [
            (i, local_dict) for i, (idx, local_dict) in enumerate(
                zip(self._numerical_idx, self._transform_dicts)
            ) if not idx and local_dict and not self._whether_redundant[i]
        ][:-1]
        # Transform
        for line in data:
            for i, local_dict in targets:
                elem = line[i]
                if isinstance(elem, str):
                    line[i] = local_dict[elem]
                elif math.isnan(elem):
                    line[i] = local_dict["nan"]
                else:
                    line[i] = local_dict[elem]
            if include_label and not self._is_numeric_label:
                line[-1] = label_dict[line[-1]]
        data = np.array(data, dtype=np.float32)
        # Handle nan
        if self._nan_handler is None:
            self._nan_handler = NanHandler(self._numerical_idx, self.nan_handler_method)
        data = self._nan_handler.handle(data)
        # Handle redundant
        n_redundant = np.sum(self._whether_redundant)
        if n_redundant > 0:
            whether_redundant = self._whether_redundant
            if not include_label:
                whether_redundant = whether_redundant[:-1]
            if refresh_redundant_info:
                warn_msg = "{} redundant: {}{}".format(
                    "These {} columns are".format(n_redundant) if n_redundant > 1 else "One column is",
                    [i for i, redundant in enumerate(whether_redundant) if redundant],
                    ", {} will be removed".format("it" if n_redundant == 1 else "they")
                )
                print(warn_msg)
                self._numerical_idx = self.remove_redundant(whether_redundant, self._numerical_idx)
                self._n_features = self.remove_redundant(whether_redundant, self._n_features)
            data = data[..., ~whether_redundant]
        # Get sparsity and class_prior
        if refresh_info:
            x, y = data[..., :-1], data[..., -1]
            self._sparsity = ((x == 0).sum() + np.isnan(x).sum()) / np.prod(x.shape)
            _, class_counts = np.unique(y, return_counts=True)
            self._class_prior = class_counts / class_counts.sum()
        # Pre-process data
        if self._pre_processor is not None:
            if not include_label:
                data = self._pre_processor.process(data, self._numerical_idx[:-1])
            else:
                data[..., :-1] = self._pre_processor.process(data[..., :-1], self._numerical_idx[:-1])
        return data

    def _prepare_xy(self, x, y, prefix, dtype=None):
        if x is None or y is None:
            if dtype is None:
                dtype = "test" if self._test_data is not None else "train"
            transform_y = False
        else:
            dtype = None
            transform_y = True
            x = self._transform_data(x)
        if dtype is not None:
            data = self._train_data if dtype == "train" else self._test_data
            x, y = data[..., :-1], data[..., -1]
        dic = self.num2label_dict
        if transform_y and dic is not None:
            if not self._is_numeric_label:
                y = [dic[yy] for yy in y]
            else:
                y = [dic[int(float(yy))] for yy in y]
        return np.asarray(x, np.float32), np.asarray(y, np.int)

    def _get_label_dict(self):
        labels = self._feature_sets[-1]
        if not all(Toolbox.is_number(str(label)) for label in labels):
            self._is_numeric_label = False
            return {key: i for i, key in enumerate(sorted(labels))}
        self._is_numeric_label = True
        label_set = set([int(float(label)) for label in labels])
        if -1 not in label_set:
            return {}
        assert len(label_set) == 2, "Including '-1' as label when n_classes > 2 is ambiguous"
        label_set.discard(-1)
        return {-1: 0, int(float(label_set.pop())): 1}

    def _get_transform_dicts(self):
        self._transform_dicts = [
            None if is_numerical is None else
            {key: i for i, key in enumerate(sorted(feature_set))}
            if not is_numerical and not all_num else {}
            for is_numerical, feature_set, all_num in zip(
                self._numerical_idx[:-1], self._feature_sets[:-1], self._all_num_idx[:-1]
            )
        ]
        if self.n_class == 1:
            self._transform_dicts.append({})
        else:
            self._transform_dicts.append(self._get_label_dict())

    def _load_data(self, data=None, numerical_idx=None, file_type="txt",
                  shuffle=True, restore=True, test_rate=0.1):
        data_folder = os.path.join("_Data", "_Cache", self._name)
        data_info_file = os.path.join("_DataInfo", "{}.info".format(self._name))
        train_data_file = os.path.join(data_folder, "train.npy")
        test_data_file = os.path.join(data_folder, "test.npy")

        use_cached_data = False
        if data is None:
            if os.path.isfile(train_data_file):
                print("Restoring data")
                use_cached_data = True
                train_data = np.load(train_data_file)
                test_data = np.load(test_data_file) if os.path.isfile(test_data_file) else None
                data = train_data if test_data is None else (train_data, test_data)

        if not use_cached_data:
            if file_type == "txt":
                sep, include_header = " ", False
            elif file_type == "csv":
                sep, include_header = ",", True
            else:
                raise NotImplementedError("File type '{}' not recognized".format(file_type))
            if data is None:
                target = os.path.join("_Data", self._name)
                if not os.path.isdir(target):
                    with open(target + ".{}".format(file_type), "r") as file:
                        data = Toolbox.get_data(file, sep, include_header)
                else:
                    with open(os.path.join(target, "train.{}".format(file_type)), "r") as file:
                        train_data = Toolbox.get_data(file, sep, include_header)
                    with open(os.path.join(target, "test.{}".format(file_type)), "r") as file:
                        test_data = Toolbox.get_data(file, sep, include_header)
                    data = (train_data, test_data)
            else:
                if not isinstance(data, tuple):
                    data = np.asarray(data, dtype=np.float32).tolist()
                else:
                    data = self.get_lists(*data)
            if isinstance(data, tuple):
                if shuffle:
                    random.shuffle(data[0])
                n_train = len(data[0])
                data = data[0] + data[1]
            else:
                if shuffle:
                    random.shuffle(data)
                n_train = int(len(data) * (1 - test_rate)) if test_rate > 0 else -1

        if not os.path.exists("_DataInfo"):
            os.makedirs("_DataInfo")
        if not os.path.isfile(data_info_file) or not restore:
            print("Generating data info")
            if numerical_idx is not None:
                self._numerical_idx = numerical_idx
            elif self._numerical_idx is not None:
                numerical_idx = self._numerical_idx
            if not self._feature_sets or not self._n_features or not self._all_num_idx:
                self._feature_sets, self._n_features, self._all_num_idx, self._numerical_idx = (
                    Toolbox.get_feature_info(data, numerical_idx)
                )
            self.n_class = self._n_features[-1]
            self._get_transform_dicts()
            with open(data_info_file, "wb") as file:
                pickle.dump([
                    self._n_features, self._numerical_idx, self._transform_dicts, self._is_numeric_label
                ], file)
        else:
            print("Restoring data info")
            with open(data_info_file, "rb") as file:
                data_info = pickle.load(file)
                self._n_features, self._numerical_idx, self._transform_dicts, self._is_numeric_label = data_info
            self.n_class = self._n_features[-1]

        if not use_cached_data:
            if n_train > 0:
                train_data, test_data = data[:n_train], data[n_train:]
            else:
                train_data, test_data = data, None
            train_data = self._transform_data(train_data, True, True, True)
            if test_data is not None:
                test_data = self._transform_data(test_data, True)

        self._categorical_columns = [
            (i, value) for i, value in enumerate(self._n_features)
            if not self._numerical_idx[i] and self._numerical_idx[i] is not None
        ]
        if not self._numerical_idx[-1]:
            self._categorical_columns.pop()

        if not use_cached_data:
            print("Caching data...")
            if not os.path.isdir(data_folder):
                os.makedirs(data_folder)
            np.save(train_data_file, train_data)
            if test_data is not None:
                np.save(test_data_file, test_data)
        self._data_loaded = True

        x, y = train_data[..., :-1], train_data[..., -1]
        if test_data is not None:
            x_test, y_test = test_data[..., :-1], test_data[..., -1]
        else:
            x_test = y_test = None

        return x, y, x_test, y_test

    @property
    def name(self):
        return "AutoNN" if self._name is None else self._name

    def _define_py_collections(self):
        self.py_collections = [
            "_pre_process_settings", "_nan_handler_settings", "_pre_processor", "_nan_handler",
            "_data_loaded", "_name", "_kwargs", "_transform_dicts", "hidden_units"
        ]

    def predict(self, x):
        x = self._transform_data(x)
        output = self._calculate(x, is_training=False)
        if self.n_class == 1:
            return output.ravel()
        return output


if __name__ == "__main__":
    auto = Auto(name="Zhongan").fit()
