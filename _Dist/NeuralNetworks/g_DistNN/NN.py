import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import tensorflow as tf

from _Dist.NeuralNetworks.f_AutoNN.NN import Auto
from _Dist.NeuralNetworks.NNUtil import Toolbox, PreProcessor


class Dist(Auto):
    def __init__(self, *args, **kwargs):
        super(Dist, self).__init__(*args, **kwargs)
        self._name_appendix = "Dist"

    def init_model_param_settings(self):
        if self.n_class == 2:
            self.model_param_settings["metric"] = self.model_param_settings.get("metric", "auc")
        super(Dist, self).init_model_param_settings()

    def reset_all_variables(self):
        self._sess.run(tf.global_variables_initializer())

    def rolling_fit(self, train_rate=0.8, cv_rate=0.1, sample_weights=None, **kwargs):
        n_data = len(self._train_generator)
        if sample_weights is not None:
            n_weights = len(sample_weights)
            assert_msg = (
                "Sample weights should match training data, "
                "but n_weights={} & n_data={} found".format(n_weights, n_data)
            )
            assert n_weights == n_data, assert_msg
        n_train = int(train_rate * n_data)
        n_test = int(cv_rate * n_data) if self._test_generator is None else len(self._test_generator)
        j, cursor, print_settings = 0, 0, kwargs.pop("print_settings", True)
        flag = test_flag = False
        if self._test_generator is not None:
            test_flag = True
            test_data, _ = self._test_generator.get_all_data()
            x_test, y_test = test_data[..., :-1], test_data[..., -1]
        else:
            x_test = y_test = None
        print("Rolling fit with train_rate={} and test_rate={}".format(train_rate, cv_rate))
        while True:
            j += 1
            train_cursor = cursor + n_train
            test_cursor = train_cursor + n_test
            if n_data - test_cursor < n_test:
                flag = True
                test_cursor = n_data
            with self._train_generator:
                if self._test_generator is None:
                    test_data, _ = self._train_generator.get_range(train_cursor, test_cursor)
                    x_test, y_test = test_data[..., :-1], test_data[..., -1]
                    self._test_generator = self._generator_base(x_test, y_test, name="TestGenerator")
                self._train_generator.set_range(cursor, train_cursor)
                kwargs["print_settings"] = print_settings
                self.fit(**kwargs)
                x, y, _ = self._gen_batch(self._train_generator, self.n_random_train_subset, True)
                print("  -  Performance of roll {}".format(j), end=" | ")
                self._evaluate(x, y, x_test, y_test)
                cursor += n_test
                print_settings = False
                if not test_flag:
                    self._test_generator = None
                if flag:
                    break
        with self._train_generator:
            self._train_generator.set_range(cursor)
            kwargs["print_settings"] = print_settings
            self.fit(**kwargs)
            if self._test_generator is not None:
                print("  -  Performance of roll {}".format(j + 1), end=" | ")
                self._evaluate(x_test=x_test, y_test=y_test)
        return self

    def increment_fit(self, x=None, y=None, x_test=None, y_test=None, sample_weights=None, **kwargs):
        if x is not None and y is not None:
            data = np.hstack([np.asarray(x, np.float32), np.asarray(y, np.float32).reshape([-1, 1])])
            if x_test is not None and y_test is not None:
                data = (data, np.hstack([
                    np.asarray(x_test, np.float32), np.asarray(y_test, np.float32).reshape([-1, 1])
                ]))
            x, y, x_test, y_test = self._load_data(data)
        else:
            data = None
            if self._test_generator is not None:
                test_data, _ = self._test_generator.get_all_data()
                x_test, y_test = test_data[..., :-1], test_data[..., -1]
        if sample_weights is not None:
            self._sample_weights = np.asarray(sample_weights, np.float32)
        self._handle_unbalance(y)
        self._handle_sparsity()
        if data is not None:
            self._train_generator = self._generator_base(x, y, self._sample_weights, name="Generator")
            if x_test is not None and y_test is not None:
                self._test_generator = self._generator_base(x_test, y_test, name="TestGenerator")
        self.fit(**kwargs)
        x, y, _ = self._gen_batch(self._train_generator, self.n_random_train_subset, True)
        print("  -  Performance of increment fit", end=" | ")
        self._evaluate(x, y, x_test, y_test)
        return self

    def _k_series_initialization(self, k, data):
        self.init_data_info()
        x, y, x_test, y_test = self._load_data(data, stage=1)
        x_test, y_test, *_ = self._load_data(
            np.hstack([x_test, y_test.reshape([-1, 1])]),
            names=("test", None), test_rate=0, stage=2
        )
        names = [("train{}".format(i), "cv{}".format(i)) for i in range(k)]
        return x, y, x_test, y_test, names

    def _k_series_evaluation(self, i, x_test, y_test):
        train, sw_train = self._train_generator.get_all_data()
        cv, sw_cv = self._test_generator.get_all_data()
        x, y = train[..., :-1], train[..., -1]
        x_cv, y_cv = cv[..., :-1], cv[..., -1]
        print("  -  Performance of run {}".format(i + 1), end=" | ")
        self._evaluate(x, y, x_cv, y_cv, x_test, y_test)

    def _merge_preprocessors_from_k_series(self, names):
        train_names, cv_names = [name[0] for name in names], [name[1] for name in names]
        self._merge_preprocessors_by_names("train", train_names)
        self._merge_preprocessors_by_names("cv", cv_names)

    def _merge_preprocessors_by_names(self, target, names):
        if len(names) == 1:
            self._pre_processors[target] = self._pre_processors.pop(names[0])
        pre_processors = [self._pre_processors.pop(name) for name in names]
        methods = [pre_processor.method for pre_processor in pre_processors]
        scale_methods = [pre_processor.scale_method for pre_processor in pre_processors]
        assert Toolbox.all_same(methods), "Pre_process method should be all_same"
        assert Toolbox.all_same(scale_methods), "Scale method should be all_same"
        new_processor = PreProcessor(methods[0], scale_methods[0])
        new_processor.mean = np.mean([pre_processor.mean for pre_processor in pre_processors], axis=0)
        new_processor.std = np.mean([pre_processor.std for pre_processor in pre_processors], axis=0)
        self._pre_processors[target] = new_processor

    def k_fold(self, k=10, data=None, test_rate=0., sample_weights=None, **kwargs):
        x, y, x_test, y_test, names = self._k_series_initialization(k, data)
        n_batch = int(len(x) / k)
        all_idx = list(range(len(x)))
        print_settings = True
        if sample_weights is not None:
            self._sample_weights = np.asarray(sample_weights, np.float32)
        sample_weights_store = self._sample_weights
        print("Training k-fold with k={} and test_rate={}".format(k, test_rate))
        for i in range(k):
            self.reset_all_variables()
            cv_idx = list(range(i * n_batch, (i + 1) * n_batch))
            train_idx = [j for j in all_idx if j < i * n_batch or j >= (i + 1) * n_batch]
            x_cv, y_cv = x[cv_idx], y[cv_idx]
            x_train, y_train = x[train_idx], y[train_idx]
            if sample_weights is not None:
                self._sample_weights = sample_weights_store[train_idx]
            else:
                self._sample_weights = None
            kwargs["print_settings"] = print_settings
            kwargs["names"] = names[i]
            self.data_info["stage"] = 2
            self.fit(x_train, y_train, x_cv, y_cv, **kwargs)
            self._k_series_evaluation(i, x_test, y_test)
            print_settings = False
        self.data_info["stage"] = 3
        self._merge_preprocessors_from_k_series(names)
        self._sample_weights = sample_weights_store
        if x_test is not None and y_test is not None:
            self._test_generator = self._generator_base(x_test, y_test, name="TestGenerator")
        return self

    def k_random(self, k=3, data=None, cv_rate=0.1, test_rate=0., sample_weights=None, **kwargs):
        x, y, x_test, y_test, names = self._k_series_initialization(k, data)
        n_cv = int(cv_rate * len(x))
        print_settings = True
        if sample_weights is not None:
            self._sample_weights = np.asarray(sample_weights, np.float32)
        sample_weights_store = self._sample_weights
        print("Training k-random with k={}, cv_rate={} and test_rate={}".format(k, cv_rate, test_rate))
        for i in range(k):
            self.reset_all_variables()
            all_idx = np.random.permutation(len(x))
            cv_idx, train_idx = all_idx[:n_cv], all_idx[n_cv:]
            x_cv, y_cv = x[cv_idx], y[cv_idx]
            x_train, y_train = x[train_idx], y[train_idx]
            if sample_weights is not None:
                self._sample_weights = sample_weights_store[train_idx]
            else:
                self._sample_weights = None
            kwargs["print_settings"] = print_settings
            kwargs["names"] = names[i]
            self.data_info["stage"] = 2
            self.fit(x_train, y_train, x_cv, y_cv, **kwargs)
            self._k_series_evaluation(i, x_test, y_test)
            print_settings = False
        self.data_info["stage"] = 3
        self._merge_preprocessors_from_k_series(names)
        self._sample_weights = sample_weights_store
        if x_test is not None and y_test is not None:
            self._test_generator = self._generator_base(x_test, y_test, name="TestGenerator")
        return self


if __name__ == '__main__':
    Dist(
        name="Zhongan",
        data_info={"data_folder": "../f_AutoNN/_Data"},
        model_param_settings={"max_epoch": 1},
        model_structure_settings={"use_wide_network": False, "use_pruner": False}
    ).k_random(snapshot_ratio=0).save()
    Dist(
        name="Zhongan",
        data_info={"data_folder": "../f_AutoNN/_Data"},
        model_param_settings={"max_epoch": 3},
        model_structure_settings={"use_wide_network": False, "use_pruner": False}
    ).load().fit(snapshot_ratio=0)
