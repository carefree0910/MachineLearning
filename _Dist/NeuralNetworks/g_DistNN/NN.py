import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import tensorflow as tf

from _Dist.NeuralNetworks.Base import Generator
from _Dist.NeuralNetworks.f_AutoNN.NN import Auto


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
                    self._test_generator = Generator(x_test, y_test, name="TestGenerator")
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
            self._train_generator = Generator(x, y, self._sample_weights, name="Generator")
            if x_test is not None and y_test is not None:
                self._test_generator = Generator(x_test, y_test, name="TestGenerator")
        self.fit(**kwargs)
        x, y, _ = self._gen_batch(self._train_generator, self.n_random_train_subset, True)
        print("  -  Performance of increment fit", end=" | ")
        self._evaluate(x, y, x_test, y_test)
        return self

    def k_fold(self, k=10, data=None, test_rate=0., sample_weights=None, **kwargs):
        if data is not None:
            self._load_data(data)
        n_batch = int(len(self._train_generator) / k)
        all_idx = list(range(len(self._train_generator)))
        print_settings = True
        if sample_weights is not None:
            self._sample_weights = np.asarray(sample_weights, np.float32)
        test_generator_store, sample_weights_store = self._test_generator, self._sample_weights
        if test_generator_store is not None:
            test_data, _ = test_generator_store.get_all_data()
            x_test, y_test = test_data[..., :-1], test_data[..., -1]
        else:
            x_test = y_test = None
        print("Training k-fold with k={} and test_rate={}".format(k, test_rate))
        for i in range(k):
            self.reset_all_variables()
            cv_idx = list(range(i * n_batch, (i + 1) * n_batch))
            train_idx = [j for j in all_idx if j < i * n_batch or j >= (i + 1) * n_batch]
            with self._train_generator:
                test_data, _ = self._train_generator.get_indices(cv_idx)
                x_cv, y_cv = test_data[..., :-1], test_data[..., -1]
                self._train_generator.set_indices(train_idx)
                self._test_generator = Generator(x_cv, y_cv, name="TestGenerator")
                if sample_weights is not None:
                    self._sample_weights = sample_weights_store[train_idx]
                else:
                    self._sample_weights = None
                kwargs["print_settings"] = print_settings
                self.fit(**kwargs)
                x, y, _ = self._gen_batch(self._train_generator, self.n_random_train_subset, True)
                print("  -  Performance of fold {}".format(i+1), end=" | ")
                self._evaluate(x, y, x_cv, y_cv, x_test, y_test)
                print_settings = False
        self._test_generator, self._sample_weights = test_generator_store, sample_weights_store
        return self

    def k_random(self, k=3, data=None, cv_rate=0.1, test_rate=0., sample_weights=None, **kwargs):
        if data is not None:
            self._load_data(data)
        n_cv = int(cv_rate * len(self._train_generator))
        print_settings = True
        if sample_weights is not None:
            self._sample_weights = np.asarray(sample_weights, np.float32)
        test_generator_store, sample_weights_store = self._test_generator, self._sample_weights
        if test_generator_store is not None:
            cv_data, _ = test_generator_store.get_all_data()
            x_test, y_test = cv_data[..., :-1], cv_data[..., -1]
        else:
            x_test = y_test = None
        print("Training k-random with k={}, cv_rate={} and test_rate={}".format(k, cv_rate, test_rate))
        for i in range(k):
            self.reset_all_variables()
            all_idx = np.random.permutation(len(self._train_generator))
            cv_idx, train_idx = all_idx[:n_cv], all_idx[n_cv:]
            with self._train_generator:
                cv_data, _ = self._train_generator.get_indices(cv_idx)
                x_cv, y_cv = cv_data[..., :-1], cv_data[..., -1]
                self._train_generator.set_indices(train_idx)
                # self._test_generator = Generator(x_cv, y_cv, name="TestGenerator")
                if sample_weights is not None:
                    self._sample_weights = sample_weights_store[train_idx]
                else:
                    self._sample_weights = None
                kwargs["print_settings"] = print_settings
                self.fit(**kwargs)
                x, y, _ = self._gen_batch(self._train_generator, self.n_random_train_subset, True)
                print("  -  Performance of run {}".format(i+1), end=" | ")
                self._evaluate(x, y, x_cv, y_cv, x_test, y_test)
                print_settings = False
        self._test_generator, self._sample_weights = test_generator_store, sample_weights_store
        return self


if __name__ == '__main__':
    Dist(
        name="Zhongan", data_folder="../f_AutoNN/_Data",
        model_param_settings={"max_epoch": 3},
        model_structure_settings={"use_wide_network": False, "use_pruner": False}
    ).fit(snapshot_ratio=0).save()
    Dist(
        name="Zhongan", data_folder="../f_AutoNN/_Data",
        model_param_settings={"max_epoch": 3},
        model_structure_settings={"use_wide_network": False, "use_pruner": False}
    ).load().fit(snapshot_ratio=0)
