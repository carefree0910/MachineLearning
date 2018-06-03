import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import copy
import unittest
import numpy as np

from Util.Util import DataUtil
from _Dist.NeuralNetworks.b_TraditionalML.SVM import DistLinearSVM
from _Dist.NeuralNetworks.g_DistNN.NN import DistBasic, DistAdvanced
from _Dist.NeuralNetworks._Tests._UnitTests.UnitTestUtil import clear_cache


base_params = {
    "name": "UnitTest", "data_info": {},
    "model_param_settings": {"n_epoch": 1, "max_epoch": 2}
}
nn = DistAdvanced(**copy.deepcopy(base_params))
basic_nn = DistBasic(**copy.deepcopy(base_params))
linear_svm = DistLinearSVM(**copy.deepcopy(base_params))
train_set, cv_set, test_set = DataUtil.gen_special_linear(1000, 2, 2, 2, one_hot=False)
(x, y), (x_cv, y_cv), (x_test, y_test) = train_set, cv_set, test_set
train_data = np.hstack([x, y.reshape([-1, 1])])
cv_data = np.hstack([x_cv, y_cv.reshape([-1, 1])])
test_data = np.hstack([x_test, y_test.reshape([-1, 1])])
train_and_cv_data = np.vstack([train_data, cv_data])


class TestDistNN(unittest.TestCase):
    def test_00_k_series_from_numpy(self):
        self.assertIsInstance(
            nn.k_random(3, (train_and_cv_data, test_data), verbose=0), DistAdvanced,
            msg="k-random failed"
        )
        self.assertIsInstance(
            nn.k_fold(3, (train_and_cv_data, test_data), verbose=0), DistAdvanced,
            msg="k-fold failed"
        )
        self.assertIsInstance(
            basic_nn.k_random(3, (train_and_cv_data, test_data), verbose=0), DistBasic,
            msg="k-random failed"
        )
        self.assertIsInstance(
            basic_nn.k_fold(3, (train_and_cv_data, test_data), verbose=0), DistBasic,
            msg="k-fold failed"
        )

    def test_01_predict(self):
        self.assertIs(nn.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Predict classes failed")
        self.assertIs(nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Predict classes failed")
        self.assertIs(basic_nn.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(basic_nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Predict classes failed")
        self.assertIs(basic_nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Predict classes failed")

    def test_02_evaluate(self):
        self.assertEqual(len(nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Evaluation failed")
        self.assertEqual(len(basic_nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Evaluation failed")

    def test_03_save(self):
        self.assertIsInstance(nn.save(), DistAdvanced, msg="Save failed")
        self.assertIsInstance(basic_nn.save(), DistBasic, msg="Save failed")

    def test_04_load(self):
        global nn, basic_nn
        nn = DistAdvanced(**base_params).load()
        basic_nn = DistBasic(**base_params).load()
        self.assertIsInstance(nn, DistAdvanced, "Load failed")
        self.assertIsInstance(basic_nn, DistBasic, "Load failed")

    def test_05_re_predict(self):
        self.assertIs(nn.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")
        self.assertIs(nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")
        self.assertIs(basic_nn.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(basic_nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")
        self.assertIs(basic_nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")

    def test_06_re_evaluate(self):
        self.assertEqual(len(nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Re-Evaluation failed")
        self.assertEqual(len(basic_nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Re-Evaluation failed")

    def test_07_param_search(self):
        params = [
            {"model_param_settings": {"lr": 1e-2}},
            {"model_param_settings": {"lr": 1e-3}, "model_structure_settings": {"use_pruner": False}}
        ]
        self.assertIsInstance(
            nn.param_search(params, data=(train_and_cv_data, test_data), verbose=0), DistAdvanced,
            msg="param_search failed"
        )
        self.assertIsInstance(
            basic_nn.param_search(params, data=(train_and_cv_data, test_data), verbose=0), DistBasic,
            msg="param_search failed"
        )

    def test_08_random_search(self):
        list_first_grid_params = {
            "model_param_settings": [
                {"lr": 1e-2},
                {"lr": 1e-2, "loss": "mse"},
                {"lr": 1e-3, "loss": "mse"}
            ],
            "model_structure_settings": [
                {"hidden_units": [256, 256]},
                {"hidden_units": [128, 128], "use_pruner": False},
                {"hidden_units": [128, 128], "use_pruner": False, "use_wide_network": False}
            ]
        }
        dict_first_grid_params = {
            "model_param_settings": {
                "lr": [1e-2, 1e-3],
                "loss": ["mse", "cross_entropy"]
            },
            "model_structure_settings": {
                "use_pruner": [False, True],
                "use_wide_network": [False, True],
                "hidden_units": [[128, 128], [256, 256]]
            },
        }
        self.assertIsInstance(
            nn.random_search(
                4, list_first_grid_params, grid_order="list_first",
                data=(train_and_cv_data, test_data), verbose=0
            ), DistAdvanced, msg="list_first_grid_search failed"
        )
        self.assertIsInstance(
            nn.random_search(
                8, dict_first_grid_params, grid_order="dict_first",
                data=(train_and_cv_data, test_data), verbose=0
            ), DistAdvanced, msg="dict_first_grid_search failed"
        )
        self.assertIsInstance(
            basic_nn.random_search(
                4, list_first_grid_params, grid_order="list_first",
                data=(train_and_cv_data, test_data), verbose=0
            ), DistBasic, msg="list_first_grid_search failed"
        )
        self.assertIsInstance(
            basic_nn.random_search(
                8, dict_first_grid_params, grid_order="dict_first",
                data=(train_and_cv_data, test_data), verbose=0
            ), DistBasic, msg="dict_first_grid_search failed"
        )

    def test_09_grid_search(self):
        list_first_grid_params = {
            "model_param_settings": [
                {"lr": 1e-2},
                {"lr": 1e-3, "loss": "mse"}
            ],
            "model_structure_settings": [
                {"hidden_units": [256, 256]},
                {"hidden_units": [128, 128], "use_pruner": False}
            ]
        }
        dict_first_grid_params = {
            "model_param_settings": {
                "lr": [1e-3, 1e-2],
                "loss": ["mse", "cross_entropy"]
            },
            "model_structure_settings": {
                "hidden_units": [[128, 256], [128, 256]]
            }
        }
        self.assertIsInstance(
            nn.grid_search(
                list_first_grid_params, grid_order="list_first",
                data=(train_and_cv_data, test_data), verbose=0
            ), DistAdvanced, msg="list_first_grid_search failed"
        )
        self.assertIsInstance(
            nn.grid_search(
                dict_first_grid_params, grid_order="dict_first",
                data=(train_and_cv_data, test_data), verbose=0
            ), DistAdvanced, msg="dict_first_grid_search failed"
        )
        self.assertIsInstance(
            basic_nn.grid_search(
                list_first_grid_params, grid_order="list_first",
                data=(train_and_cv_data, test_data), verbose=0
            ), DistBasic, msg="list_first_grid_search failed"
        )
        self.assertIsInstance(
            basic_nn.grid_search(
                dict_first_grid_params, grid_order="dict_first",
                data=(train_and_cv_data, test_data), verbose=0
            ), DistBasic, msg="dict_first_grid_search failed"
        )

    def test_10_range_search(self):
        range_grid_params = {
            "model_param_settings": {
                "lr": ["float", 1e-3, 1e-1, "log"],
                "loss": ["choice", ["mse", "cross_entropy"]]
            },
            "model_structure_settings": {
                "hidden_units": [
                    ["int", "int"],
                    [128, 256], [128, 256]
                ],
                "pruner_params": {
                    "alpha": ["float", 1e-4, 1e-2, "log"],
                    "beta": ["float", 0.3, 3, "log"],
                    "gamma": ["float", 0.5, 2, "log"]
                }
            },
            "pre_process_settings": {
                "pre_process_method": ["choice", ["normalize", None]],
                "reuse_mean_and_std": ["choice", [True, False]]
            },
            "nan_handler_settings": {
                "nan_handler_method": ["choice", ["median", "mean"]],
                "reuse_nan_handler_values": ["choice", [True, False]]
            }
        }
        self.assertIsInstance(
            nn.range_search(
                8, range_grid_params,
                data=(train_and_cv_data, test_data), verbose=0
            ), DistAdvanced, msg="range_search failed"
        )
        self.assertIsInstance(
            basic_nn.range_search(
                8, range_grid_params,
                data=(train_and_cv_data, test_data), verbose=0
            ), DistBasic, msg="range_search failed"
        )

    def test_99_clear_cache(self):
        clear_cache()


class TestDistLinearSVM(unittest.TestCase):
    def test_00_k_series_from_numpy(self):
        self.assertIsInstance(
            linear_svm.k_random(3, (train_and_cv_data, test_data), verbose=0), DistLinearSVM,
            msg="k-random failed"
        )
        self.assertIsInstance(
            linear_svm.k_fold(3, (train_and_cv_data, test_data), verbose=0), DistLinearSVM,
            msg="k-fold failed"
        )

    def test_01_predict(self):
        self.assertIs(linear_svm.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(linear_svm.predict(cv_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(linear_svm.predict(test_set[0]).dtype, np.dtype("float32"), "Predict failed")

    def test_02_evaluate(self):
        self.assertEqual(len(linear_svm.evaluate(*train_set, *cv_set, *test_set)), 3, "Evaluation failed")

    def test_03_save(self):
        self.assertIsInstance(linear_svm.save(), DistLinearSVM, msg="Save failed")

    def test_04_load(self):
        global linear_svm
        model = DistLinearSVM(**base_params).load()
        self.assertIsInstance(model, DistLinearSVM, "Load failed")

    def test_05_re_predict(self):
        self.assertIs(linear_svm.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(linear_svm.predict(cv_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(linear_svm.predict(test_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")

    def test_06_re_evaluate(self):
        self.assertEqual(len(linear_svm.evaluate(*train_set, *cv_set, *test_set)), 3, "Re-Evaluation failed")

    def test_99_clear_cache(self):
        clear_cache()


if __name__ == '__main__':
    unittest.main()
