import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import copy
import shutil
import unittest
import numpy as np

from Util.Util import DataUtil
from _Dist.NeuralNetworks.b_TraditionalML.SVM import AutoLinearSVM
from _Dist.NeuralNetworks.f_AutoNN.NN import AutoBasic, AutoAdvanced
from _Dist.NeuralNetworks._Tests._UnitTests.UnitTestUtil import clear_cache


base_params = {
    "name": "UnitTest", "data_info": {},
    "model_param_settings": {"n_epoch": 1, "max_epoch": 2}
}
nn = AutoAdvanced(**copy.deepcopy(base_params))
basic_nn = AutoBasic(**copy.deepcopy(base_params))
linear_svm = AutoLinearSVM(**copy.deepcopy(base_params))
train_set, cv_set, test_set = DataUtil.gen_special_linear(1000, 2, 2, 2, one_hot=False)

auto_mushroom_params = copy.deepcopy(base_params)
auto_mushroom_params["name"] = "mushroom"
auto_mushroom_params["data_info"]["file_type"] = "txt"
auto_mushroom_params["data_info"]["data_folder"] = "../_Data"
mushroom_labels = {"p", "e"}
mushroom_file = "../_Data/mushroom"
auto_mushroom_nn = AutoAdvanced(**copy.deepcopy(auto_mushroom_params))
auto_mushroom_basic_nn = AutoBasic(**copy.deepcopy(auto_mushroom_params))
auto_mushroom_linear_svm = AutoLinearSVM(**copy.deepcopy(auto_mushroom_params))

auto_adult_params = copy.deepcopy(base_params)
auto_adult_params["name"] = "Adult"
auto_adult_params["data_info"]["file_type"] = "csv"
auto_adult_params["data_info"]["data_folder"] = "../_Data"
adult_file = "../_Data/Adult/test"
auto_adult_nn = AutoAdvanced(**copy.deepcopy(auto_adult_params))
auto_adult_basic_nn = AutoBasic(**copy.deepcopy(auto_adult_params))
auto_adult_linear_svm = AutoLinearSVM(**copy.deepcopy(auto_adult_params))

auto_lmgpip_params = copy.deepcopy(base_params)
auto_lmgpip_params["name"] = "lmgpip"
auto_lmgpip_params["data_info"]["file_type"] = "txt"
auto_lmgpip_params["data_info"]["data_folder"] = "../_Data"
lmgpip_test_file = "../_Data/lmgpip_test"
auto_lmgpip_nn = AutoAdvanced(**auto_lmgpip_params)
auto_lmgpip_basic_nn = AutoBasic(**auto_lmgpip_params)

auto_lmgpip_regressor_params = copy.deepcopy(auto_lmgpip_params)
auto_lmgpip_regressor_params["data_info"]["is_regression"] = True
auto_lmgpip_regressor = AutoAdvanced(**auto_lmgpip_regressor_params)
auto_lmgpip_basic_regressor = AutoBasic(**auto_lmgpip_regressor_params)


class TestAutoNN(unittest.TestCase):
    def test_00_train_from_numpy(self):
        self.assertIsInstance(
            nn.fit(*train_set, *cv_set, verbose=0), AutoAdvanced,
            msg="Train failed"
        )
        self.assertIsInstance(
            basic_nn.fit(*train_set, *cv_set, verbose=0), AutoBasic,
            msg="Train failed"
        )

    def test_01_predict(self):
        self.assertIs(nn.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(basic_nn.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Predict classes failed")
        self.assertIs(basic_nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Predict classes failed")
        self.assertIs(nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Predict classes failed")
        self.assertIs(basic_nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Predict classes failed")

    def test_02_evaluate(self):
        self.assertEqual(len(nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Evaluation failed")
        self.assertEqual(len(basic_nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Evaluation failed")

    def test_03_save(self):
        self.assertIsInstance(nn.save(), AutoAdvanced, msg="Save failed")
        self.assertIsInstance(basic_nn.save(), AutoBasic, msg="Save failed")

    def test_04_load(self):
        global nn, basic_nn
        nn = AutoAdvanced(**base_params).load()
        basic_nn = AutoBasic(**base_params).load()
        self.assertIsInstance(nn, AutoAdvanced, "Load failed")
        self.assertIsInstance(basic_nn, AutoBasic, "Load failed")

    def test_05_re_predict(self):
        self.assertIs(nn.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(basic_nn.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")
        self.assertIs(basic_nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")
        self.assertIs(nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")
        self.assertIs(basic_nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")

    def test_06_re_evaluate(self):
        self.assertEqual(len(nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Re-Evaluation failed")
        self.assertEqual(len(basic_nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Re-Evaluation failed")

    def test_07_re_train(self):
        self.assertIsInstance(
            nn.fit(*train_set, *cv_set, verbose=0), AutoAdvanced,
            msg="Re-Train failed"
        )
        self.assertIsInstance(
            basic_nn.fit(*train_set, *cv_set, verbose=0), AutoBasic,
            msg="Re-Train failed"
        )

    def test_08_train_from_txt(self):
        self.assertIsInstance(
            auto_mushroom_nn.fit(verbose=0), AutoAdvanced,
            msg="Train failed"
        )
        self.assertIsInstance(
            auto_mushroom_basic_nn.fit(verbose=0), AutoBasic,
            msg="Train failed"
        )

    def test_09_predict_from_txt(self):
        for local_nn in (auto_mushroom_nn, auto_mushroom_basic_nn):
            self.assertIs(
                local_nn.predict_from_file(
                    mushroom_file, "txt", include_label=True
                ).dtype, np.dtype("float32"), msg="Predict failed"
            )
            self.assertIs(
                local_nn.predict_classes_from_file(
                    mushroom_file, "txt", include_label=True
                ).dtype, np.dtype("int32"), msg="Predict classes failed"
            )
            predict_labels = np.unique(local_nn.predict_labels_from_file(
                mushroom_file, "txt", include_label=True
            ))
            for pred_label in predict_labels:
                self.assertIn(pred_label, mushroom_labels, msg="Predict labels failed")

    def test_10_save(self):
        self.assertIsInstance(auto_mushroom_nn.save(), AutoAdvanced, msg="Save failed")
        self.assertIsInstance(auto_mushroom_basic_nn.save(), AutoBasic, msg="Save failed")

    def test_11_load(self):
        global auto_mushroom_nn, auto_mushroom_basic_nn
        auto_mushroom_nn = AutoAdvanced(**auto_mushroom_params).load()
        auto_mushroom_basic_nn = AutoBasic(**auto_mushroom_params).load()
        self.assertIsInstance(auto_mushroom_nn, AutoAdvanced, "Load failed")
        self.assertIsInstance(auto_mushroom_basic_nn, AutoBasic, "Load failed")

    def test_12_re_predict_from_txt(self):
        for local_nn in (auto_mushroom_nn, auto_mushroom_basic_nn):
            self.assertIs(
                local_nn.predict_from_file(
                    mushroom_file, "txt", include_label=True
                ).dtype, np.dtype("float32"), msg="Re-Predict failed"
            )
            self.assertIs(
                local_nn.predict_classes_from_file(
                    mushroom_file, "txt", include_label=True
                ).dtype, np.dtype("int32"), msg="Re-Predict classes failed"
            )
            predict_labels = np.unique(local_nn.predict_labels_from_file(
                mushroom_file, "txt", include_label=True
            ))
            for pred_label in predict_labels:
                self.assertIn(pred_label, mushroom_labels, msg="Re-Predict labels failed")

    def test_13_re_train_from_txt(self):
        self.assertIsInstance(
            auto_mushroom_nn.fit(verbose=0), AutoAdvanced,
            msg="Re-Train failed"
        )
        self.assertIsInstance(
            auto_mushroom_basic_nn.fit(verbose=0), AutoBasic,
            msg="Re-Train failed"
        )

    def test_14_train_from_csv(self):
        self.assertIsInstance(
            auto_adult_nn.fit(verbose=0), AutoAdvanced,
            msg="Train failed"
        )
        self.assertIsInstance(
            auto_adult_basic_nn.fit(verbose=0), AutoBasic,
            msg="Train failed"
        )

    def test_15_predict_from_csv(self):
        for local_nn in (auto_adult_nn, auto_adult_basic_nn):
            self.assertIs(
                local_nn.predict_from_file(
                    adult_file, "csv", include_label=True
                ).dtype, np.dtype("float32"), msg="Predict failed"
            )
            self.assertIs(
                local_nn.predict_classes_from_file(
                    adult_file, "csv", include_label=True
                ).dtype, np.dtype("int32"), msg="Predict classes failed"
            )

    def test_16_save(self):
        self.assertIsInstance(auto_adult_nn.save(), AutoAdvanced, msg="Save failed")
        self.assertIsInstance(auto_adult_basic_nn.save(), AutoBasic, msg="Save failed")

    def test_17_load(self):
        global auto_adult_nn, auto_adult_basic_nn
        auto_adult_nn = AutoAdvanced(**auto_adult_params).load()
        auto_adult_basic_nn = AutoBasic(**auto_adult_params).load()
        self.assertIsInstance(auto_adult_nn, AutoAdvanced, "Load failed")
        self.assertIsInstance(auto_adult_basic_nn, AutoBasic, "Load failed")

    def test_18_re_predict_from_csv(self):
        for local_nn in (auto_adult_nn, auto_adult_basic_nn):
            self.assertIs(
                local_nn.predict_from_file(
                    adult_file, "csv", include_label=True
                ).dtype, np.dtype("float32"), msg="Re-Predict failed"
            )
            self.assertIs(
                local_nn.predict_classes_from_file(
                    adult_file, "csv", include_label=True
                ).dtype, np.dtype("int32"), msg="Re-Predict classes failed"
            )

    def test_19_re_train_from_csv(self):
        self.assertIsInstance(
            auto_adult_nn.fit(verbose=0), AutoAdvanced,
            msg="Re-Train failed"
        )
        self.assertIsInstance(
            auto_adult_basic_nn.fit(verbose=0), AutoBasic,
            msg="Re-Train failed"
        )

    def test_20_train_from_mixed_features(self):
        self.assertIsInstance(
            auto_lmgpip_nn.fit(verbose=0), AutoAdvanced,
            msg="Train failed"
        )
        self.assertIsInstance(
            auto_lmgpip_basic_nn.fit(verbose=0), AutoBasic,
            msg="Train failed"
        )

    def test_21_predict_from_mixed_features(self):
        for local_nn in (auto_lmgpip_nn, auto_lmgpip_basic_nn):
            self.assertIs(
                local_nn.predict_from_file(
                    lmgpip_test_file, "txt", include_label=False
                ).dtype, np.dtype("float32"), msg="Predict failed"
            )
            self.assertIs(
                local_nn.predict_classes_from_file(
                    lmgpip_test_file, "txt", include_label=False
                ).dtype, np.dtype("int32"), msg="Predict classes failed"
            )

    def test_22_save(self):
        self.assertIsInstance(auto_lmgpip_nn.save(), AutoAdvanced, msg="Save failed")
        self.assertIsInstance(auto_lmgpip_basic_nn.save(), AutoBasic, msg="Save failed")

    def test_23_load(self):
        global auto_lmgpip_nn, auto_lmgpip_basic_nn
        auto_lmgpip_nn = AutoAdvanced(**auto_lmgpip_params).load()
        auto_lmgpip_basic_nn = AutoBasic(**auto_lmgpip_params).load()
        self.assertIsInstance(auto_lmgpip_nn, AutoAdvanced, "Load failed")
        self.assertIsInstance(auto_lmgpip_basic_nn, AutoBasic, "Load failed")

    def test_24_re_predict_from_mixed_features(self):
        for local_nn in (auto_lmgpip_nn, auto_lmgpip_basic_nn):
            self.assertIs(
                local_nn.predict_from_file(
                    lmgpip_test_file, "txt", include_label=False
                ).dtype, np.dtype("float32"), msg="Re-Predict failed"
            )
            self.assertIs(
                local_nn.predict_classes_from_file(
                    lmgpip_test_file, "txt", include_label=False
                ).dtype, np.dtype("int32"), msg="Re-Predict classes failed"
            )

    def test_25_re_train_from_mixed_features(self):
        self.assertIsInstance(
            auto_lmgpip_nn.fit(verbose=0), AutoAdvanced,
            msg="Re-Train failed"
        )
        self.assertIsInstance(
            auto_lmgpip_basic_nn.fit(verbose=0), AutoBasic,
            msg="Re-Train failed"
        )

    def test_26_train_regressor_from_mixed_features(self):
        shutil.rmtree("../_Data/_Cache")
        shutil.rmtree("../_Data/_DataInfo")
        self.assertIsInstance(
            auto_lmgpip_regressor.fit(verbose=0), AutoAdvanced,
            msg="Train failed"
        )
        self.assertIsInstance(
            auto_lmgpip_basic_regressor.fit(verbose=0), AutoBasic,
            msg="Train failed"
        )

    def test_27_predict_regressor_from_mixed_features(self):
        for local_nn in (auto_lmgpip_regressor, auto_lmgpip_basic_regressor):
            self.assertIs(
                local_nn.predict_from_file(
                    lmgpip_test_file, "txt", include_label=False
                ).dtype, np.dtype("float32"), msg="Predict failed"
            )
            with self.assertRaises(ValueError):
                local_nn.predict_classes_from_file(
                    lmgpip_test_file, "txt", include_label=False
                )

    def test_28_save(self):
        self.assertIsInstance(auto_lmgpip_regressor.save(), AutoAdvanced, msg="Save failed")
        self.assertIsInstance(auto_lmgpip_basic_regressor.save(), AutoBasic, msg="Save failed")

    def test_29_load(self):
        global auto_lmgpip_regressor, auto_lmgpip_basic_regressor
        auto_lmgpip_regressor = AutoAdvanced(**auto_lmgpip_regressor_params).load()
        auto_lmgpip_basic_regressor = AutoBasic(**auto_lmgpip_regressor_params).load()
        self.assertIsInstance(auto_lmgpip_regressor, AutoAdvanced, "Load failed")
        self.assertIsInstance(auto_lmgpip_basic_regressor, AutoBasic, "Load failed")

    def test_30_re_predict_regressor_from_mixed_features(self):
        for local_nn in (auto_lmgpip_regressor, auto_lmgpip_basic_regressor):
            self.assertIs(
                local_nn.predict_from_file(
                    lmgpip_test_file, "txt", include_label=False
                ).dtype, np.dtype("float32"), msg="Re-Predict failed"
            )
            with self.assertRaises(ValueError):
                local_nn.predict_classes_from_file(
                    lmgpip_test_file, "txt", include_label=False
                )

    def test_31_re_train_regressor_from_mixed_features(self):
        self.assertIsInstance(
            auto_lmgpip_regressor.fit(verbose=0), AutoAdvanced,
            msg="Re-Train failed"
        )
        self.assertIsInstance(
            auto_lmgpip_basic_regressor.fit(verbose=0), AutoBasic,
            msg="Re-Train failed"
        )

    def test_99_clear_cache(self):
        clear_cache()


class TestAutoLinearSVM(unittest.TestCase):
    def test_00_train_from_numpy(self):
        self.assertIsInstance(
            linear_svm.fit(*train_set, *cv_set, verbose=0), AutoLinearSVM,
            msg="Train failed"
        )

    def test_01_predict(self):
        self.assertIs(linear_svm.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(linear_svm.predict(cv_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(linear_svm.predict(test_set[0]).dtype, np.dtype("float32"), "Predict failed")

    def test_02_evaluate(self):
        self.assertEqual(len(linear_svm.evaluate(*train_set, *cv_set, *test_set)), 3, "Evaluation failed")

    def test_03_save(self):
        self.assertIsInstance(linear_svm.save(), AutoLinearSVM, msg="Save failed")

    def test_04_load(self):
        global linear_svm
        linear_svm = AutoLinearSVM(**base_params).load()
        self.assertIsInstance(linear_svm, AutoLinearSVM, "Load failed")

    def test_05_re_predict(self):
        self.assertIs(linear_svm.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(linear_svm.predict(cv_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(linear_svm.predict(test_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")

    def test_06_re_evaluate(self):
        self.assertEqual(len(linear_svm.evaluate(*train_set, *cv_set, *test_set)), 3, "Re-Evaluation failed")

    def test_07_re_train(self):
        self.assertIsInstance(
            linear_svm.fit(*train_set, *cv_set, verbose=0), AutoLinearSVM,
            msg="Re-Train failed"
        )

    def test_08_train_from_txt(self):
        self.assertIsInstance(
            auto_mushroom_linear_svm.fit(verbose=0), AutoLinearSVM,
            msg="Train failed"
        )

    def test_09_predict_from_txt(self):
        self.assertIs(
            auto_mushroom_linear_svm.predict_from_file(
                mushroom_file, "txt", include_label=True
            ).dtype, np.dtype("float32"), msg="Predict failed"
        )

    def test_10_save(self):
        self.assertIsInstance(auto_mushroom_linear_svm.save(), AutoLinearSVM, msg="Save failed")

    def test_11_load(self):
        global auto_mushroom_linear_svm
        auto_mushroom_linear_svm = AutoLinearSVM(**auto_mushroom_params).load()
        self.assertIsInstance(auto_mushroom_linear_svm, AutoLinearSVM, "Load failed")

    def test_12_re_predict_from_txt(self):
        self.assertIs(
            auto_mushroom_linear_svm.predict_from_file(
                mushroom_file, "txt", include_label=True
            ).dtype, np.dtype("float32"), msg="Re-Predict failed"
        )

    def test_13_re_train_from_txt(self):
        self.assertIsInstance(
            auto_mushroom_linear_svm.fit(verbose=0), AutoLinearSVM,
            msg="Re-Train failed"
        )

    def test_14_train_from_csv(self):
        self.assertIsInstance(
            auto_adult_linear_svm.fit(verbose=0), AutoLinearSVM,
            msg="Train failed"
        )

    def test_15_predict_from_csv(self):
        self.assertIs(
            auto_adult_linear_svm.predict_from_file(
                adult_file, "csv", include_label=True
            ).dtype, np.dtype("float32"), msg="Predict failed"
        )

    def test_16_save(self):
        self.assertIsInstance(auto_adult_linear_svm.save(), AutoLinearSVM, msg="Save failed")

    def test_17_load(self):
        global auto_adult_linear_svm
        auto_adult_linear_svm = AutoLinearSVM(**auto_adult_params).load()
        self.assertIsInstance(auto_adult_linear_svm, AutoLinearSVM, "Load failed")

    def test_18_re_predict_from_csv(self):
        self.assertIs(
            auto_adult_linear_svm.predict_from_file(
                adult_file, "csv", include_label=True
            ).dtype, np.dtype("float32"), msg="Re-Predict failed"
        )

    def test_19_re_train_from_csv(self):
        self.assertIsInstance(
            auto_adult_linear_svm.fit(verbose=0), AutoLinearSVM,
            msg="Re-Train failed"
        )

    def test_99_clear_cache(self):
        clear_cache()


if __name__ == '__main__':
    unittest.main()
