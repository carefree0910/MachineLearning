import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import copy
import unittest
import numpy as np

from Util.Util import DataUtil
from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.b_TraditionalML.SVM import LinearSVM, SVM
from _Dist.NeuralNetworks._Tests._UnitTests.UnitTestUtil import clear_cache


base_params = {
    "name": "UnitTest",
    "model_param_settings": {"n_epoch": 3, "max_epoch": 5}
}
svm = SVM(**copy.deepcopy(base_params))
nn = Basic(**copy.deepcopy(base_params))
linear_svm = LinearSVM(**copy.deepcopy(base_params))
train_set, cv_set, test_set = DataUtil.gen_special_linear(1000, 2, 2, 2, one_hot=False)


class TestSVM(unittest.TestCase):
    def test_00_train(self):
        self.assertIsInstance(
            svm.fit(*train_set, *cv_set, verbose=0), SVM,
            msg="Train failed"
        )

    def test_01_predict(self):
        self.assertIs(svm.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(svm.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Predict classes failed")

    def test_02_evaluate(self):
        self.assertEqual(len(svm.evaluate(*train_set, *cv_set)), 3, "Evaluation failed")

    def test_03_save(self):
        self.assertIsInstance(svm.save(), SVM, msg="Save failed")

    def test_04_load(self):
        global svm
        svm = SVM(base_params["name"]).load()
        self.assertIsInstance(svm, SVM, "Load failed")

    def test_05_re_predict(self):
        self.assertIs(svm.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(svm.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")

    def test_06_re_evaluate(self):
        self.assertEqual(len(svm.evaluate(*train_set, *cv_set)), 3, "Re-Evaluation failed")

    def test_07_re_train(self):
        self.assertIsInstance(
            svm.fit(*train_set, *cv_set, verbose=0), SVM,
            msg="Re-Train failed"
        )

    def test_99_clear_cache(self):
        clear_cache()


class TestBasicNN(unittest.TestCase):
    def test_00_train(self):
        self.assertIsInstance(
            nn.fit(*train_set, *cv_set, verbose=0), Basic,
            msg="Train failed"
        )

    def test_01_predict(self):
        self.assertIs(nn.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Predict classes failed")

    def test_02_evaluate(self):
        self.assertEqual(len(nn.evaluate(*train_set, *cv_set)), 3, "Evaluation failed")

    def test_03_save(self):
        self.assertIsInstance(nn.save(), Basic, msg="Save failed")

    def test_04_load(self):
        global nn
        nn = Basic(base_params["name"]).load()
        self.assertIsInstance(nn, Basic, "Load failed")

    def test_05_re_predict(self):
        self.assertIs(nn.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")

    def test_06_re_evaluate(self):
        self.assertEqual(len(nn.evaluate(*train_set, *cv_set)), 3, "Re-Evaluation failed")

    def test_07_re_train(self):
        self.assertIsInstance(
            nn.fit(*train_set, *cv_set, verbose=0), Basic,
            msg="Re-Train failed"
        )

    def test_99_clear_cache(self):
        clear_cache()


class TestLinearSVM(unittest.TestCase):
    def test_00_train(self):
        self.assertIsInstance(
            linear_svm.fit(*train_set, *cv_set, verbose=0), LinearSVM,
            msg="Train failed"
        )

    def test_01_predict(self):
        self.assertIs(linear_svm.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(linear_svm.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Predict classes failed")
        self.assertIs(linear_svm.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Predict classes failed")

    def test_02_evaluate(self):
        self.assertEqual(len(linear_svm.evaluate(*train_set, *cv_set, *test_set)), 3, "Evaluation failed")

    def test_03_save(self):
        self.assertIsInstance(linear_svm.save(), LinearSVM, msg="Save failed")

    def test_04_load(self):
        global linear_svm
        linear_svm = LinearSVM(base_params["name"]).load()
        self.assertIsInstance(linear_svm, LinearSVM, "Load failed")

    def test_05_re_predict(self):
        self.assertIs(linear_svm.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(linear_svm.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")
        self.assertIs(linear_svm.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")

    def test_06_re_evaluate(self):
        self.assertEqual(len(linear_svm.evaluate(*train_set, *cv_set, *test_set)), 3, "Re-Evaluation failed")

    def test_07_re_train(self):
        self.assertIsInstance(
            linear_svm.fit(*train_set, *cv_set, verbose=0), LinearSVM,
            msg="Re-Train failed"
        )

    def test_99_clear_cache(self):
        clear_cache()


if __name__ == '__main__':
    unittest.main()
