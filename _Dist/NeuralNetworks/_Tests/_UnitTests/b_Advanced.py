import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import unittest
import numpy as np

from Util.Util import DataUtil
from _Dist.NeuralNetworks.e_AdvancedNN.NN import Advanced
from _Dist.NeuralNetworks._Tests._UnitTests.UnitTestUtil import clear_cache


base_params = {
    "name": "UnitTest",
    "data_info": {
        "numerical_idx": [True] * 6 + [False],
        "categorical_columns": []
    },
    "model_param_settings": {"n_epoch": 3, "max_epoch": 5}
}
nn = Advanced(**base_params)
train_set, cv_set, test_set = DataUtil.gen_special_linear(1000, 2, 2, 2, one_hot=False)


class TestAdvancedNN(unittest.TestCase):
    def test_00_train(self):
        self.assertIsInstance(
            nn.fit(*train_set, *cv_set, verbose=0), Advanced,
            msg="Train failed"
        )

    def test_01_predict(self):
        self.assertIs(nn.predict(train_set[0]).dtype, np.dtype("float32"), "Predict failed")
        self.assertIs(nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Predict classes failed")
        self.assertIs(nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Predict classes failed")

    def test_02_evaluate(self):
        self.assertEqual(len(nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Evaluation failed")

    def test_03_save(self):
        self.assertIsInstance(nn.save(), Advanced, msg="Save failed")

    def test_04_load(self):
        global nn
        nn = Advanced(**base_params).load()
        self.assertIsInstance(nn, Advanced, "Load failed")

    def test_05_re_predict(self):
        self.assertIs(nn.predict(train_set[0]).dtype, np.dtype("float32"), "Re-Predict failed")
        self.assertIs(nn.predict_classes(cv_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")
        self.assertIs(nn.predict_classes(test_set[0]).dtype, np.dtype("int32"), "Re-Predict classes failed")

    def test_06_re_evaluate(self):
        self.assertEqual(len(nn.evaluate(*train_set, *cv_set, *test_set)), 3, "Re-Evaluation failed")

    def test_07_re_train(self):
        self.assertIsInstance(
            nn.fit(*train_set, *cv_set, verbose=0), Advanced,
            msg="Re-Train failed"
        )

    def test_99_clear_cache(self):
        clear_cache()


if __name__ == '__main__':
    unittest.main()
