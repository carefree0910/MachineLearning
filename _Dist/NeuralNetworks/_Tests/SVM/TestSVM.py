import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from copy import deepcopy

from Util.Util import DataUtil
from _Dist.NeuralNetworks._Tests.TestUtil import draw_acc
from _Dist.NeuralNetworks.b_TraditionalML.SVM import SVM
from _Dist.NeuralNetworks.f_AutoNN.DistNN import AutoAdvanced

base_params = {"model_param_settings": {"n_epoch": 30, "metric": "acc"}}
(x, y), (x_test, y_test) = DataUtil.gen_noisy_linear(n_dim=2, n_valid=2, test_ratio=0.01, one_hot=False)
svm = SVM(**deepcopy(base_params)).fit(
    x, y, x_test, y_test, snapshot_ratio=1).visualize2d(x_test, y_test)
nn = AutoAdvanced("NoisyLinear", **deepcopy(base_params), pre_process_settings={"reuse_mean_and_std": True}).fit(
    x, y, x_test, y_test, snapshot_ratio=1).visualize2d(x_test, y_test)
draw_acc(svm, nn)
