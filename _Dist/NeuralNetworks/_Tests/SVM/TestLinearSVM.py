import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from copy import deepcopy

from Util.Util import DataUtil
from _Dist.NeuralNetworks._Tests.TestUtil import draw_acc
from _Dist.NeuralNetworks.f_AutoNN.DistNN import AutoAdvanced
from _Dist.NeuralNetworks.b_TraditionalML.SVM import LinearSVM, AutoLinearSVM, DistLinearSVM

base_params = {"model_param_settings": {"n_epoch": 30, "metric": "acc"}}
(x, y), (x_test, y_test) = DataUtil.gen_noisy_linear(n_dim=2, n_valid=2, test_ratio=0.01, one_hot=False)
svm = LinearSVM(**deepcopy(base_params)).fit(
    x, y, x_test, y_test, snapshot_ratio=1).visualize2d(x_test, y_test)
nn = AutoAdvanced("NoisyLinear", **deepcopy(base_params), pre_process_settings={"reuse_mean_and_std": True}).fit(
    x, y, x_test, y_test, snapshot_ratio=1).visualize2d(x_test, y_test)
draw_acc(svm, nn)

base_params["data_info"] = {"data_folder": "../_Data"}
svm = AutoLinearSVM("mushroom", **deepcopy(base_params)).fit(snapshot_ratio=0)
nn = AutoAdvanced("mushroom", **deepcopy(base_params)).fit(snapshot_ratio=0)
draw_acc(svm, nn)

base_params["data_info"]["file_type"] = "csv"
svm = AutoLinearSVM("Adult", **deepcopy(base_params)).fit(snapshot_ratio=0)
nn = AutoAdvanced("Adult", **deepcopy(base_params)).fit(snapshot_ratio=0)
draw_acc(svm, nn)

DistLinearSVM("Adult", **deepcopy(base_params)).k_random()
