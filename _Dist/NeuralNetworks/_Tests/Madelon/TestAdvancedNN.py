import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.e_AdvancedNN.NN import Advanced
from _Dist.NeuralNetworks._Tests.TestUtil import draw_acc
from _Dist.NeuralNetworks._Tests.Madelon.MadelonUtil import get_madelon

x, y, x_test, y_test = get_madelon()


def normalize(arr):
    arr = arr.copy()
    arr -= arr.mean(0)
    arr /= arr.std(0)
    return arr


def block_evaluate():
    advanced_nn = Advanced(**advanced_params).fit(x, y, x_test, y_test, snapshot_ratio=0)
    print("BasicNN              ", end="")
    basic.evaluate(x, y, None, None, x_test, y_test)
    print("AdvancedNN           ", end="")
    advanced_nn.evaluate(x, y, None, None, x_test, y_test)
    return advanced_nn


x, x_test = normalize(x), normalize(x_test)

ylim = (0.5, 1.05)
base_params = {
    "model_param_settings": {"n_epoch": 200, "metric": "acc"},
    "model_structure_settings": {"hidden_units": [152, 153]}
}
basic = Basic(**base_params).fit(x, y, x_test, y_test, snapshot_ratio=0)

advanced_params = {"data_info": {
    "numerical_idx": [True] * 500 + [False], "categorical_columns": []
}}
advanced_params.update(base_params)
advanced_params["model_param_settings"]["keep_prob"] = 0.5
advanced_params["model_param_settings"]["use_batch_norm"] = False
advanced_params["model_structure_settings"]["use_pruner"] = False
advanced_params["model_structure_settings"]["use_wide_network"] = False
advanced = block_evaluate()
draw_acc(basic, advanced, ylim=ylim)

advanced_params["model_param_settings"]["keep_prob"] = 0.25
advanced = block_evaluate()
draw_acc(basic, advanced, ylim=ylim)

advanced_params["model_param_settings"]["keep_prob"] = 0.1
advanced_params["model_param_settings"]["n_epoch"] = 600
advanced = block_evaluate()
draw_acc(basic, advanced, ylim=ylim)

advanced_params["model_param_settings"]["n_epoch"] = 200
advanced_params["model_param_settings"]["keep_prob"] = 0.25
advanced_params["model_param_settings"]["use_batch_norm"] = True
advanced = block_evaluate()
draw_acc(basic, advanced, ylim=ylim)

advanced_params["model_structure_settings"]["use_pruner"] = True
advanced_params["model_structure_settings"]["use_wide_network"] = True
advanced = block_evaluate()
draw_acc(basic, advanced, ylim=ylim)
