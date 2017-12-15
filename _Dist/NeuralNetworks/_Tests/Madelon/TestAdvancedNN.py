import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np

from _Dist.NeuralNetworks.NNUtil import Toolbox
from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.e_AdvancedNN.NN import Advanced

with open(os.path.join(root_path, "_Data", "madelon.txt"), "r") as file:
    data = np.array(Toolbox.get_data(file), np.float32)
train_set, test_set = data[:2000], data[2000:]
x, y = train_set[..., :-1], train_set[..., -1]
x_test, y_test = test_set[..., :-1], test_set[..., -1]


def normalize(arr):
    arr = arr.copy()
    arr -= arr.mean(0)
    arr /= arr.std(0)
    return arr

x, x_test = normalize(x), normalize(x_test)

base_params = {
    "model_param_settings": {"n_epoch": 40},
    "model_structure_settings": {"hidden_units": [256, 256]}
}
basic = Basic(**base_params).fit(x, y, x_test, y_test)

advanced_params = {"data_info": {
    "numerical_idx": [True] * 500 + [False], "categorical_columns": []
}}
advanced_params.update(base_params)
advanced_params["model_structure_settings"].update({
    "use_wide_network": False, "use_pruner": False
})
advanced_params["model_param_settings"].update({
    "keep_prob": 0.5, "use_batch_norm": False
})
advanced = Advanced(**advanced_params).fit(x, y, x_test, y_test)
