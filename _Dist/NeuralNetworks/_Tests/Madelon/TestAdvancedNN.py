import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")

from _Dist.NeuralNetworks.NNUtil import Toolbox
from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.e_AdvancedNN.NN import Advanced

with open(os.path.join(root_path, "_Data", "madelon.txt"), "r") as file:
    data = np.array(Toolbox.get_data(file), np.float32)
np.random.shuffle(data)
train_set, test_set = data[:2000], data[2000:]
x, y = train_set[..., :-1], train_set[..., -1]
x_test, y_test = test_set[..., :-1], test_set[..., -1]


def normalize(arr):
    arr = arr.copy()
    arr -= arr.mean(0)
    arr /= arr.std(0)
    return arr


def draw_acc(*models, ylim=(0.5, 1.05), draw_train=True):
    plt.figure()
    for nn in models:
        name = str(nn)
        el, tl = nn.log["train_acc"], nn.log["test_acc"]
        ee_base = np.arange(len(el))
        cse_base = np.linspace(0, len(el) - 1, len(tl))
        if draw_train:
            plt.plot(ee_base, el, label="Train acc ({})".format(name))
        plt.plot(cse_base, tl, label="Test acc ({})".format(name))
    plt.ylim(*ylim)
    plt.legend(prop={'size': 14})
    plt.show()


def block_evaluate():
    advanced_nn = Advanced(**advanced_params).fit(x, y, x_test, y_test, snapshot_ratio=0)
    print("BasicNN              ", end="")
    basic.evaluate(x, y, None, None, x_test, y_test)
    print("AdvancedNN           ", end="")
    advanced_nn.evaluate(x, y, None, None, x_test, y_test)
    return advanced_nn


x, x_test = normalize(x), normalize(x_test)

base_params = {
    "model_param_settings": {"n_epoch": 200},
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
draw_acc(basic, advanced)

advanced_params["model_param_settings"]["keep_prob"] = 0.25
advanced = block_evaluate()
draw_acc(basic, advanced)

advanced_params["model_param_settings"]["keep_prob"] = 0.1
advanced_params["model_param_settings"]["n_epoch"] = 600
advanced = block_evaluate()
draw_acc(basic, advanced)

advanced_params["model_param_settings"]["n_epoch"] = 200
advanced_params["model_param_settings"]["keep_prob"] = 0.25
advanced_params["model_param_settings"]["use_batch_norm"] = True
advanced = block_evaluate()
draw_acc(basic, advanced)

advanced_params["model_structure_settings"]["use_pruner"] = True
advanced_params["model_structure_settings"]["use_wide_network"] = True
advanced = block_evaluate()
draw_acc(basic, advanced)
