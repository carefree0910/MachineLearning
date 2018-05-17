import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt

from Util.Util import DataUtil
from _Dist.NeuralNetworks._Tests.Pruner.Advanced import Advanced

# (x, y), (x_test, y_test), *_ = DataUtil.get_dataset("mnist", "_Data/mnist.txt", n_train=1600, quantized=True)
(x, y), (x_test, y_test) = DataUtil.gen_noisy_linear(n_dim=100, n_valid=5, one_hot=False)

data_info = {
    "numerical_idx": [True] * 100 + [False],
    "categorical_columns": []
}

# nn = Advanced(
#     "NoisyLinear",
#     data_info=data_info,
#     model_param_settings={
#         "n_epoch": 40
#     },
#     model_structure_settings={
#         "use_wide_network": False,
#         "use_pruner": True,
#         "pruner_params": {
#             "prune_method": "surgery",
#             # "alpha": 1e-8,
#             # "beta": 1e8
#         },
#         "hidden_units": [512]
#     }
# ).fit(x, y, x_test, y_test, snapshot_ratio=0)
nn = Advanced("NoisyLinear", data_info=data_info).fit(x, y, x_test, y_test, snapshot_ratio=0)


def normalize(arr):
    arr = np.array(arr, np.float32)
    arr -= arr.mean()
    arr /= arr.std()
    return arr


plt.figure()
pruned_ratio = normalize(nn.log["pruned_ratio"])
recovered_ratio = normalize(nn.log["recovered_ratio"])
running_recovered = nn.log["running_recovered"]
w_abs_mean = nn.log["w_abs_mean"]
plt.plot(np.arange(len(pruned_ratio)), pruned_ratio, label="pruned_ratio")
plt.plot(np.arange(len(recovered_ratio)), recovered_ratio, label="recovered_ratio")
plt.plot(np.arange(len(w_abs_mean)), normalize(w_abs_mean), label="w_abs_mean")
plt.legend(loc=4, prop={'size': 14})
plt.show()
plt.figure()
plt.plot(np.arange(len(running_recovered)), running_recovered, label="running_recovered")
plt.legend(prop={'size': 14})
plt.show()
plt.figure()
w_abs_mean = nn.log["w_abs_mean"]
pruned_w_abs_mean = nn.log["pruned_w_abs_mean"]
org_survived_w_abs_mean = nn.log["org_w_abs_mean"]
org_pruned_w_abs_mean = nn.log["org_pruned_w_abs_mean"]
plt.plot(np.arange(len(w_abs_mean)), w_abs_mean, label="w_abs_mean")
plt.plot(np.arange(len(pruned_w_abs_mean)), pruned_w_abs_mean, label="pruned_w_abs_mean")
plt.plot(np.arange(len(org_pruned_w_abs_mean)), org_pruned_w_abs_mean, label="org_pruned_w_abs_mean")
plt.plot(np.arange(len(org_survived_w_abs_mean)), org_survived_w_abs_mean, label="org_survived_w_abs_mean")
plt.legend(prop={'size': 14})
plt.show()
plt.figure()
pruned_w_abs_residual = nn.log["pruned_w_abs_residual"]
plt.plot(np.arange(len(pruned_w_abs_residual)), pruned_w_abs_residual, label="pruned_w_abs_residual")
plt.legend(prop={'size': 14})
plt.show()
plt.figure()
target_ratio = nn.log["target_ratio"]
plt.plot(np.arange(len(target_ratio)), target_ratio, label="target_ratio")
plt.legend(prop={'size': 14})
plt.show()
plt.figure()
ptr = nn.log["pruned_target_ratio"]
plt.plot(np.arange(len(ptr)), ptr, label="pruned_target_ratio")
plt.legend(prop={'size': 14})
plt.show()
plt.figure()
rwwamr = np.array(nn.log["rwwamr"])
rwowamr = np.array(nn.log["rwowamr"])
rwnwamr = np.array(nn.log["rwnwamr"])
plt.plot(np.arange(len(rwwamr)), rwwamr, label="rwwamr")
plt.plot(np.arange(len(rwowamr)), rwowamr, label="rwowamr")
plt.plot(np.arange(len(rwnwamr)), rwnwamr, label="rwnwamr")
plt.legend(prop={'size': 14})
plt.show()
plt.figure()
accs = nn.log["acc"]
pwamr = np.array(nn.log["pruned_w_abs_mean"]) / np.array(nn.log["w_abs_mean"])
plt.plot(np.arange(len(pwamr)), pwamr, label="pruned_w_abs_mean_ratio")
plt.plot(np.arange(len(accs)), accs, label="acc")
plt.legend()
plt.show()
plt.figure()
pwamr = np.array(nn.log["pruned_w_abs_mean"]) / np.array(nn.log["w_abs_mean"])
plt.plot(np.arange(len(pwamr)), pwamr, label="pruned_w_abs_mean_ratio")
plt.legend(prop={'size': 14})
plt.show()
