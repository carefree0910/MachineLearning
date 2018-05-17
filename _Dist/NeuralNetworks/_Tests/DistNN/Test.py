import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.g_DistNN.NN import DistAdvanced


base_params = {
    "data_info": {
        "data_folder": "../_Data",
        "file_type": "csv"
    },
    "model_param_settings": {
        "metric": "acc",
        "n_epoch": 1,
        "max_epoch": 2
    }
}

grid_params = {
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
dist = DistAdvanced("Adult", **base_params).grid_search(grid_params, "dict_first")
