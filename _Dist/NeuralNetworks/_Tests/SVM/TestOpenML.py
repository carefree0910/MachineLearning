import os
import sys
root_path = os.path.abspath("./")
if root_path not in sys.path:
    sys.path.append(root_path)

import copy
import numpy as np
import tensorflow as tf

from openml import datasets

from _Dist.NeuralNetworks.b_TraditionalML.SVM import DistLinearSVM

GPU_ID = None
K_RANDOM = 9
IDS = [
    38, 46, 179,
    184, 389, 554,
    772, 917, 1049,
    1111, 1120, 1128,
    293,
]


def swap(arr, i1, i2):
    arr[..., i1], arr[..., i2] = arr[..., i2], arr[..., i1].copy()


def download_data():
    data_folder = "_Data"
    idx_folder = os.path.join(data_folder, "idx")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    if not os.path.isdir(idx_folder):
        os.makedirs(idx_folder)
    for idx in IDS:
        print("Downloading {}".format(idx))
        data_file = os.path.join(data_folder, "{}.txt".format(idx))
        idx_file = os.path.join(idx_folder, "{}.npy".format(idx))
        if os.path.isfile(data_file) and os.path.isfile(idx_file):
            continue
        dataset = datasets.get_dataset(idx)
        data, categorical_idx, names = dataset.get_data(
            return_categorical_indicator=True,
            return_attribute_names=True
        )
        data = data.toarray() if not isinstance(data, np.ndarray) else data
        target_idx = names.index(dataset.default_target_attribute)
        numerical_idx = ~np.array(categorical_idx)
        swap(numerical_idx, target_idx, -1)
        swap(data, target_idx, -1)
        with open(data_file, "w") as file:
            file.write("\n".join([" ".join(map(lambda n: str(n), line)) for line in data]))
        np.save(idx_file, numerical_idx)


def main():
    base_params = {
        "data_info": {},
        "model_param_settings": {},
        # "model_param_settings": {"n_epoch": 1, "max_epoch": 1},
        # "model_structure_settings": {"use_wide_network": False, "use_pruner": False}
    }
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if GPU_ID is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    base_params["model_param_settings"]["sess_config"] = config
    for idx in IDS:
        numerical_idx = np.load("_Data/idx/{}.npy".format(idx))
        local_params = copy.deepcopy(base_params)
        local_params["name"] = str(idx)
        local_params["data_info"]["numerical_idx"] = numerical_idx
        DistLinearSVM(**copy.deepcopy(local_params)).k_random(K_RANDOM, cv_rate=0.1, test_rate=0.1)


if __name__ == '__main__':
    download_data()
    main()
