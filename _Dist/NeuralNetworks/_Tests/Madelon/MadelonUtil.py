import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")

from _Dist.NeuralNetworks.NNUtil import Toolbox


def get_madelon():
    with open(os.path.join(root_path, "_Data", "madelon.txt"), "r") as file:
        data = np.array(Toolbox.get_data(file), np.float32)
    np.random.shuffle(data)
    train_set, test_set = data[:2000], data[2000:]
    x, y = train_set[..., :-1], train_set[..., -1]
    x_test, y_test = test_set[..., :-1], test_set[..., -1]
    return x, y, x_test, y_test
