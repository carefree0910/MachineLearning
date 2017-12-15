import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np

from _Dist.NeuralNetworks.NNUtil import Toolbox
from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.d_Traditional2NN.Toolbox import DT2NN

with open(os.path.join(root_path, "_Data", "madelon.txt"), "r") as file:
    data = np.array(Toolbox.get_data(file), np.float32)
np.random.shuffle(data)
train_set, test_set = data[:2000], data[2000:]
x, y = train_set[..., :-1], train_set[..., -1]
x_test, y_test = test_set[..., :-1], test_set[..., -1]

basic = Basic(model_structure_settings={"hidden_units": [152, 153]}).fit(x, y, x_test, y_test)
dt2dnn = DT2NN(model_param_settings={"activations": ["sign", "softmax"]}).fit(x, y, x_test, y_test)
