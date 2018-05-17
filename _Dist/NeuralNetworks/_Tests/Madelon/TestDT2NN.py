import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.d_Traditional2NN.Toolbox import DT2NN
from _Dist.NeuralNetworks._Tests.Madelon.MadelonUtil import get_madelon

x, y, x_test, y_test = get_madelon()

basic = Basic(model_structure_settings={"hidden_units": [152, 153]}).fit(x, y, x_test, y_test)
dt2dnn = DT2NN(model_param_settings={"activations": ["sign", "softmax"]}).fit(x, y, x_test, y_test)
