import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks._Tests.TestUtil import draw_acc
from _Dist.NeuralNetworks.f_AutoNN.NN import AutoBasic, AutoAdvanced
from _Dist.NeuralNetworks._Tests.Madelon.MadelonUtil import get_madelon

x, y, x_test, y_test = get_madelon()

base_params = {
    "name": "Madelon",
    "model_param_settings": {"n_epoch": 200, "metric": "acc"},
    "model_structure_settings": {"hidden_units": [152, 153]}
}
basic = AutoBasic(**base_params).fit(x, y, x_test, y_test, snapshot_ratio=0)
advanced = AutoAdvanced(**base_params).fit(x, y, x_test, y_test, snapshot_ratio=0)
draw_acc(basic, advanced, ylim=(0.5, 1.05))
