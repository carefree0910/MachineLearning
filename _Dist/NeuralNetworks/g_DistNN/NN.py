import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.Base import DistMixin
from _Dist.NeuralNetworks.f_AutoNN.NN import AutoBasic, AutoAdvanced


class DistBasic(AutoBasic, DistMixin):
    pass


class DistAdvanced(AutoAdvanced, DistMixin):
    pass
