import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.e_AdvancedNN.NN import Advanced
from _Dist.NeuralNetworks.Base import AutoMixin, AutoMeta


class AutoBasic(AutoMixin, Basic, metaclass=AutoMeta):
    pass


class AutoAdvanced(AutoMixin, Advanced, metaclass=AutoMeta):
    pass
