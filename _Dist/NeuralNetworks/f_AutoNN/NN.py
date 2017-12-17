import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.e_AdvancedNN.NN import Advanced
from _Dist.NeuralNetworks.Base import AutoBase, AutoMeta


class AutoBasic(AutoBase, Basic, metaclass=AutoMeta):
    pass


class AutoAdvanced(AutoBase, Advanced, metaclass=AutoMeta):
    pass
