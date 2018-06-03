import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.DistBase import AutoBase, AutoMeta
from _Dist.NeuralNetworks.e_AdvancedNN.DistNN import Basic, Advanced


class AutoBasic(AutoBase, Basic, metaclass=AutoMeta):
    pass


class AutoAdvanced(AutoBase, Advanced, metaclass=AutoMeta):
    pass
