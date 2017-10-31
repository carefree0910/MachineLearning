import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from Util.Bases import ClassifierBase
from Util.Metas import SKCompatibleMeta

from sklearn.linear_model.perceptron import Perceptron


class SKPerceptron(Perceptron, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
