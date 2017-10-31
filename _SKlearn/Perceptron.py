from Util.Bases import ClassifierBase
from Util.Metas import SKCompatibleMeta

from sklearn.linear_model.perceptron import Perceptron


class SKPerceptron(Perceptron, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
