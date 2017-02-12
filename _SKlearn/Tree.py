from Util.Bases import ClassifierBase
from Util.Metas import SklearnCompatibleMeta

from sklearn.tree import DecisionTreeClassifier


class SKTree(DecisionTreeClassifier, ClassifierBase, metaclass=SklearnCompatibleMeta):
    pass
