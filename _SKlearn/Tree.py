from Util.Bases import ClassifierBase
from Util.Metas import SKCompatibleMeta

from sklearn.tree import DecisionTreeClassifier


class SKTree(DecisionTreeClassifier, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
