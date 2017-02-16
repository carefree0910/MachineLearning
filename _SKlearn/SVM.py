from Util.Bases import ClassifierBase
from Util.Metas import SklearnCompatibleMeta

from sklearn.svm import SVC


class SklearnSVM(SVC, ClassifierBase, metaclass=SklearnCompatibleMeta):
    pass
