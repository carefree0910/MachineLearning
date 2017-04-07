from Util.Bases import ClassifierBase
from Util.Metas import SKCompatibleMeta

from sklearn.svm import SVC, LinearSVC


class SKSVM(SVC, ClassifierBase, metaclass=SKCompatibleMeta):
    pass


class SKLinearSVM(LinearSVC, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
