from Util.Bases import ClassifierBase
from Util.Metas import SKCompatibleMeta

from sklearn.svm import SVC


class SKSVM(SVC, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
