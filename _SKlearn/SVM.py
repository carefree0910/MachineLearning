import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from Util.Bases import ClassifierBase
from Util.Metas import SKCompatibleMeta

from sklearn.svm import SVC, LinearSVC


class SKSVM(SVC, ClassifierBase, metaclass=SKCompatibleMeta):
    pass


class SKLinearSVM(LinearSVC, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
