import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from Util.Bases import ClassifierBase
from Util.Metas import SKCompatibleMeta

import sklearn.naive_bayes as nb


class SKMultinomialNB(nb.MultinomialNB, ClassifierBase, metaclass=SKCompatibleMeta):
    pass


class SKGaussianNB(nb.GaussianNB, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
