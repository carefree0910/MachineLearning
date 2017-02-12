from Util.Bases import ClassifierBase
from Util.Metas import SklearnCompatibleMeta

import sklearn.naive_bayes as nb


class SKMultinomialNB(nb.MultinomialNB, ClassifierBase, metaclass=SklearnCompatibleMeta):
    pass


class SKGaussianNB(nb.GaussianNB, ClassifierBase, metaclass=SklearnCompatibleMeta):
    pass
