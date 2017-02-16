from Util.Bases import ClassifierBase
from Util.Metas import SklearnCompatibleMeta

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


class SKAdaBoost(AdaBoostClassifier, ClassifierBase, metaclass=SklearnCompatibleMeta):
    pass


class SKRandomForest(RandomForestClassifier, ClassifierBase, metaclass=SklearnCompatibleMeta):
    pass
