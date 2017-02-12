from math import log

from b_NaiveBayes.Vectorized.MultinomialNB import MultinomialNB
from b_NaiveBayes.Vectorized.GaussianNB import GaussianNB
from c_CvDTree.Tree import *

from _SKlearn.NaiveBayes import *
from _SKlearn.Tree import *


class AdaBoost(ClassifierBase, metaclass=ClassifierMeta):
    AdaBoostTiming = Timing()
    _weak_clf = {
        "SKMNB": SKMultinomialNB,
        "SKGNB": SKGaussianNB,
        "SKTree": SKTree,

        "MNB": MultinomialNB,
        "GNB": GaussianNB,
        "ID3": ID3Tree,
        "C45": C45Tree,
        "Cart": CartTree
    }

    def __init__(self):
        self._clf, self._clfs, self._clfs_weights = "", [], []
        self._kwarg_cache = {}

    @property
    def title(self):
        rs = "Classifier: {}; Num: {}".format(self._clf, len(self._clfs))
        rs += " " + self.params
        return rs

    @property
    def params(self):
        rs = ""
        if self._kwarg_cache:
            tmp_rs = []
            for key, value in self._kwarg_cache.items():
                tmp_rs.append("{}: {}".format(key, value))
            rs += "( " + "; ".join(tmp_rs) + " )"
        return rs

    @AdaBoostTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, clf=None, epoch=10, sample_weight=None, eps=1e-12, *args, **kwargs):
        if clf is None or AdaBoost._weak_clf[clf] is None:
            clf = "CvDTree"
            kwargs = {"max_depth": 3}
        self._kwarg_cache = kwargs
        self._clf = clf
        if sample_weight is None:
            sample_weight = np.ones(len(x)) / len(x)
        else:
            sample_weight = np.array(sample_weight)
        for _ in range(epoch):
            tmp_clf = AdaBoost._weak_clf[clf](*args, **kwargs)
            tmp_clf.fit(x, y, sample_weight=sample_weight)
            y_pred = tmp_clf.predict(x)
            em = min(max((y_pred != y).astype(np.int8).dot(sample_weight[:, None])[0], eps), 1 - eps)
            am = 0.5 * log(1 / em - 1)
            tmp_vec = sample_weight * np.exp(-am * y * y_pred)
            sample_weight = tmp_vec / np.sum(tmp_vec)
            self._clfs.append(deepcopy(tmp_clf))
            self._clfs_weights.append(am)

    @AdaBoostTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x):
        x = np.array(x)
        rs = np.zeros(len(x))
        for clf, am in zip(self._clfs, self._clfs_weights):
            rs += am * clf.predict(x)
        return np.sign(rs)
