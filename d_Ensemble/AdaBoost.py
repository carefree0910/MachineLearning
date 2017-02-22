from math import log

from b_NaiveBayes.Vectorized.MultinomialNB import MultinomialNB
from b_NaiveBayes.Vectorized.GaussianNB import GaussianNB
from c_CvDTree.Tree import *
from d_Ensemble.RandomForest import RandomForest
from e_SVM.Perceptron import Perceptron
from e_SVM.KP import KernelPerceptron
from e_SVM.SVM import SVM

from _SKlearn.NaiveBayes import *
from _SKlearn.Tree import SKTree
from _SKlearn.SVM import SKSVM


class AdaBoost(ClassifierBase, metaclass=ClassifierMeta):
    AdaBoostTiming = Timing()
    _weak_clf = {
        "SKMNB": SKMultinomialNB,
        "SKGNB": SKGaussianNB,
        "SKTree": SKTree,
        "SKSVM": SKSVM,

        "MNB": MultinomialNB,
        "GNB": GaussianNB,
        "ID3": ID3Tree,
        "C45": C45Tree,
        "Cart": CartTree,
        "RF": RandomForest,
        "Perceptron": Perceptron,
        "KP": KernelPerceptron,
        "SVM": SVM
    }

    def __init__(self):
        self._clf, self._clfs, self._clfs_weights = "", [], []
        self._kwarg_cache = {}

    @property
    def params(self):
        rs = ""
        if self._kwarg_cache:
            tmp_rs = []
            for key, value in self._kwarg_cache.items():
                tmp_rs.append("{}: {}".format(key, value))
            rs += "( " + "; ".join(tmp_rs) + " )"
        return rs

    @property
    def title(self):
        rs = "Classifier: {}; Num: {}".format(self._clf, len(self._clfs))
        rs += " " + self.params
        return rs

    @AdaBoostTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, clf=None, epoch=10, eps=1e-12, **kwargs):
        x, y = np.atleast_2d(x), np.array(y)
        if clf is None:
            clf = "Cart"
            kwargs = {"max_depth": 1}
        self._kwarg_cache = kwargs
        self._clf = clf
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
        else:
            sample_weight = np.array(sample_weight)
        for _ in range(epoch):
            tmp_clf = AdaBoost._weak_clf[clf](**kwargs)
            tmp_clf.fit(x, y, sample_weight=sample_weight)
            y_pred = tmp_clf.predict(x)
            em = min(max((y_pred != y).astype(np.int8).dot(sample_weight[..., None])[0], eps), 1 - eps)
            am = 0.5 * log(1 / em - 1)
            sample_weight *= np.exp(-am * y * y_pred)
            sample_weight /= np.sum(sample_weight)
            self._clfs.append(deepcopy(tmp_clf))
            self._clfs_weights.append(am)

    @AdaBoostTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, bound=None, get_raw_results=False):
        x = np.atleast_2d(x)
        rs = np.zeros(len(x))
        if bound is None:
            _clfs, _clfs_weights = self._clfs, self._clfs_weights
        else:
            _clfs, _clfs_weights = self._clfs[:bound], self._clfs_weights[:bound]
        for clf, am in zip(_clfs, _clfs_weights):
            rs += am * clf.predict(x)
        if not get_raw_results:
            return np.sign(rs)
        return rs
