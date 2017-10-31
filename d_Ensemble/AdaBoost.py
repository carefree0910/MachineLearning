import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from math import log

from b_NaiveBayes.Vectorized.MultinomialNB import MultinomialNB
from b_NaiveBayes.Vectorized.GaussianNB import GaussianNB
from c_CvDTree.Tree import *
from d_Ensemble.RandomForest import RandomForest
from e_SVM.Perceptron import Perceptron
from e_SVM.KP import KP
from e_SVM.SVM import SVM

from Util.ProgressBar import ProgressBar

from _SKlearn.NaiveBayes import *
from _SKlearn.Tree import SKTree
from _SKlearn.SVM import SKSVM


def boost_task(args):
    x, clfs, n_cores = args
    return [clf.predict(x, n_cores=n_cores) for clf in clfs]


class AdaBoost(ClassifierBase):
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
        "KP": KP,
        "SVM": SVM
    }

    def __init__(self, **kwargs):
        super(AdaBoost, self).__init__(**kwargs)
        self._clf, self._clfs, self._clfs_weights = "", [], []
        self._kwarg_cache = {}

        self._params["clf"] = kwargs.get("clf", None)
        self._params["epoch"] = kwargs.get("epoch", 10)
        self._params["eps"] = kwargs.get("eps", 1e-12)

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
    def fit(self, x, y, sample_weight=None, clf=None, epoch=None, eps=None, **kwargs):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if clf is None:
            clf = self._params["clf"]
        if epoch is None:
            epoch = self._params["epoch"]
        if eps is None:
            eps = self._params["eps"]
        x, y = np.atleast_2d(x), np.asarray(y)
        if clf is None:
            clf = "Cart"
            kwargs = {"max_depth": 1}
        self._kwarg_cache = kwargs
        self._clf = clf
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
        else:
            sample_weight = np.asarray(sample_weight)
        bar = ProgressBar(max_value=epoch, name="AdaBoost")
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
            bar.update()
        self._clfs_weights = np.array(self._clfs_weights, dtype=np.float32)

    @AdaBoostTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, bound=None, **kwargs):
        x = np.atleast_2d(x)
        if bound is None:
            clfs, clfs_weights = self._clfs, self._clfs_weights
        else:
            clfs, clfs_weights = self._clfs[:bound], self._clfs_weights[:bound]
        matrix = self._multi_clf(x, clfs, boost_task, kwargs)
        matrix *= clfs_weights
        rs = np.sum(matrix, axis=1)
        del matrix
        if not get_raw_results:
            return np.sign(rs)
        return rs
