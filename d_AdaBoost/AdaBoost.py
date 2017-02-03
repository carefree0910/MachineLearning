from math import log

from sklearn.tree import DecisionTreeClassifier
import sklearn.naive_bayes as nb

from b_NaiveBayes.Vectorized.MultinomialNB import MultinomialNB
from b_NaiveBayes.Vectorized.GaussianNB import GaussianNB
from c_CvDTree.Tree import *


class AdaBoost:

    _weak_clf = {
        "CvDTree": DecisionTreeClassifier,
        "NB": nb.GaussianNB,

        "MNB": MultinomialNB,
        "GNB": GaussianNB,
        "ID3": ID3Tree,
        "C45": C45Tree,
        "Cart": CartTree
    }

    """
        Naive implementation of AdaBoost
        Requirement: weak_clf should contain:
            1) 'fit'      method, which supports sample_weight
            2) 'predict'  method, which returns binary numpy array
    """

    def __init__(self):
        self._clf, self._clfs, self._clfs_weights = "", [], []
        self._sample_weight = None
        self._kwarg_cache = {}

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

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

    def fit(self, x, y, clf=None, epoch=10, eps=1e-12, *args, **kwargs):
        if clf is None or AdaBoost._weak_clf[clf] is None:
            clf = "CvDTree"
            kwargs = {"max_depth": 3}
        self._kwarg_cache = kwargs
        self._clf = clf
        self._sample_weight = np.ones(len(x)) / len(x)
        for _ in range(epoch):
            tmp_clf = AdaBoost._weak_clf[clf](*args, **kwargs)
            tmp_clf.fit(x, y, sample_weights=self._sample_weight)
            y_pred = tmp_clf.predict(x)
            em = min(max((y_pred != y).dot(self._sample_weight[:, None])[0], eps), 1 - eps)
            am = 0.5 * log(1 / em - 1)
            tmp_vec = self._sample_weight * np.exp(-am * y * y_pred)
            self._sample_weight = tmp_vec / np.sum(tmp_vec)
            self._clfs.append(deepcopy(tmp_clf))
            self._clfs_weights.append(am)

    def predict(self, x):
        x = np.array(x)
        rs = np.zeros(len(x))
        for clf, am in zip(self._clfs, self._clfs_weights):
            rs += am * clf.predict(x)
        return np.sign(rs)

    def estimate(self, x, y):
        y_pred = self.predict(x)
        print("Acc: {:8.6} %".format(100 * np.sum(y_pred == y) / len(y)))

    def visualize2d(self, x, y, dense=200):
        length = len(x)
        axis = np.array([[.0] * length, [.0] * length])
        for i, xx in enumerate(x):
            axis[0][i] = xx[0]
            axis[1][i] = xx[1]
        xs, ys = np.array(x), np.array(y, np.int8)
        ys[ys < 0] = 0

        print("=" * 30 + "\n" + self.title)
        decision_function = lambda _x: self.predict(_x)

        nx, ny, margin = dense, dense, 0.1
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_margin = max(abs(x_min), abs(x_max)) * margin
        y_margin = max(abs(y_min), abs(y_max)) * margin
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)

        t = time.time()
        z = decision_function(base_matrix).reshape((nx, ny))
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        per = 1 / 2
        colors = plt.cm.rainbow([i * per for i in range(2)])

        plt.figure()
        plt.title(self.title)
        plt.pcolormesh(xy_xf, xy_yf, z > 0, cmap=plt.cm.Paired)
        plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=[colors[y] for y in ys])
        plt.show()

        print("Done.")

    def visualize(self):
        try:
            for i, clf in enumerate(self._clfs):
                clf.visualize()
                _ = input("Press any key to continue...")
        except AttributeError:
            return
