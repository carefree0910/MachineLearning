import numpy as np

from Util.Timing import Timing
from Util.Bases import ClassifierBase
from Util.Metas import ClassifierMeta


class SVMConfig:
    default_c = 1
    default_p = 4


class SVM(ClassifierBase, metaclass=ClassifierMeta):
    SVMTiming = Timing()

    def __init__(self):
        self._c = None
        self._kernel = self._kernel_name = self._kernel_param = None
        self._x = self._y = self._gram = None
        self._alpha = self._w = self._b = self._es = None

    @property
    def title(self):
        return "{} {} ({})".format(self._kernel_name, self, self._kernel_param)

    # Kernel

    @staticmethod
    @SVMTiming.timeit(level=1, prefix="[Core] ")
    def _poly(x, y, p):
        return (x.dot(y.T) + 1) ** p

    # noinspection PyTypeChecker
    @staticmethod
    @SVMTiming.timeit(level=1, prefix="[Core] ")
    def _rbf(x, y, gamma):
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))

    @SVMTiming.timeit(level=1, prefix="[Core] ")
    def _predict(self):
        return np.sum(self._w * self._gram, axis=1) + self._b

    # SMO

    @SVMTiming.timeit(level=1)
    def _pick_first(self, tol):
        con1 = self._alpha > 0
        con2 = self._alpha < self._c
        y_pred = self._predict()
        err1 = self._y * y_pred - 1
        err2 = err1.copy()
        err3 = err1.copy()
        err1[con1 | (err1 >= 0)] = 0
        err2[(~con1 | ~con2) | (err2 == 0)] = 0
        err3[con2 | (err3 <= 0)] = 0
        err = err1 ** 2 + err2 ** 2 + err3 ** 2
        idx = np.argmax(err)
        if err[idx] < tol:
            return
        return idx

    @SVMTiming.timeit(level=1)
    def _pick_second(self, idx1):
        idx = np.random.randint(len(self._y))
        while idx == idx1:
            idx = np.random.randint(len(self._y))
        return idx

    @SVMTiming.timeit(level=2)
    def _get_lower_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return max(0., self._alpha[idx2] - self._alpha[idx1])
        return max(0., self._alpha[idx2] + self._alpha[idx1] - self._c)

    @SVMTiming.timeit(level=2)
    def _get_upper_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return min(self._c, self._c + self._alpha[idx2] - self._alpha[idx1])
        return min(self._c, self._alpha[idx2] + self._alpha[idx1])

    @SVMTiming.timeit(level=1)
    def _update_alpha(self, idx1, idx2):
        l, h = self._get_lower_bound(idx1, idx2), self._get_upper_bound(idx1, idx2)
        y1, y2 = self._y[idx1], self._y[idx2]
        e1, e2 = self._es[idx1], self._es[idx2]
        eta = self._gram[idx1][idx1] + self._gram[idx2][idx2] - 2 * self._gram[idx1][idx2]
        a2_new = self._alpha[idx2] + (y2 * (e1 - e2)) / eta
        if a2_new > h:
            a2_new = h
        elif a2_new < l:
            a2_new = l
        a1_old, a2_old = self._alpha[idx1], self._alpha[idx2]
        a1_new = self._alpha[idx1] + y1 * y2 * (self._alpha[idx2] - a2_new)
        self._alpha[idx1] = a1_new
        self._alpha[idx2] = a2_new
        self._update_w()
        self._update_b(idx1, idx2, e1, e2, a1_old, a2_old, a1_new, a2_new)
        self._update_es()

    @SVMTiming.timeit(level=1)
    def _update_w(self):
        self._w = self._alpha * self._y

    @SVMTiming.timeit(level=1)
    def _update_b(self, idx1, idx2, e1, e2, a1_old, a2_old, a1_new, a2_new):
        gram_12 = self._gram[idx1][idx2]
        y1, y2 = self._y[idx1], self._y[idx2]
        b1 = -e1 - y1 * self._gram[idx1][idx1] * (a1_new - a1_old) - y2 * gram_12 * (a2_new - a2_old)
        b2 = -e2 - y1 * gram_12 * (a1_new - a1_old) - y2 * self._gram[idx2][idx2] * (a2_new - a2_old)
        self._b += (b1 + b2) / 2

    @SVMTiming.timeit(level=1)
    def _update_es(self):
        self._es = self._predict() - self._y

    # API

    @SVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, kernel="poly", epoch=10 ** 4, tol=1e-8, **kwargs):
        self._c = kwargs.get("c", SVMConfig.default_c)
        self._x, self._y = np.atleast_2d(x), np.array(y)
        if kernel == "poly":
            _p = kwargs.get("p", SVMConfig.default_p)
            self._kernel_name = "Polynomial"
            self._kernel_param = "degree = {}".format(_p)
            self._kernel = lambda _x, _y: SVM._poly(_x, _y, _p)
        elif kernel == "rbf":
            _gamma = kwargs.get("gamma", 1 / self._x.shape[1])
            self._kernel_name = "RBF"
            self._kernel_param = "gamma = {}".format(_gamma)
            self._kernel = lambda _x, _y: SVM._rbf(_x, _y, _gamma)
        else:
            raise NotImplementedError("Kernel '{}' has not defined".format(kernel))
        self._gram = self._kernel(self._x, self._x)
        self._alpha, self._w = np.zeros(len(x)), np.zeros(len(x))
        self._es = -self._y
        self._b = 0
        for _ in range(epoch):
            idx1 = self._pick_first(tol)
            if idx1 is None:
                return
            idx2 = self._pick_second(idx1)
            self._update_alpha(idx1, idx2)

    @SVMTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False):
        x = self._kernel(np.atleast_2d(x), self._x)
        y_pred = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(y_pred)
        return y_pred
