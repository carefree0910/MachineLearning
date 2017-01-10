import numpy as np
import matplotlib.pyplot as plt

from Zhihu.NaiveSVM.Util import Timing

np.random.seed(142857)


class SVMConfig:
    default_c = 1
    default_p = 4
    default_sigma = 1


class SVM:
    SVMTiming = Timing()

    def __init__(self):
        self._c = None
        self._kernel = None
        self._x = self._y = None
        self._alpha = self._w = self._b = self._es = None

    def __str__(self):
        return "SVM"

    __repr__ = __str__

    # Kernel

    @staticmethod
    @SVMTiming.timeit(level=1, prefix="[Private StaticMethod] ")
    def _poly(x, y, p):
        if len(x.shape) == 1:
            return (np.sum(x * y) + 1) ** p
        return (np.sum(x * y, axis=1) + 1) ** p

    @staticmethod
    @SVMTiming.timeit(level=1, prefix="[Private StaticMethod] ")
    def _gaussian(x, y, sigma):
        if len(x.shape) == 1:
            # noinspection PyTypeChecker
            return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))
        # noinspection PyTypeChecker
        return np.exp(-np.sum((x - y) ** 2, axis=1) / (2 * sigma ** 2))

    # SMO

    @SVMTiming.timeit(level=1)
    def _pick_first(self, eps=10e-8):
        if self._w is None:
            self._update_w()
        con1 = self._alpha <= 0
        con2 = (0 < self._alpha) & (self._alpha < self._c)
        con3 = self._alpha >= self._c
        y_pred = self.predict()
        dis = self._y * y_pred - 1
        err1 = dis ** 2
        err2 = err1.copy()
        err3 = err1.copy()
        err1[~con1 | (dis >= 0)] = 0
        err2[~con2 | (dis == 0)] = 0
        err3[~con3 | (dis <= 0)] = 0
        err = err1 + err2 + err3
        idx = np.argmax(err)
        if err[idx] < eps:
            return
        return idx

    @SVMTiming.timeit(level=1)
    def _pick_second(self, idx1):
        idx = idx1
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
        x1, x2 = self._x[idx1], self._x[idx2]
        y1, y2 = self._y[idx1], self._y[idx2]
        e1, e2 = self._es[idx1], self._es[idx2]
        eta = self._kernel(x1, x1) + self._kernel(x2, x2) - 2 * self._kernel(x1, x2)
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
        x1, x2 = self._x[idx1], self._x[idx2]
        y1, y2 = self._y[idx1], self._y[idx2]
        b1 = -e1 - y1 * self._kernel(x1, x1) * (a1_new - a1_old) - y2 * self._kernel(x2, x1) * (a2_new - a2_old)
        b2 = -e2 - y1 * self._kernel(x1, x2) * (a1_new - a1_old) - y2 * self._kernel(x2, x2) * (a2_new - a2_old)
        self._b += (b1 + b2) / 2

    @SVMTiming.timeit(level=1)
    def _update_es(self):
        self._es = self.predict() - self._y

    # API

    def feed_timing(self, timing):
        self.SVMTiming = timing

    @SVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, kernel="poly", **kwargs):
        if kernel == "poly":
            self._kernel = lambda _x, _y: SVM._poly(_x, _y, kwargs.get("p", SVMConfig.default_p))
        elif kernel == "gaussian":
            self._kernel = lambda _x, _y: SVM._gaussian(_x, _y, kwargs.get("sigma", SVMConfig.default_sigma))
        else:
            raise NotImplementedError("Kernel '{}' not implemented".format(kernel))
        self._c = kwargs.get("c", SVMConfig.default_c)
        self._x, self._y = np.atleast_2d(x), np.array(y)
        self._alpha = np.zeros(len(x))
        self._es = -self._y
        self._b = 0
        while True:
            idx1 = self._pick_first()
            if idx1 is None:
                return
            idx2 = self._pick_second(idx1)
            self._update_alpha(idx1, idx2)

    def predict(self, x=None):
        x = self._x if x is None else np.atleast_2d(x)
        x = np.array([self._kernel(x, xi) for xi in self._x])
        return np.sum(self._w * x.T, axis=1) + self._b

    def predict_classes(self, x=None):
        y_pred = self.predict(x)
        y1 = y_pred > 0
        y_pred[y1], y_pred[~y1] = 1, -1
        return y_pred

    def evaluate(self, x=None, y=None):
        if x is None or y is None:
            x, y = self._x, self._y
        y_pred = self.predict_classes(x)
        print("Acc: {:8.6} %".format(np.sum(y_pred == y) * 100 / len(y)))

    def visualize_2d(self, x=None, y=None, plot_scale=2, plot_precision=0.01):
        if x is None or y is None:
            x, y = self._x, self._y
        plot_num = int(1 / plot_precision)
        xf = np.linspace(np.min(x) * plot_scale, np.max(x) * plot_scale, plot_num)
        yf = np.linspace(np.min(x) * plot_scale, np.max(x) * plot_scale, plot_num)
        input_x, input_y = np.meshgrid(xf, yf)
        input_xs = np.c_[input_x.ravel(), input_y.ravel()]
        output_ys_2d = self.predict_classes(input_xs).reshape(len(xf), len(yf))
        plt.contourf(input_x, input_y, output_ys_2d, cmap=plt.cm.Spectral)
        plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.axis("off")
        plt.show()
