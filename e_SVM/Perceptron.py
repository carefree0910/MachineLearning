import numpy as np

from Util.Bases import ClassifierBase, KernelBase
from Util.Metas import ClassifierMeta, SubClassChangeNamesMeta
from Util.Timing import Timing
from Util.Util import DataUtil


class Perceptron(ClassifierBase, metaclass=ClassifierMeta):
    PerceptronTiming = Timing()

    def __init__(self):
        self._w = self._b = None

    @PerceptronTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, lr=1, epoch=10 ** 4):
        x, y = np.atleast_2d(x), np.array(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.array(sample_weight) * len(y)
        self._w = np.zeros(x.shape[1])
        self._b = 0
        for _ in range(epoch):
            y_pred = self.predict(x)
            _idx = np.argmax((y_pred != y) * sample_weight)
            if y_pred[_idx] == y[_idx]:
                return
            self._w += lr * y[_idx] * x[_idx] * sample_weight[_idx]
            self._b += lr * y[_idx] * sample_weight[_idx]

    @PerceptronTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False):
        rs = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(rs)
        return rs


class KernelPerceptron(KernelBase, metaclass=SubClassChangeNamesMeta):
    KernelPerceptronTiming = Timing()

    def __init__(self):
        KernelBase.__init__(self)
        self._fit_args, self._fit_args_names = [1], ["lr"]

    @KernelPerceptronTiming.timeit(level=1, prefix="[Core] ")
    def _update_w(self, idx):
        self._dw_cache = self._y[idx]
        self._w[idx] += self._dw_cache

    @KernelPerceptronTiming.timeit(level=1, prefix="[Core] ")
    def _update_b(self, idx):
        self._b_cache = self._b
        self._b += self._y[idx]

    @KernelPerceptronTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, sample_weight, lr):
        y_pred = np.sign(self._predict())
        # noinspection PyTypeChecker
        _idx = np.argmax((y_pred != self._y) * sample_weight)
        if y_pred[_idx] == self._y[_idx]:
            return True
        self._alpha[_idx] += lr
        self._update_w(_idx)
        self._update_b(_idx)
        self._update_pred_cache(_idx)

if __name__ == '__main__':
    # xs, ys = DataUtil.gen_two_clusters(center=5, dis=1, scale=2, one_hot=False)
    xs, ys = DataUtil.gen_spin(20, 4, 2, 2, one_hot=False)
    ys[ys == 0] = -1
    perceptron = KernelPerceptron()
    # perceptron.fit(xs, ys, kernel="rbf", epoch=10 ** 5)
    perceptron.fit(xs, ys, p=12, epoch=10 ** 5)
    perceptron.estimate(xs, ys)
    perceptron.visualize2d(xs, ys, dense=400)
    perceptron.show_timing_log()
