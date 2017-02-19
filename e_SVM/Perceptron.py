import numpy as np

from Util.Bases import ClassifierBase
from Util.Metas import ClassifierMeta
from Util.Timing import Timing
from Util.Util import DataUtil


class Perceptron(ClassifierBase, metaclass=ClassifierMeta):
    PerceptronTiming = Timing()

    def __init__(self):
        self._w = self._b = None

    @PerceptronTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, lr=1, epoch=10 ** 6):
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

if __name__ == '__main__':
    _x, _y = DataUtil.gen_two_clusters(center=5, dis=3, scale=2, one_hot=False)
    perceptron = Perceptron()
    perceptron.fit(_x, _y)
    perceptron.estimate(_x, _y)
    perceptron.visualize2d(_x, _y, dense=400)
    perceptron.show_timing_log()
