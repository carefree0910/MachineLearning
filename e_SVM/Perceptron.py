import numpy as np

from Util.Bases import ClassifierBase
from Util.Metas import ClassifierMeta
from Util.Util import DataUtil


class Perceptron(ClassifierBase, metaclass=ClassifierMeta):
    def __init__(self):
        self._w = self._b = None

    def fit(self, x, y, sample_weight=None, lr=1, epoch=1000):
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

    def predict(self, x, get_raw_results=False):
        rs = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(rs)
        return rs

if __name__ == '__main__':
    # x1 = np.arange(5) * 0.1 + 0.25
    # x2 = 1 - x1
    # gap = 0.01
    # x1 = np.vstack((x1, x2)).T
    # x2 = x1 + gap
    # _x = np.vstack((x1, x2))
    # _y = np.array([-1] * 5 + [1] * 5)
    _x, _y = DataUtil.gen_two_clusters(one_hot=False)
    perceptron = Perceptron()
    perceptron.fit(_x, _y)
    perceptron.estimate(_x, _y)
    perceptron.visualize2d(_x, _y)
