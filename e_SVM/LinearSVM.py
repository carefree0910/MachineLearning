import numpy as np

from Util.Bases import ClassifierBase
from Util.Metas import ClassifierMeta
from Util.Timing import Timing


class LinearSVM(ClassifierBase, metaclass=ClassifierMeta):
    LinearSVMTiming = Timing()

    def __init__(self):
        self._w = self._b = self._c = None
        self._prediction_cache = None

    @LinearSVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, c=1, lr=0.01, epoch=10 ** 4):
        x, y = np.atleast_2d(x), np.array(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.array(sample_weight) * len(y)
        self._w, self._prediction_cache = np.zeros(x.shape[1]), np.zeros(len(x))
        self._b, self._c = 0, c
        for _ in range(epoch):
            y_pred = self.predict(x, get_raw_results=True)
            _err = (1 - y_pred * y) * sample_weight
            _indices = np.random.permutation(len(y))
            _idx = _indices[np.argmax(_err[_indices])]
            if _err[_idx] <= 0:
                return
            _rate, _delta = 1 - lr / self._c, lr * y[_idx] * sample_weight[_idx]
            self._w *= _rate
            self._b *= _rate
            self._w += _delta * x[_idx]
            self._b += _delta

    @LinearSVMTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False):
        rs = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(rs)
        return rs
