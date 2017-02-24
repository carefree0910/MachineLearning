import numpy as np

from Util.Bases import ClassifierBase
from Util.Metas import ClassifierMeta
from Util.Timing import Timing


class LinearSVM(ClassifierBase, metaclass=ClassifierMeta):
    LinearSVMTiming = Timing()

    def __init__(self):
        self._w = self._b = None

    @LinearSVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, c=1, lr=0.01, epoch=10 ** 4, tol=1e-3):
        x, y = np.atleast_2d(x), np.array(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.array(sample_weight) * len(y)
        self._w = np.zeros(x.shape[1])
        self._b = 0
        for _ in range(epoch):
            _err = (1 - self.predict(x, get_raw_results=True) * y) * sample_weight
            _indices = np.random.permutation(len(y))
            _idx = _indices[np.argmax(_err[_indices])]
            if _err[_idx] <= tol:
                return
            _delta = lr * c * y[_idx] * sample_weight[_idx]
            self._w *= 1 - lr
            self._w += _delta * x[_idx]
            self._b += _delta

    @LinearSVMTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False):
        rs = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(rs)
        return rs
