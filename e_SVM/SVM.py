import numpy as np

from Util.Timing import Timing
from Util.Bases import KernelBase


class SVM(KernelBase):
    SVMTiming = Timing()

    def __init__(self):
        KernelBase.__init__(self)
        self._fit_args, self._fit_args_names = [1e-3], ["tol"]
        self._c = None

    @SVMTiming.timeit(level=1, prefix="[SMO] ")
    def _pick_first(self, tol):
        con1 = self._alpha > 0
        con2 = self._alpha < self._c
        err1 = self._y * self._prediction_cache - 1
        err2 = err1.copy()
        err3 = err1.copy()
        err1[con1 | (err1 >= 0)] = 0
        err2[(~con1 | ~con2) | (err2 == 0)] = 0
        err3[con2 | (err3 <= 0)] = 0
        err = err1 ** 2 + err2 ** 2 + err3 ** 2
        # noinspection PyTypeChecker
        idx = np.argmax(err)
        if err[idx] < tol:
            return
        return idx

    @SVMTiming.timeit(level=1, prefix="[SMO] ")
    def _pick_second(self, idx1):
        idx = np.random.randint(len(self._y))
        while idx == idx1:
            idx = np.random.randint(len(self._y))
        return idx

    @SVMTiming.timeit(level=2, prefix="[SMO] ")
    def _get_lower_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return max(0., self._alpha[idx2] - self._alpha[idx1])
        return max(0., self._alpha[idx2] + self._alpha[idx1] - self._c)

    @SVMTiming.timeit(level=2, prefix="[SMO] ")
    def _get_upper_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return min(self._c, self._c + self._alpha[idx2] - self._alpha[idx1])
        return min(self._c, self._alpha[idx2] + self._alpha[idx1])

    @SVMTiming.timeit(level=1, prefix="[SMO] ")
    def _update_alpha(self, idx1, idx2):
        l, h = self._get_lower_bound(idx1, idx2), self._get_upper_bound(idx1, idx2)
        y1, y2 = self._y[idx1], self._y[idx2]
        e1 = self._prediction_cache[idx1] - self._y[idx1]
        e2 = self._prediction_cache[idx2] - self._y[idx2]
        eta = self._gram[idx1][idx1] + self._gram[idx2][idx2] - 2 * self._gram[idx1][idx2]
        a2_new = self._alpha[idx2] + (y2 * (e1 - e2)) / eta
        if a2_new > h:
            a2_new = h
        elif a2_new < l:
            a2_new = l
        a1_old, a2_old = self._alpha[idx1], self._alpha[idx2]
        da2 = a2_new - a2_old
        da1 = -y1 * y2 * da2
        self._alpha[idx1] += da1
        self._alpha[idx2] = a2_new
        self._update_dw_cache(idx1, idx2, da1, da2, y1, y2)
        self._update_db_cache(idx1, idx2, da1, da2, y1, y2, e1, e2)
        self._update_pred_cache(idx1, idx2)

    @SVMTiming.timeit(level=1, prefix="[Core] ")
    def _update_dw_cache(self, idx1, idx2, da1, da2, y1, y2):
        self._dw_cache = np.array([da1 * y1, da2 * y2])
        self._w[idx1] += self._dw_cache[0]
        self._w[idx2] += self._dw_cache[1]

    @SVMTiming.timeit(level=1, prefix="[Core] ")
    def _update_db_cache(self, idx1, idx2, da1, da2, y1, y2, e1, e2):
        gram_12 = self._gram[idx1][idx2]
        b1 = -e1 - y1 * self._gram[idx1][idx1] * da1 - y2 * gram_12 * da2
        b2 = -e2 - y1 * gram_12 * da1 - y2 * self._gram[idx2][idx2] * da2
        self._db_cache = (b1 + b2) * 0.5
        self._b += self._db_cache

    @SVMTiming.timeit(level=4, prefix="[Util] ")
    def _prepare(self, **kwargs):
        self._c = kwargs.get("c", self._config.default_c)

    @SVMTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, sample_weight, tol):
        idx1 = self._pick_first(tol)
        if idx1 is None:
            return True
        idx2 = self._pick_second(idx1)
        self._update_alpha(idx1, idx2)
