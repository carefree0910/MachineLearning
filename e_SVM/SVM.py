import numpy as np
import tensorflow as tf

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
    def _prepare(self, sample_weight, **kwargs):
        self._c = kwargs.get("c", self._config.default_c)

    @SVMTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, sample_weight, tol):
        idx1 = self._pick_first(tol)
        if idx1 is None:
            return True
        idx2 = self._pick_second(idx1)
        self._update_alpha(idx1, idx2)


class TFSVM(KernelBase):
    TFSVMTiming = Timing()

    def __init__(self):
        super(TFSVM, self).__init__()
        self._fit_args, self._fit_args_names = [1e-3], ["tol"]
        self._lr = self._cost = self._train_step = None
        self._sess = tf.Session()
        self._do_log = False

    def _prepare(self, sample_weight, **kwargs):
        self._lr = kwargs.get("lr", self._config.default_lr)
        sample_weight = tf.constant(sample_weight, dtype=tf.float32, name="sample_weight")
        x, y = tf.constant(self._x, dtype=tf.float32), tf.constant(self._y, dtype=tf.float32)
        self._gram = tf.constant(self._kernel(self._x, self._x), dtype=tf.float32, name="gram")
        self._w = tf.Variable(np.zeros(len(self._x))[None, ...], dtype=tf.float32, name="w")
        self._b = tf.Variable(.0, dtype=tf.float32, name="b")
        y_pred = self.predict(x, True, True)
        self._cost = tf.reduce_sum(tf.maximum(1 - y * y_pred, 0) * sample_weight) + 0.5 * tf.matmul(
            self._w, tf.matmul(self._gram, self._w, transpose_b=True)
        )[0][0]
        self._train_step = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self._cost)
        self._sess.run(tf.global_variables_initializer())

    @TFSVMTiming.timeit(level=1, prefix="[API] ")
    def _fit(self, sample_weight, tol):
        _l = self._sess.run([self._cost, self._train_step])[0]
        if _l < tol:
            return True

    @TFSVMTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, gram_provided=False):
        if gram_provided:
            gram = self._gram
        else:
            gram = tf.constant(self._kernel(np.atleast_2d(x), self._x).T, dtype=tf.float32)
        rs = tf.matmul(self._w, gram) + self._b
        if not get_raw_results:
            rs = tf.sign(rs)
        if not gram_provided:
            rs = self._sess.run(rs)
        return rs
