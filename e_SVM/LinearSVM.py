import numpy as np
import tensorflow as tf

from Util.Timing import Timing
from Util.Bases import ClassifierBase
from Util.ProgressBar import ProgressBar


class LinearSVM(ClassifierBase):
    LinearSVMTiming = Timing()

    def __init__(self, **kwargs):
        super(LinearSVM, self).__init__(**kwargs)
        self._w = self._b = None

        self._params["c"] = kwargs.get("c", 1)
        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["epoch"] = kwargs.get("epoch", 10 ** 4)
        self._params["tol"] = kwargs.get("tol", 1e-3)

    @LinearSVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, c=None, lr=None, epoch=None, tol=None):
        if sample_weight is None:
            sample_weight = self._params["sw"]
        if c is None:
            c = self._params["c"]
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        if tol is None:
            tol = self._params["tol"]
        x, y = np.atleast_2d(x), np.array(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.array(sample_weight) * len(y)
        self._w = np.zeros(x.shape[1])
        self._b = 0
        bar = ProgressBar(max_value=epoch, name="LinearSVM")
        bar.start()
        for _ in range(epoch):
            _err = (1 - self.predict(x, get_raw_results=True) * y) * sample_weight
            _indices = np.random.permutation(len(y))
            _idx = _indices[np.argmax(_err[_indices])]
            if _err[_idx] <= tol:
                bar.update(epoch)
                return
            _delta = lr * c * y[_idx] * sample_weight[_idx]
            self._w *= 1 - lr
            self._w += _delta * x[_idx]
            self._b += _delta
            bar.update()

    @LinearSVMTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False):
        rs = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(rs)
        return rs


class TFLinearSVM(ClassifierBase):
    TFLinearSVMTiming = Timing()

    def __init__(self, **kwargs):
        super(TFLinearSVM, self).__init__(**kwargs)
        self._w = self._b = None
        self._sess = tf.Session()

        self._params["c"] = kwargs.get("c", 1)
        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["epoch"] = kwargs.get("epoch", 10 ** 4)
        self._params["tol"] = kwargs.get("tol", 1e-3)

    @TFLinearSVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, c=None, lr=None, epoch=None, tol=None):
        if sample_weight is None:
            sample_weight = self._params["sw"]
        if c is None:
            c = self._params["c"]
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        if tol is None:
            tol = self._params["tol"]
        if sample_weight is None:
            sample_weight = tf.constant(np.ones(len(y)), dtype=tf.float32, name="sample_weight")
        else:
            sample_weight = tf.constant(np.array(sample_weight) * len(y), dtype=tf.float32, name="sample_weight")
        x, y = tf.constant(x, dtype=tf.float32), tf.constant(y, dtype=tf.float32)
        self._w = tf.Variable(np.zeros(x.shape[1]), dtype=tf.float32, name="w")
        self._b = tf.Variable(0., dtype=tf.float32, name="b")
        y_pred = self.predict(x, True, False)
        cost = tf.reduce_sum(tf.maximum(1 - y * y_pred, 0) * sample_weight) + c * tf.nn.l2_loss(self._w)
        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
        self._sess.run(tf.global_variables_initializer())
        bar = ProgressBar(max_value=epoch, name="TFLinearSVM")
        bar.start()
        for _ in range(epoch):
            _l = self._sess.run([cost, train_step])[0]
            if _l < tol:
                bar.update(epoch)
                break
            bar.update()

    @TFLinearSVMTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, out_of_sess=True):
        rs = tf.reduce_sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            rs = tf.sign(rs)
        if out_of_sess:
            rs = self._sess.run(rs)
        return rs
