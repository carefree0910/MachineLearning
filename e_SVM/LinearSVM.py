import cv2
import numpy as np
import tensorflow as tf

from Util.Util import VisUtil
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
    def fit(self, x, y, sample_weight=None, c=None, lr=None, epoch=None, tol=None,
            show_animations=None, make_mp4=None):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if c is None:
            c = self._params["c"]
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        if tol is None:
            tol = self._params["tol"]
        if show_animations is None:
            show_animations = self._params["show_animations"]
        if make_mp4 is None:
            make_mp4 = self._params["make_mp4"]
        draw_animations = show_animations or make_mp4
        x, y = np.atleast_2d(x), np.asarray(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray(sample_weight) * len(y)

        self._w = np.zeros(x.shape[1])
        self._b = 0
        ims = []
        bar = ProgressBar(max_value=epoch, name="LinearSVM")
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
            if draw_animations and x.shape[1] == 2:
                img = self.get_2d_plot(x, y)
                if show_animations:
                    cv2.imshow("Perceptron", img)
                    cv2.waitKey(1)
                if make_mp4:
                    ims.append(img)
            bar.update()
        if make_mp4:
            VisUtil.make_mp4(ims, "LinearSVM")

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
        self._tfx = self._y_pred_raw = self._y_pred = None
        self._sess = tf.Session()

        self._params["c"] = kwargs.get("c", 1)
        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["epoch"] = kwargs.get("epoch", 10 ** 4)
        self._params["tol"] = kwargs.get("tol", 1e-3)

    @TFLinearSVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, c=None, lr=None, epoch=None, tol=None,
            show_animations=None, make_mp4=None):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if c is None:
            c = self._params["c"]
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        if tol is None:
            tol = self._params["tol"]
        if show_animations is None:
            show_animations = self._params["show_animations"]
        if make_mp4 is None:
            make_mp4 = self._params["make_mp4"]
        draw_animations = show_animations or make_mp4
        x, y = np.atleast_2d(x), np.asarray(y)
        if sample_weight is None:
            sample_weight = tf.constant(np.ones([len(y), 1]), dtype=tf.float32, name="sample_weight")
        else:
            sample_weight = tf.constant(np.asarray(sample_weight)[..., None] * len(y),
                                        dtype=tf.float32, name="sample_weight")

        self._w = tf.Variable(np.zeros([x.shape[1], 1]), dtype=tf.float32, name="w")
        self._b = tf.Variable(0., dtype=tf.float32, name="b")
        self._tfx = tf.placeholder(tf.float32, [None, x.shape[1]])
        self._y_pred_raw = tf.matmul(self._tfx, self._w) + self._b
        self._y_pred = tf.sign(self._y_pred_raw)
        cost = tf.reduce_sum(
            tf.nn.relu(1 - y[..., None] * self._y_pred_raw) * sample_weight
        ) + c * tf.nn.l2_loss(self._w)
        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
        self._sess.run(tf.global_variables_initializer())
        bar = ProgressBar(max_value=epoch, name="TFLinearSVM")
        ims = []
        for _ in range(epoch):
            _l = self._sess.run([cost, train_step], {self._tfx: x})[0]
            if _l < tol:
                bar.update(epoch)
                break
            if draw_animations and x.shape[1] == 2:
                img = self.get_2d_plot(x, y)
                if show_animations:
                    cv2.imshow("Perceptron", img)
                    cv2.waitKey(1)
                if make_mp4:
                    ims.append(img)
            bar.update()
        if make_mp4:
            VisUtil.make_mp4(ims, "TFLinearSVM")

    @TFLinearSVMTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False):
        rs = self._y_pred_raw if get_raw_results else self._y_pred
        return self._sess.run(rs, {self._tfx: x}).ravel()
