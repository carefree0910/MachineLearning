import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np

from Util.Timing import Timing
from Util.Bases import ClassifierBase
from Util.ProgressBar import ProgressBar


class Perceptron(ClassifierBase):
    PerceptronTiming = Timing()

    def __init__(self, **kwargs):
        super(Perceptron, self).__init__(**kwargs)
        self._w = self._b = None

        self._params["lr"] = kwargs.get("lr", 0.01)
        self._params["epoch"] = kwargs.get("epoch", 10 ** 4)

    @PerceptronTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, lr=None, epoch=None, animation_params=None):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        *animation_properties, animation_params = self._get_animation_params(animation_params)

        x, y = np.atleast_2d(x), np.asarray(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray(sample_weight) * len(y)

        self._w = np.zeros(x.shape[1])
        self._b = 0.
        ims = []
        bar = ProgressBar(max_value=epoch, name="Perceptron")
        for i in range(epoch):
            err = -y * self.predict(x, True) * sample_weight
            idx = np.argmax(err)
            if err[idx] < 0:
                bar.terminate()
                break
            delta = lr * y[idx] * sample_weight[idx]
            self._w += delta * x[idx]
            self._b += delta
            self._handle_animation(i, x, y, ims, animation_params, *animation_properties)
            bar.update()
        self._handle_mp4(ims, animation_properties)

    @PerceptronTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, **kwargs):
        rs = np.asarray(x, dtype=np.float32).dot(self._w) + self._b
        if get_raw_results:
            return rs
        return np.sign(rs).astype(np.float32)


class Perceptron2(Perceptron):
    def fit(self, x, y, sample_weight=None, lr=None, epoch=None, animation_params=None):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        *animation_properties, animation_params = self._get_animation_params(animation_params)

        x, y = np.atleast_2d(x), np.asarray(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray(sample_weight) * len(y)

        self._w = np.random.random(x.shape[1])
        self._b = 0.
        ims = []
        bar = ProgressBar(max_value=epoch, name="Perceptron")
        for i in range(epoch):
            y_pred = self.predict(x, True)
            err = -y * y_pred * sample_weight
            idx = np.argmax(err)
            if err[idx] < 0:
                bar.terminate()
                break
            w_norm = np.linalg.norm(self._w)
            delta = lr * y[idx] * sample_weight[idx] / w_norm
            self._w += delta * (x[idx] - y_pred[idx] * self._w / w_norm ** 2)
            self._b += delta
            self._handle_animation(i, x, y, ims, animation_params, *animation_properties)
            bar.update()
        self._handle_mp4(ims, animation_properties)
