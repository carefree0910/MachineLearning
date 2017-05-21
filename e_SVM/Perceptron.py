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
        self._b = 0
        ims = []
        bar = ProgressBar(max_value=epoch, name="Perceptron")
        for i in range(epoch):
            y_pred = self.predict(x)
            _err = (y_pred != y) * sample_weight
            _indices = np.random.permutation(len(y))
            _idx = _indices[np.argmax(_err[_indices])]
            if y_pred[_idx] == y[_idx]:
                bar.update(epoch)
                break
            _delta = lr * y[_idx] * sample_weight[_idx]
            self._w += _delta * x[_idx]
            self._b += _delta
            self._handle_animation(i, x, y, ims, animation_params, *animation_properties)
            bar.update()
        self._handle_mp4(ims, animation_properties)

    @PerceptronTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False):
        rs = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(rs)
        return rs
