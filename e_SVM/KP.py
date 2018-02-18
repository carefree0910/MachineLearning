import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt

from Util.Util import DataUtil
from Util.Timing import Timing
from Util.Bases import KernelBase, GDKernelBase


class KP(KernelBase):
    KernelPerceptronTiming = Timing()

    def __init__(self, **kwargs):
        super(KP, self).__init__(**kwargs)
        self._fit_args, self._fit_args_names = [0.01], ["lr"]

    @KernelPerceptronTiming.timeit(level=1, prefix="[Core] ")
    def _update_dw_cache(self, idx, lr, sample_weight):
        self._dw_cache = lr * self._y[idx] * sample_weight[idx]
        self._w[idx] += self._dw_cache

    @KernelPerceptronTiming.timeit(level=1, prefix="[Core] ")
    def _update_db_cache(self, idx, lr, sample_weight):
        self._db_cache = self._dw_cache
        self._b += self._db_cache

    @KernelPerceptronTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, sample_weight, lr):
        err = (np.sign(self._prediction_cache) != self._y) * sample_weight
        indices = np.random.permutation(len(self._y))
        idx = indices[np.argmax(err[indices])]
        if self._prediction_cache[idx] == self._y[idx]:
            return True
        self._update_dw_cache(idx, lr, sample_weight)
        self._update_db_cache(idx, lr, sample_weight)
        self._update_pred_cache(idx)


class GDKP(GDKernelBase):
    GDKPTiming = Timing()

    @GDKPTiming.timeit(level=1, prefix="[Core] ")
    def _get_grads(self, x_batch, y_batch, y_pred, sample_weight_batch, *args):
        err = -y_batch * (x_batch.dot(self._alpha) + self._b) * sample_weight_batch
        mask = err >= 0  # type: np.ndarray
        if not np.any(mask):
            self._model_grads = [None, None]
        else:
            delta = -y_batch[mask] * sample_weight_batch[mask]
            self._model_grads = [
                np.sum(delta[..., None] * x_batch[mask], axis=0),
                np.sum(delta)
            ]
        return np.sum(err[mask])


if __name__ == '__main__':
    # xs, ys = DataUtil.gen_two_clusters(center=5, dis=1, scale=2, one_hot=False)
    xs, ys = DataUtil.gen_spiral(20, 4, 2, 2, one_hot=False)
    # xs, ys = DataUtil.gen_xor(one_hot=False)
    ys[ys == 0] = -1

    animation_params = {
        "show": False, "mp4": False, "period": 50,
        "dense": 400, "draw_background": True
    }

    kp = KP(animation_params=animation_params)
    kp.fit(xs, ys, kernel="poly", p=12, epoch=200)
    kp.evaluate(xs, ys)
    kp.visualize2d(xs, ys, dense=400)

    kp = GDKP(animation_params=animation_params)
    kp.fit(xs, ys, kernel="poly", p=12, epoch=10000)
    kp.evaluate(xs, ys)
    kp.visualize2d(xs, ys, dense=400)

    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", n_train=100, quantize=True, tar_idx=0)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    kp = KP()
    logs = [log[0] for log in kp.fit(
        x_train, y_train, metrics=["acc"], x_test=x_test, y_test=y_test
    )]
    kp.evaluate(x_train, y_train)
    kp.evaluate(x_test, y_test)

    plt.figure()
    plt.title(kp.title)
    plt.plot(range(len(logs)), logs)
    plt.show()

    kp = GDKP()
    logs = [log[0] for log in kp.fit(
        x_train, y_train, metrics=["acc"], x_test=x_test, y_test=y_test
    )]
    kp.evaluate(x_train, y_train)
    kp.evaluate(x_test, y_test)

    plt.figure()
    plt.title(kp.title)
    plt.plot(range(len(logs)), logs)
    plt.show()

    kp.show_timing_log()
