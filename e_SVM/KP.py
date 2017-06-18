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

    def __init__(self, **kwargs):
        super(GDKP, self).__init__(**kwargs)
        self._fit_args, self._fit_args_names = [1e-3], ["tol"]

    @GDKPTiming.timeit(level=1, prefix="[Core] ")
    def _loss(self, y, y_pred, sample_weight):
        return np.sum(
            np.maximum(0, 1 - y * y_pred) * sample_weight
        )

    @GDKPTiming.timeit(level=1, prefix="[Core] ")
    def _get_grads(self, x_batch, y_batch, y_pred, sample_weight_batch, *args):
        err = -y_batch * (x_batch.dot(self._alpha) + self._b)
        if np.max(err) < 0:
            return [None, None]
        mask = err >= 0
        delta = -y_batch[mask]
        self._model_grads = [
            np.sum(delta[..., None] * x_batch[mask], axis=0),
            np.sum(delta)
        ]

if __name__ == '__main__':
    # xs, ys = DataUtil.gen_two_clusters(center=5, dis=1, scale=2, one_hot=False)
    xs, ys = DataUtil.gen_spiral(20, 4, 2, 2, one_hot=False)
    # xs, ys = DataUtil.gen_xor(one_hot=False)
    ys[ys == 0] = -1

    animation_params = {
        "show": False, "mp4": False, "period": 500,
        "dense": 400, "draw_background": True
    }

    kp = KP(animation_params=animation_params)
    kp.fit(xs, ys, p=12, epoch=10 ** 4)
    kp.evaluate(xs, ys)
    kp.visualize2d(xs, ys, dense=400)

    kp = GDKP(animation_params=animation_params)
    kp.fit(xs, ys, p=12, epoch=10 ** 4)
    kp.evaluate(xs, ys)
    kp.visualize2d(xs, ys, dense=400)

    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", train_num=100, quantize=True, tar_idx=0)
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
