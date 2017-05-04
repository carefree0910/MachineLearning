import numpy as np
import matplotlib.pyplot as plt

from Util.Util import DataUtil
from Util.Timing import Timing
from Util.Bases import KernelBase


class KernelPerceptron(KernelBase):
    KernelPerceptronTiming = Timing()

    def __init__(self, **kwargs):
        super(KernelPerceptron, self).__init__(**kwargs)
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
        _err = (np.sign(self._prediction_cache) != self._y) * sample_weight
        _indices = np.random.permutation(len(self._y))
        _idx = _indices[np.argmax(_err[_indices])]
        if self._prediction_cache[_idx] == self._y[_idx]:
            return True
        self._update_dw_cache(_idx, lr, sample_weight)
        self._update_db_cache(_idx, lr, sample_weight)
        self._update_pred_cache(_idx)

if __name__ == '__main__':
    # # xs, ys = DataUtil.gen_two_clusters(center=5, dis=1, scale=2, one_hot=False)
    # xs, ys = DataUtil.gen_spiral(20, 4, 2, 2, one_hot=False)
    # # xs, ys = DataUtil.gen_xor(one_hot=False)
    # ys[ys == 0] = -1
    # perceptron = KernelPerceptron()
    # _logs = [_log[0] for _log in perceptron.fit(
    #     xs, ys, metrics=["acc"], epoch=10 ** 5
    # )]
    # # perceptron.fit(xs, ys, kernel="rbf", epoch=10 ** 6)
    # # perceptron.fit(xs, ys, p=12, epoch=10 ** 5)
    # perceptron.evaluate(xs, ys)
    # perceptron.visualize2d(xs, ys, dense=400)

    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", train_num=100, quantize=True, tar_idx=0)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    perceptron = KernelPerceptron()
    _logs = [_log[0] for _log in perceptron.fit(
        x_train, y_train, metrics=["acc"], x_test=x_test, y_test=y_test
    )]
    perceptron.evaluate(x_train, y_train)
    perceptron.evaluate(x_test, y_test)

    plt.figure()
    plt.title(perceptron.title)
    plt.plot(range(len(_logs)), _logs)
    plt.show()

    perceptron.show_timing_log()
