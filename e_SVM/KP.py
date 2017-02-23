import numpy as np
import matplotlib.pyplot as plt

from Util.Bases import KernelBase
from Util.Metas import SubClassChangeNamesMeta
from Util.Timing import Timing
from Util.Util import DataUtil


class KernelPerceptron(KernelBase, metaclass=SubClassChangeNamesMeta):
    KernelPerceptronTiming = Timing()

    def __init__(self):
        KernelBase.__init__(self)
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
        # noinspection PyTypeChecker
        _idx = _indices[np.argmax(_err[_indices])]
        if self._prediction_cache[_idx] == self._y[_idx]:
            return True
        self._alpha[_idx] += lr
        self._update_dw_cache(_idx, lr, sample_weight)
        self._update_db_cache(_idx, lr, sample_weight)
        self._update_pred_cache(_idx)

if __name__ == '__main__':
    # xs, ys = DataUtil.gen_two_clusters(center=5, dis=1, scale=2, one_hot=False)
    xs, ys = DataUtil.gen_spin(20, 4, 2, 2, one_hot=False)
    # xs, ys = DataUtil.gen_xor(one_hot=False)
    ys[ys == 0] = -1
    perceptron = KernelPerceptron()
    _logs = [_log[0] for _log in perceptron.fit(
        xs, ys, kernel="rbf", metrics=["acc"], epoch=10 ** 5
    )]
    # perceptron.fit(xs, ys, kernel="rbf", epoch=10 ** 6)
    # perceptron.fit(xs, ys, p=12, epoch=10 ** 5)
    perceptron.estimate(xs, ys)
    perceptron.visualize2d(xs, ys, dense=400)

    # (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
    #     "mushroom", "../_Data/mushroom.txt", train_num=100, quantize=True, tar_idx=0)
    # y_train[y_train == 0] = -1
    # y_test[y_test == 0] = -1
    #
    # perceptron = KernelPerceptron()
    # _logs = [_log[0] for _log in perceptron.fit(
    #     x_train, y_train, kernel="rbf", metrics=["acc"], x_test=x_test, y_test=y_test
    # )]
    # perceptron.estimate(x_train, y_train)
    # perceptron.estimate(x_test, y_test)
    #
    plt.figure()
    plt.title(perceptron.title)
    plt.plot(range(len(_logs)), _logs)
    plt.show()

    perceptron.show_timing_log()
