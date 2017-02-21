import numpy as np

from Util.Timing import Timing
from Util.Metas import ClassifierMeta


class TimingBase:
    def __str__(self):
        pass

    def __repr__(self):
        pass

    def feed_timing(self, timing):
        pass

    def show_timing_log(self, level=2):
        pass


class ClassifierBase:
    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __getitem__(self, item):
        pass

    @staticmethod
    def acc(y, y_pred, weights=None):
        pass

    def estimate(self, x, y):
        pass

    def visualize2d(self, x, y, padding=0.1, dense=200,
                    title=None, show_org=False, show_background=True, emphasize=None):
        pass

    def visualize3d(self, x, y, padding=0.1, dense=200,
                    title=None, show_org=False, show_background=True, emphasize=None):
        pass

    def feed_timing(self, timing):
        pass

    def show_timing_log(self, level=2):
        pass


class KernelConfig:
    default_c = 1
    default_p = 4


class KernelBase(ClassifierBase, metaclass=ClassifierMeta):
    KernelBaseTiming = Timing()

    def __init__(self):
        self._fit_args, self._fit_args_names = None, []
        self._x = self._y = self._gram = None
        self._w = self._b = self._alpha = None
        self._kernel = self._kernel_name = self._kernel_param = None
        self._prediction_cache = self._dw_cache = self._b_cache = None

    @property
    def title(self):
        return "{} {} ({})".format(self._kernel_name, self, self._kernel_param)

    # Kernel

    @staticmethod
    @KernelBaseTiming.timeit(level=1, prefix="[Kernel] ")
    def _poly(x, y, p):
        return (x.dot(y.T) + 1) ** p

    # noinspection PyTypeChecker
    @staticmethod
    @KernelBaseTiming.timeit(level=1, prefix="[Kernel] ")
    def _rbf(x, y, gamma):
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))

    @KernelBaseTiming.timeit(level=1, prefix="[Core] ")
    def _predict(self):
        if self._prediction_cache is None:
            self._prediction_cache = np.sum(self._w * self._gram, axis=1) + self._b
        return self._prediction_cache

    def _update_w(self, *args):
        pass

    def _update_b(self, *args):
        pass

    @KernelBaseTiming.timeit(level=1, prefix="[Core] ")
    def _update_pred_cache(self, *args):
        self._prediction_cache += self._b - self._b_cache
        self._prediction_cache += np.sum(self._dw_cache * self._gram[..., args], axis=1)

    def _prepare(self, **kwargs):
        pass

    def _fit(self, *args):
        pass

    @KernelBaseTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, kernel="poly", epoch=10 ** 4, **kwargs):
        self._x, self._y = np.atleast_2d(x), np.array(y)
        if kernel == "poly":
            _p = kwargs.get("p", KernelConfig.default_p)
            self._kernel_name = "Polynomial"
            self._kernel_param = "degree = {}".format(_p)
            self._kernel = lambda _x, _y: KernelBase._poly(_x, _y, _p)
        elif kernel == "rbf":
            _gamma = kwargs.get("gamma", 1 / self._x.shape[1])
            self._kernel_name = "RBF"
            self._kernel_param = "gamma = {}".format(_gamma)
            self._kernel = lambda _x, _y: KernelBase._rbf(_x, _y, _gamma)
        else:
            raise NotImplementedError("Kernel '{}' has not defined".format(kernel))
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.array(sample_weight) * len(y)
        self._alpha, self._w = np.zeros(len(x)), np.zeros(len(x))
        self._gram = self._kernel(self._x, self._x)
        self._b = 0
        self._prepare(**kwargs)
        _fit_args = []
        for _name, _arg in zip(self._fit_args_names, self._fit_args):
            if _name in kwargs:
                _arg = kwargs[_name]
            _fit_args.append(_arg)
        for _ in range(epoch):
            if self._fit(sample_weight, *_fit_args):
                break

    @KernelBaseTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False):
        x = self._kernel(np.atleast_2d(x), self._x)
        y_pred = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(y_pred)
        return y_pred
