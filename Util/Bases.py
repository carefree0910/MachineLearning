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

    def get_metrics(self, metrics):
        pass

    def predict(self, x, get_raw_results=False):
        pass

    def estimate(self, x, y, metrics=None, tar=None, prefix="Acc"):
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
    default_p = 3


class KernelBase(ClassifierBase, metaclass=ClassifierMeta):
    KernelBaseTiming = Timing()

    def __init__(self):
        self._config = KernelConfig()
        self._fit_args, self._fit_args_names = None, []
        self._x = self._y = self._gram = None
        self._w = self._b = self._alpha = None
        self._kernel = self._kernel_name = self._kernel_param = None
        self._prediction_cache = self._dw_cache = self._db_cache = None

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

    def _update_dw_cache(self, *args):
        pass

    def _update_db_cache(self, *args):
        pass

    @KernelBaseTiming.timeit(level=1, prefix="[Core] ")
    def _update_pred_cache(self, *args):
        self._prediction_cache += self._db_cache
        if len(args) == 1:
            self._prediction_cache += self._dw_cache * self._gram[args[0]]
        elif len(args) == len(self._gram):
            self._prediction_cache = self._dw_cache.dot(self._gram)
        else:
            self._prediction_cache += self._dw_cache.dot(self._gram[args, ...])

    def _prepare(self, **kwargs):
        pass

    def _fit(self, *args):
        pass

    @KernelBaseTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, kernel="rbf", epoch=10 ** 4,
            x_test=None, y_test=None, metrics=None, **kwargs):
        self._x, self._y = np.atleast_2d(x), np.array(y)
        if kernel == "poly":
            _p = kwargs.get("p", self._config.default_p)
            self._kernel_name = "Polynomial"
            self._kernel_param = "degree = {}".format(_p)
            self._kernel = lambda _x, _y: KernelBase._poly(_x, _y, _p)
        elif kernel == "rbf":
            _gamma = kwargs.get("gamma", 1 / self._x.shape[1])
            self._kernel_name = "RBF"
            self._kernel_param = r"$\gamma = {:8.6}$".format(_gamma)
            self._kernel = lambda _x, _y: KernelBase._rbf(_x, _y, _gamma)
        else:
            raise NotImplementedError("Kernel '{}' has not defined".format(kernel))
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.array(sample_weight) * len(y)

        self._alpha, self._w, self._prediction_cache = (
            np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)))
        self._gram = self._kernel(self._x, self._x)
        self._b = 0
        self._prepare(**kwargs)

        _fit_args, _logs = [], []
        for _name, _arg in zip(self._fit_args_names, self._fit_args):
            if _name in kwargs:
                _arg = kwargs[_name]
            _fit_args.append(_arg)
        if metrics is not None:
            self.get_metrics(metrics)
        _test_gram = None
        if x_test is not None and y_test is not None:
            _xv, _yv = np.atleast_2d(x_test), np.array(y_test)
            _test_gram = self._kernel(_xv, self._x)
        else:
            _xv, _yv = self._x, self._y
        for _ in range(epoch):
            if self._fit(sample_weight, *_fit_args):
                break
            if metrics is not None:
                _local_logs = []
                for metric in metrics:
                    if _test_gram is None:
                        _local_logs.append(metric(self._y, np.sign(self._prediction_cache)))
                    else:
                        _local_logs.append(metric(_yv, self.predict(_test_gram, provide_gram=True)))
                _logs.append(_local_logs)
        return _logs

    @KernelBaseTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, provide_gram=False):
        if not provide_gram:
            x = self._kernel(np.atleast_2d(x), self._x)
        y_pred = x.dot(self._w) + self._b
        if not get_raw_results:
            return np.sign(y_pred)
        return y_pred
