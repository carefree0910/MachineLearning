import time
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Util.Timing import Timing
from Util.ProgressBar import ProgressBar


class TimingBase:
    def show_timing_log(self):
        pass


class ModelBase:
    clf_timing = Timing()

    def __init__(self, **kwargs):
        self._title = self._name = None
        self._metrics, self._available_metrics = [], {
            "acc": ClassifierBase.acc
        }
        self._params = {
            "sw": kwargs.get("sample_weight", None)
        }

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    @property
    def name(self):
        return self.__class__.__name__ if self._name is None else self._name

    @property
    def title(self):
        return str(self) if self._title is None else self._title

    @staticmethod
    def disable_timing():
        ModelBase.clf_timing.disable()

    @staticmethod
    def show_timing_log(level=2):
        ModelBase.clf_timing.show_timing_log(level)

    def scatter2d(self, x, y, padding=0.5, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        if labels.ndim == 1:
            _dic = {c: i for i, c in enumerate(set(labels))}
            n_label = len(_dic)
            labels = np.array([_dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        _indices = [labels == i for i in range(np.max(labels) + 1)]
        _scatters = []
        plt.figure()
        plt.title(title)
        for _index in _indices:
            _scatters.append(plt.scatter(axis[0][_index], axis[1][_index], c=colors[_index]))
        plt.legend(_scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(_scatters))],
                   ncol=math.ceil(math.sqrt(len(_scatters))), fontsize=8)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def scatter3d(self, x, y, padding=0.1, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def transform_arr(arr):
            if arr.ndim == 1:
                _dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(_dic)
                arr = np.array([_dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        if title is None:
            try:
                title = self.title
            except AttributeError:
                title = str(self)

        labels, n_label = transform_arr(labels)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]
        _indices = [labels == i for i in range(n_label)]
        _scatters = []
        fig = plt.figure()
        plt.title(title)
        ax = fig.add_subplot(111, projection='3d')
        for _index in _indices:
            _scatters.append(ax.scatter(axis[0][_index], axis[1][_index], axis[2][_index], c=colors[_index]))
        ax.legend(_scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(_scatters))],
                  ncol=math.ceil(math.sqrt(len(_scatters))), fontsize=8)
        plt.show()


class ClassifierBase(ModelBase):
    clf_timing = Timing()

    @staticmethod
    def acc(y, y_pred, weights=None):
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.asarray(y_pred)
        if weights is not None:
            return np.sum((y == y_pred) * weights) / len(y)
        return np.sum(y == y_pred) / len(y)

    # noinspection PyTypeChecker
    @staticmethod
    def f1_score(y, y_pred):
        tp = np.sum(y * y_pred)
        if tp == 0:
            return .0
        fp = np.sum((1 - y) * y_pred)
        fn = np.sum(y * (1 - y_pred))
        return 2 * tp / (2 * tp + fn + fp)

    def get_metrics(self, metrics):
        if len(metrics) == 0:
            for metric in self._metrics:
                metrics.append(metric)
            return metrics
        for i in range(len(metrics) - 1, -1, -1):
            metric = metrics[i]
            if isinstance(metric, str):
                try:
                    metrics[i] = self._available_metrics[metric]
                except AttributeError:
                    metrics.pop(i)
        return metrics

    def predict(self, x, get_raw_results=False):
        pass

    @clf_timing.timeit(level=1, prefix="[API] ")
    def evaluate(self, x, y, metrics=None, tar=None, prefix="Acc"):
        if metrics is None:
            metrics = ["acc"]
        self.get_metrics(metrics)
        logs, y_pred = [], self.predict(x)
        y = np.asarray(y)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        for metric in metrics:
            logs.append(metric(y, y_pred))
        if tar is None:
            tar = 0
        if isinstance(tar, int):
            print(prefix + ": {:12.8}".format(logs[tar]))
        return logs

    def visualize2d(self, x, y, padding=0.1, dense=200,
                    title=None, show_org=False, show_background=True, emphasize=None, extra=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        decision_function = lambda _xx: self.predict(_xx)

        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)

        t = time.time()
        z = decision_function(base_matrix).reshape((nx, ny))
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        if labels.ndim == 1:
            _dic = {c: i for i, c in enumerate(set(labels))}
            n_label = len(_dic)
            labels = np.array([_dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        if show_org:
            plt.figure()
            plt.scatter(axis[0], axis[1], c=colors)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.show()

        plt.figure()
        plt.title(title)
        if show_background:
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Paired)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)
        if emphasize is not None:
            _indices = np.array([False] * len(axis[0]))
            _indices[np.asarray(emphasize)] = True
            plt.scatter(axis[0][_indices], axis[1][_indices], s=80,
                        facecolors="None", zorder=10)
        if extra is not None:
            plt.scatter(*np.asarray(extra).T, s=80, zorder=25, facecolors="red")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def visualize3d(self, x, y, padding=0.1, dense=100,
                    title=None, show_org=False, show_background=True, emphasize=None, extra=None):
        if False:
            print(Axes3D.add_artist)
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        decision_function = lambda _x: self.predict(_x)

        nx, ny, nz, padding = dense, dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def get_base(_nx, _ny, _nz):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            _zf = np.linspace(z_min, z_max, _nz)
            n_xf, n_yf, n_zf = np.meshgrid(_xf, _yf, _zf)
            return _xf, _yf, _zf, np.c_[n_xf.ravel(), n_yf.ravel(), n_zf.ravel()]

        xf, yf, zf, base_matrix = get_base(nx, ny, nz)

        t = time.time()
        z_xyz = decision_function(base_matrix).reshape((nx, ny, nz))
        p_classes = decision_function(x).astype(np.int8)
        _, _, _, base_matrix = get_base(10, 10, 10)
        z_classes = decision_function(base_matrix).astype(np.int8)
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        z_xy = np.average(z_xyz, axis=2)
        z_yz = np.average(z_xyz, axis=1)
        z_xz = np.average(z_xyz, axis=0)

        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        yz_xf, yz_yf = np.meshgrid(yf, zf, sparse=True)
        xz_xf, xz_yf = np.meshgrid(xf, zf, sparse=True)

        def transform_arr(arr):
            if arr.ndim == 1:
                _dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(_dic)
                arr = np.array([_dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        labels, n_label = transform_arr(labels)
        p_classes, _ = transform_arr(p_classes)
        z_classes, _ = transform_arr(z_classes)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])
        if extra is not None:
            ex0, ex1, ex2 = np.asarray(extra).T
        else:
            ex0 = ex1 = ex2 = None

        if title is None:
            try:
                title = self.title
            except AttributeError:
                title = str(self)

        if show_org:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(axis[0], axis[1], axis[2], c=colors[labels])
            plt.show()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        plt.title(title)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        ax1.set_title("Org")
        ax2.set_title("Pred")
        ax3.set_title("Boundary")

        ax1.scatter(axis[0], axis[1], axis[2], c=colors[labels])
        ax2.scatter(axis[0], axis[1], axis[2], c=colors[p_classes], s=15)
        if extra is not None:
            ax2.scatter(ex0, ex1, ex2, s=80, zorder=25, facecolors="red")
        xyz_xf, xyz_yf, xyz_zf = base_matrix[..., 0], base_matrix[..., 1], base_matrix[..., 2]
        ax3.scatter(xyz_xf, xyz_yf, xyz_zf, c=colors[z_classes], s=15)

        plt.show()
        plt.close()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        def _draw(_ax, _x, _xf, _y, _yf, _z):
            if show_background:
                _ax.pcolormesh(_x, _y, _z > 0, cmap=plt.cm.Paired)
            else:
                _ax.contour(_xf, _yf, _z, c='k-', levels=[0])

        def _emphasize(_ax, axis0, axis1, _c):
            _ax.scatter(axis0, axis1, c=_c)
            if emphasize is not None:
                _indices = np.array([False] * len(axis[0]))
                _indices[np.asarray(emphasize)] = True
                _ax.scatter(axis0[_indices], axis1[_indices], s=80,
                            facecolors="None", zorder=10)

        def _extra(_ax, axis0, axis1, _c, _ex0, _ex1):
            _emphasize(_ax, axis0, axis1, _c)
            if extra is not None:
                _ax.scatter(_ex0, _ex1, s=80, zorder=25, facecolors="red")

        colors = colors[labels]

        ax1.set_title("xy figure")
        _draw(ax1, xy_xf, xf, xy_yf, yf, z_xy)
        _extra(ax1, axis[0], axis[1], colors, ex0, ex1)

        ax2.set_title("yz figure")
        _draw(ax2, yz_xf, yf, yz_yf, zf, z_yz)
        _extra(ax2, axis[1], axis[2], colors, ex1, ex2)

        ax3.set_title("xz figure")
        _draw(ax3, xz_xf, xf, xz_yf, zf, z_xz)
        _extra(ax3, axis[0], axis[2], colors, ex0, ex2)

        plt.show()

        print("Done.")


class KernelBase(ClassifierBase):
    KernelBaseTiming = Timing()

    def __init__(self, **kwargs):
        super(KernelBase, self).__init__(**kwargs)
        self._do_log = True
        self._fit_args, self._fit_args_names = None, []
        self._x = self._y = self._gram = None
        self._w = self._b = self._alpha = None
        self._kernel = self._kernel_name = self._kernel_param = None
        self._prediction_cache = self._dw_cache = self._db_cache = None

        self._params["kernel"] = kwargs.get("kernel", "rbf")
        self._params["epoch"] = kwargs.get("epoch", 10 ** 4)
        self._params["x_test"] = kwargs.get("x_test", None)
        self._params["y_test"] = kwargs.get("y_test", None)
        self._params["metrics"] = kwargs.get("metrics", None)
        self._params["c"] = kwargs.get("c", 1)
        self._params["p"] = kwargs.get("p", 3)
        self._params["lr"] = kwargs.get("lr", 0.001)

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

    def _prepare(self, sample_weight, **kwargs):
        pass

    def _fit(self, *args):
        pass

    @KernelBaseTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, kernel=None, epoch=None,
            x_test=None, y_test=None, metrics=None, **kwargs):
        if sample_weight is None:
            sample_weight = self._params["sw"]
        if kernel is None:
            kernel = self._params["kernel"]
        if epoch is None:
            epoch = self._params["epoch"]
        if x_test is None:
            x_test = self._params["x_test"]
        if y_test is None:
            y_test = self._params["y_test"]
        if metrics is None:
            metrics = self._params["metrics"]
        self._x, self._y = np.atleast_2d(x), np.asarray(y)
        if kernel == "poly":
            _p = kwargs.get("p", self._params["p"])
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
            sample_weight = np.asarray(sample_weight) * len(y)

        self._alpha, self._w, self._prediction_cache = (
            np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)))
        self._gram = self._kernel(self._x, self._x)
        self._b = 0
        self._prepare(sample_weight, **kwargs)

        _fit_args, _logs = [], []
        for _name, _arg in zip(self._fit_args_names, self._fit_args):
            if _name in kwargs:
                _arg = kwargs[_name]
            _fit_args.append(_arg)
        if self._do_log:
            if metrics is not None:
                self.get_metrics(metrics)
            _test_gram = None
            if x_test is not None and y_test is not None:
                _xv, _yv = np.atleast_2d(x_test), np.asarray(y_test)
                _test_gram = self._kernel(_xv, self._x)
            else:
                _xv, _yv = self._x, self._y
        else:
            _yv = _test_gram = None
        bar = ProgressBar(max_value=epoch, name=str(self))
        bar.start()
        for _ in range(epoch):
            if self._fit(sample_weight, *_fit_args):
                bar.update(epoch)
                break
            if self._do_log and metrics is not None:
                _local_logs = []
                for metric in metrics:
                    if _test_gram is None:
                        _local_logs.append(metric(self._y, np.sign(self._prediction_cache)))
                    else:
                        _local_logs.append(metric(_yv, self.predict(_test_gram, gram_provided=True)))
                _logs.append(_local_logs)
            bar.update()
        return _logs

    @KernelBaseTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, gram_provided=False):
        if not gram_provided:
            x = self._kernel(np.atleast_2d(x), self._x)
        y_pred = x.dot(self._w) + self._b
        if not get_raw_results:
            return np.sign(y_pred)
        return y_pred


class RegressorBase(ModelBase):
    def predict(self, x):
        return x

    def visualize2d(self, x, y, padding=0.1, dense=400, title=None):
        x, y = np.asarray(x).ravel(), np.asarray(y)

        print("=" * 30 + "\n" + str(self))

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        x_padding = max(abs(x_min), abs(x_max)) * padding
        x_min -= x_padding
        x_max += x_padding

        t = time.time()
        x_base = np.linspace(x_min, x_max, dense)
        y_pred = self.predict(x_base[..., None])
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")

        if title is None:
            title = self.title

        plt.figure()
        plt.title(title)
        plt.scatter(x, y, c="g", s=20)
        plt.plot(x_base, y_pred)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")
