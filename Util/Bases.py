import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Util.Timing import Timing


class ClassifierBase:
    clf_timing = Timing()

    def __init__(self, *args, **kwargs):
        self._title = self._name = None
        self._metrics = [ClassifierBase.acc]
        self._available_metrics = {
            "acc": ClassifierBase.acc
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
        ClassifierBase.clf_timing.disable()

    @staticmethod
    def show_timing_log(level=2):
        ClassifierBase.clf_timing.show_timing_log(level)

    @staticmethod
    def acc(y, y_pred, weights=None):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
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
    def estimate(self, x, y, metrics=None, tar=None, prefix="Acc"):
        if metrics is None:
            metrics = ["acc"]
        self.get_metrics(metrics)
        logs, y_pred = [], self.predict(x)
        y = np.array(y)
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
                    title=None, show_org=False, show_background=True, emphasize=None):
        axis, labels = np.array(x).T, np.array(y)

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
            _indices[np.array(emphasize)] = True
            plt.scatter(axis[0][_indices], axis[1][_indices], s=80,
                        facecolors="None", zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def visualize3d(self, x, y, padding=0.1, dense=100,
                    title=None, show_org=False, show_background=True, emphasize=None):
        if False:
            print(Axes3D.add_artist)
        axis, labels = np.array(x).T, np.array(y)

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
                _indices[np.array(emphasize)] = True
                _ax.scatter(axis0[_indices], axis1[_indices], s=80,
                            facecolors="None", zorder=10)

        colors = colors[labels]

        ax1.set_title("xy figure")
        _draw(ax1, xy_xf, xf, xy_yf, yf, z_xy)
        _emphasize(ax1, axis[0], axis[1], colors)

        ax2.set_title("yz figure")
        _draw(ax2, yz_xf, yf, yz_yf, zf, z_yz)
        _emphasize(ax2, axis[1], axis[2], colors)

        ax3.set_title("xz figure")
        _draw(ax3, xz_xf, xf, xz_yf, zf, z_xz)
        _emphasize(ax3, axis[0], axis[2], colors)

        plt.show()

        print("Done.")


class KernelConfig:
    default_c = 1
    default_p = 3


class KernelBase(ClassifierBase):
    KernelBaseTiming = Timing()

    def __init__(self):
        super(KernelBase, self).__init__()
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
