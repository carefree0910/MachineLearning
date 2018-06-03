import io
import cv2
import time
import math
import ctypes
import multiprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D

from NN.Basic.Optimizers import OptFactory

from Util.Util import VisUtil
from Util.Timing import Timing
from Util.ProgressBar import ProgressBar

try:
    import torch
    from torch.autograd import Variable
except ImportError:
    torch = Variable = None


class TimingBase:
    def show_timing_log(self):
        pass


class ModelBase:
    """
        Base for all models
        Magic methods:
            1) __str__     : return self.name; __repr__ = __str__
            2) __getitem__ : access to protected members
        Properties:
            1) name  : name of this model, self.__class__.__name__ or self._name
            2) title : used in matplotlib (plt.title())
        Static method:
            1) disable_timing  : disable Timing()
            2) show_timing_log : show Timing() records
    """

    clf_timing = Timing()

    def __init__(self, **kwargs):
        self._plot_label_dict = {}
        self._title = self._name = None
        self._metrics, self._available_metrics = [], {
            "acc": ClassifierBase.acc
        }
        self._params = {
            "sample_weight": kwargs.get("sample_weight", None)
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

    # Handle animation

    @staticmethod
    def _refresh_animation_params(animation_params):
        animation_params["show"] = animation_params.get("show", False)
        animation_params["mp4"] = animation_params.get("mp4", False)
        animation_params["period"] = animation_params.get("period", 1)

    def _get_animation_params(self, animation_params):
        if animation_params is None:
            animation_params = self._params["animation_params"]
        else:
            ClassifierBase._refresh_animation_params(animation_params)
        show, mp4, period = animation_params["show"], animation_params["mp4"], animation_params["period"]
        return show or mp4, show, mp4, period, animation_params

    def _handle_animation(self, i, x, y, ims, animation_params, draw_ani, show_ani, make_mp4, ani_period,
                          name=None, img=None):
        if draw_ani and x.shape[1] == 2 and (i + 1) % ani_period == 0:
            if img is None:
                img = self.get_2d_plot(x, y, **animation_params)
            if name is None:
                name = str(self)
            if show_ani:
                cv2.imshow(name, img)
                cv2.waitKey(1)
            if make_mp4:
                ims.append(img)

    def _handle_mp4(self, ims, animation_properties, name=None):
        if name is None:
            name = str(self)
        if animation_properties[2] and ims:
            VisUtil.make_mp4(ims, name)

    def get_2d_plot(self, x, y, padding=1, dense=200, draw_background=False, emphasize=None, extra=None, **kwargs):
        pass

    # Visualization

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
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        indices = [labels == i for i in range(np.max(labels) + 1)]
        scatters = []
        plt.figure()
        plt.title(title)
        for idx in indices:
            scatters.append(plt.scatter(axis[0][idx], axis[1][idx], c=colors[idx]))
        plt.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                   ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
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
                dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(dic)
                arr = np.array([dic[label] for label in arr])
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
        indices = [labels == i for i in range(n_label)]
        scatters = []
        fig = plt.figure()
        plt.title(title)
        ax = fig.add_subplot(111, projection='3d')
        for _index in indices:
            scatters.append(ax.scatter(axis[0][_index], axis[1][_index], axis[2][_index], c=colors[_index]))
        ax.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                  ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
        plt.show()

    # Util

    def predict(self, x, get_raw_results=False, **kwargs):
        pass


class ClassifierBase(ModelBase):
    """
        Base for classifiers
        Static methods:
            1) acc, f1_score           : Metrics
            2) _multi_clf, _multi_data : Parallelization
    """

    clf_timing = Timing()

    def __init__(self, **kwargs):
        super(ClassifierBase, self).__init__(**kwargs)
        self._params["animation_params"] = kwargs.get("animation_params", {})
        ClassifierBase._refresh_animation_params(self._params["animation_params"])

    # Metrics

    @staticmethod
    def acc(y, y_pred, weights=None):
        y, y_pred = np.asarray(y), np.asarray(y_pred)
        if weights is not None:
            return np.average((y == y_pred) * weights)
        return np.average(y == y_pred)

    # noinspection PyTypeChecker
    @staticmethod
    def f1_score(y, y_pred):
        tp = np.sum(y * y_pred)
        if tp == 0:
            return .0
        fp = np.sum((1 - y) * y_pred)
        fn = np.sum(y * (1 - y_pred))
        return 2 * tp / (2 * tp + fn + fp)

    # Parallelization

    # noinspection PyUnusedLocal
    @staticmethod
    def _multi_clf(x, clfs, task, kwargs, stack=np.vstack, target="single"):
        if target != "parallel":
            return np.array([clf.predict(x) for clf in clfs], dtype=np.float32).T
        n_cores = kwargs.get("n_cores", 2)
        n_cores = multiprocessing.cpu_count() if n_cores <= 0 else n_cores
        if n_cores == 1:
            matrix = np.array([clf.predict(x, n_cores=1) for clf in clfs], dtype=np.float32).T
        else:
            pool = Pool(processes=n_cores)
            batch_size = int(len(clfs) / n_cores)
            clfs = [clfs[i*batch_size:(i+1)*batch_size] for i in range(n_cores)]
            x_size = np.prod(x.shape)  # type: int
            shared_base = multiprocessing.Array(ctypes.c_float, int(x_size))
            shared_matrix = np.ctypeslib.as_array(shared_base.get_obj()).reshape(x.shape)
            shared_matrix[:] = x
            matrix = stack(
                pool.map(task, ((shared_matrix, clfs, n_cores) for clfs in clfs))
            ).T.astype(np.float32)
        return matrix

    # noinspection PyUnusedLocal
    def _multi_data(self, x, task, kwargs, stack=np.hstack, target="single"):
        if target != "parallel":
            return task((x, self, 1))
        n_cores = kwargs.get("n_cores", 2)
        n_cores = multiprocessing.cpu_count() if n_cores <= 0 else n_cores
        if n_cores == 1:
            matrix = task((x, self, n_cores))
        else:
            pool = Pool(processes=n_cores)
            batch_size = int(len(x) / n_cores)
            batch_base, batch_data, cursor = [], [], 0
            x_dim = x.shape[1]
            for i in range(n_cores):
                if i == n_cores - 1:
                    batch_data.append(x[cursor:])
                    batch_base.append(multiprocessing.Array(ctypes.c_float, (len(x) - cursor) * x_dim))
                else:
                    batch_data.append(x[cursor:cursor + batch_size])
                    batch_base.append(multiprocessing.Array(ctypes.c_float, batch_size * x_dim))
                cursor += batch_size
            shared_arrays = [
                np.ctypeslib.as_array(shared_base.get_obj()).reshape(-1, x_dim)
                for shared_base in batch_base
            ]
            for i, data in enumerate(batch_data):
                shared_arrays[i][:] = data
            matrix = stack(
                pool.map(task, ((x, self, n_cores) for x in shared_arrays))
            )
        return matrix.astype(np.float32)

    # Training

    @staticmethod
    def _get_train_repeat(x, batch_size):
        train_len = len(x)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len > batch_size
        return 1 if not do_random_batch else int(train_len / batch_size) + 1

    def _batch_work(self, *args):
        pass

    def _batch_training(self, x, y, batch_size, train_repeat, *args):
        pass

    # Visualization

    def get_2d_plot(self, x, y, padding=1, dense=200, title=None,
                    draw_background=False, emphasize=None, extra=None, **kwargs):
        axis, labels = np.asarray(x).T, np.asarray(y)
        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])  # type: float
        y_min, y_max = np.min(axis[1]), np.max(axis[1])  # type: float
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
        z = self.predict(base_matrix).reshape((nx, ny))

        if labels.ndim == 1:
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        buffer_ = io.BytesIO()
        plt.figure()
        if title is None:
            title = self.title
        plt.title(title)
        if draw_background:
            xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Pastel1)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)
        if emphasize is not None:
            indices = np.array([False] * len(axis[0]))
            indices[np.asarray(emphasize)] = True
            plt.scatter(axis[0][indices], axis[1][indices], s=80,
                        facecolors="None", zorder=10)
        if extra is not None:
            plt.scatter(*np.asarray(extra).T, s=80, zorder=25, facecolors="red")

        plt.savefig(buffer_, format="png")
        plt.close()
        buffer_.seek(0)
        image = Image.open(buffer_)
        canvas = np.asarray(image)[..., :3]
        buffer_.close()
        return canvas

    def visualize2d(self, x, y, padding=0.1, dense=200, title=None,
                    show_org=False, draw_background=True, emphasize=None, extra=None, **kwargs):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
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
        z = self.predict(base_matrix, **kwargs).reshape((nx, ny))
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        if labels.ndim == 1:
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
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
        if draw_background:
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Pastel1)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)
        if emphasize is not None:
            indices = np.array([False] * len(axis[0]))
            indices[np.asarray(emphasize)] = True
            plt.scatter(axis[0][indices], axis[1][indices], s=80,
                        facecolors="None", zorder=10)
        if extra is not None:
            plt.scatter(*np.asarray(extra).T, s=80, zorder=25, facecolors="red")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def visualize3d(self, x, y, padding=0.1, dense=100, title=None,
                    show_org=False, draw_background=True, emphasize=None, extra=None, **kwargs):
        if False:
            print(Axes3D.add_artist)
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))

        def decision_function(xx):
            return self.predict(xx, **kwargs)

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
                dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(dic)
                arr = np.array([dic[label] for label in arr])
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
            if draw_background:
                _ax.pcolormesh(_x, _y, _z > 0, cmap=plt.cm.Pastel1)
            else:
                _ax.contour(_xf, _yf, _z, c='k-', levels=[0])

        def _emphasize(_ax, axis0, axis1, _c):
            _ax.scatter(axis0, axis1, c=_c)
            if emphasize is not None:
                indices = np.array([False] * len(axis[0]))
                indices[np.asarray(emphasize)] = True
                _ax.scatter(axis0[indices], axis1[indices], s=80,
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

    # Util

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

    @clf_timing.timeit(level=1, prefix="[API] ")
    def evaluate(self, x, y, metrics=None, tar=0, prefix="Acc", **kwargs):
        if metrics is None:
            metrics = ["acc"]
        self.get_metrics(metrics)
        logs, y_pred = [], self.predict(x, **kwargs)
        y = np.asarray(y)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        for metric in metrics:
            logs.append(metric(y, y_pred))
        if isinstance(tar, int):
            print(prefix + ": {:12.8}".format(logs[tar]))
        return logs


class GDBase(ClassifierBase):
    """
        Gradient descent classifier framework
        Requirements:
            1) Store all parameters in self._model_parameters
            2) Calculate all gradients of the parameters and store them in self._grads
              in self._get_grads() method
        See self._update_model_params() method for more insights
    """

    GDBaseTiming = Timing()

    def __init__(self, **kwargs):
        super(GDBase, self).__init__(**kwargs)
        self._optimizer = self._model_parameters = self._model_grads = None

    def _get_grads(self, x_batch, y_batch, y_pred, sample_weight_batch, *args):
        return 0

    def _update_model_params(self):
        for i, (param, grad) in enumerate(zip(self._model_parameters, self._model_grads)):
            if grad is not None:
                param -= self._optimizer.run(i, grad)

    @GDBaseTiming.timeit(level=1, prefix="[Core] ")
    def _batch_training(self, x, y, batch_size, train_repeat, *args, **kwargs):
        sample_weight, *args = args
        epoch_loss = 0
        for i in range(train_repeat):
            if train_repeat != 1:
                batch = np.random.permutation(len(x))[:batch_size]
                x_batch, y_batch = x[batch], y[batch]
                sample_weight_batch = sample_weight[batch]
            else:
                x_batch, y_batch, sample_weight_batch = x, y, sample_weight
            y_pred = self.predict(x_batch, get_raw_results=True, **kwargs)
            epoch_loss += self._get_grads(x_batch, y_batch, y_pred, sample_weight_batch, *args)
            self._update_model_params()
            self._batch_work(i, *args)
        return epoch_loss / train_repeat


class TFClassifierBase(ClassifierBase):
    """
        Tensorfow classifier framework
        Implemented tensorflow ver. metrics
    """

    clf_timing = Timing()

    def __init__(self, **kwargs):
        super(TFClassifierBase, self).__init__(**kwargs)
        self._tfx = self._tfy = None
        self._y_pred_raw = self._y_pred = None
        self._sess = tf.Session()

    @clf_timing.timeit(level=2, prefix="[Core] ")
    def _batch_training(self, x, y, batch_size, train_repeat, *args):
        loss, train_step, *args = args
        epoch_cost = 0
        for i in range(train_repeat):
            if train_repeat != 1:
                batch = np.random.choice(len(x), batch_size)
                x_batch, y_batch = x[batch], y[batch]
            else:
                x_batch, y_batch = x, y
            epoch_cost += self._sess.run([loss, train_step], {
                self._tfx: x_batch, self._tfy: y_batch
            })[0]
            self._batch_work(i, *args)
        return epoch_cost / train_repeat


if torch is not None:
    class TorchBasicClassifierBase(ClassifierBase):
        """ Basic torch's classifier base """
        def _handle_animation(self, i, x, y, ims, animation_params, draw_ani, show_ani, make_mp4, ani_period,
                              name=None, img=None):
            x, y = x.numpy(), y.numpy()
            super(TorchBasicClassifierBase, self)._handle_animation(
                i, x, y, ims, animation_params, draw_ani, show_ani, make_mp4, ani_period, name, img
            )

    class TorchAutoClassifierBase(TorchBasicClassifierBase):
        """
            Torch classifier framework
            Utilizing torch's auto-grad, but not utilizing torch's optimizers
            Static method:
                _arr_to_variable : change iterable(s) to torch Variable(s)
        """
        def __init__(self, **kwargs):
            super(TorchAutoClassifierBase, self).__init__(**kwargs)
            self._optimizer = self._model_parameters = None

        @staticmethod
        def _arr_to_variable(requires_grad, *args):
            return [
                Variable(
                    torch.from_numpy(np.asarray(arr, dtype=np.float32)),
                    requires_grad=requires_grad
                ) for arr in args
            ]

        # Training

        def _update_model_params(self):
            for i, param in enumerate(self._model_parameters):
                if param.grad is not None:
                    param.data -= self._optimizer.run(i, param.grad.data)

        def _reset_grad(self):
            for param in self._model_parameters:
                param.grad.data.zero_()

        def batch_training(self, x, y, batch_size, train_repeat, *args):
            loss_function, *args = args
            epoch_cost = 0
            for i in range(train_repeat):
                if train_repeat != 1:
                    batch = torch.randperm(len(x))[:batch_size]
                    x_batch, y_batch = x[batch], y[batch]
                else:
                    x_batch, y_batch = x, y
                y_pred = self._predict(x_batch, get_raw_results=True)
                local_loss = loss_function(y_batch, y_pred)
                epoch_cost += local_loss
                local_loss.backward()
                self._update_model_params()
                self._reset_grad()
                self._batch_work(i, *args)
            return float((epoch_cost / train_repeat).data.numpy()[0])

        # Util

        def _predict(self, x, get_raw_results=False, **kwargs):
            pass

        def predict(self, x, get_raw_results=False, **kwargs):
            return self._predict(x, get_raw_results, **kwargs).data.numpy()

        def _handle_animation(self, i, x, y, ims, animation_params, draw_ani, show_ani, make_mp4, ani_period,
                              name=None, img=None):
            x, y = x.data.numpy(), y.data.numpy()
            super(TorchBasicClassifierBase, self)._handle_animation(
                i, x, y, ims, animation_params, draw_ani, show_ani, make_mp4, ani_period, name, img
            )
else:
    TorchBasicClassifierBase = TorchAutoClassifierBase = None


class KernelBase(ClassifierBase):
    """ Kernel classifier with SMO-like algorithm """

    KernelBaseTiming = Timing()

    def __init__(self, **kwargs):
        super(KernelBase, self).__init__(**kwargs)
        self._do_log = True
        self._is_torch = False
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

    @staticmethod
    @KernelBaseTiming.timeit(level=1, prefix="[Kernel] ")
    def _rbf(x, y, gamma):
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))

    # Training

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

    def _torch_transform(self, y_cv):
        pass

    def _fit(self, *args):
        pass

    @KernelBaseTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, kernel=None, epoch=None,
            x_test=None, y_test=None, metrics=None, animation_params=None, **kwargs):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]  # type: list
        if kernel is None:
            kernel = self._params["kernel"]
        if epoch is None:
            epoch = self._params["epoch"]
        if x_test is None:
            x_test = self._params["x_test"]  # type: list
        if y_test is None:
            y_test = self._params["y_test"]  # type: list
        if metrics is None:
            metrics = self._params["metrics"]  # type: list
        *animation_properties, animation_params = self._get_animation_params(animation_params)
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

        fit_args, logs, ims = [], [], []
        for name, arg in zip(self._fit_args_names, self._fit_args):
            if name in kwargs:
                arg = kwargs[name]
            fit_args.append(arg)
        if self._do_log:
            if metrics is not None:
                self.get_metrics(metrics)
            test_gram = None
            if x_test is not None and y_test is not None:
                x_cv, y_cv = np.atleast_2d(x_test), np.asarray(y_test)
                test_gram = self._kernel(self._x, x_cv)
            else:
                x_cv, y_cv = self._x, self._y
        else:
            y_cv = test_gram = None

        if self._is_torch:
            y_cv, self._x, self._y = self._torch_transform(y_cv)

        bar = ProgressBar(max_value=epoch, name=str(self))
        for i in range(epoch):
            if self._fit(sample_weight, *fit_args):
                bar.terminate()
                break
            if self._do_log and metrics is not None:
                local_logs = []
                for metric in metrics:
                    if test_gram is None:
                        if self._is_torch:
                            local_y = self._y.data.numpy()
                        else:
                            local_y = self._y
                        local_logs.append(metric(local_y, np.sign(self._prediction_cache)))
                    else:
                        if self._is_torch:
                            local_y = y_cv.data.numpy()
                        else:
                            local_y = y_cv
                        local_logs.append(metric(local_y, self.predict(test_gram, gram_provided=True)))
                logs.append(local_logs)
            self._handle_animation(i, self._x, self._y, ims, animation_params, *animation_properties)
            bar.update()
        self._handle_mp4(ims, animation_properties)
        return logs

    # Util

    @KernelBaseTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, gram_provided=False):
        if not gram_provided:
            x = self._kernel(self._x, np.atleast_2d(x))
        y_pred = self._w.dot(x) + self._b
        if not get_raw_results:
            return np.sign(y_pred)
        return y_pred


class GDKernelBase(KernelBase, GDBase):
    """ Kernel classifier with gradient descent algorithm """

    GDKernelBaseTiming = Timing()

    def __init__(self, **kwargs):
        super(GDKernelBase, self).__init__(**kwargs)
        self._fit_args, self._fit_args_names = [1e-3], ["tol"]
        self._batch_size = kwargs.get("batch_size", 128)
        self._optimizer = kwargs.get("optimizer", "Adam")
        self._train_repeat = 0

    def _prepare(self, sample_weight, **kwargs):
        lr = kwargs.get("lr", self._params["lr"])
        self._alpha = np.random.random(len(self._x)).astype(np.float32)
        self._b = np.random.random(1).astype(np.float32)
        self._model_parameters = [self._alpha, self._b]
        self._optimizer = OptFactory().get_optimizer_by_name(
            self._optimizer, self._model_parameters, lr, self._params["epoch"]
        )

    @GDKernelBaseTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, sample_weight, tol):
        if self._train_repeat == 0:
            self._train_repeat = self._get_train_repeat(self._x, self._batch_size)
        l = self._batch_training(
            self._gram, self._y, self._batch_size, self._train_repeat,
            sample_weight, gram_provided=True
        )
        if l < tol:
            return True

    @GDKernelBaseTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, gram_provided=False):
        if not gram_provided:
            x = self._kernel(self._x, np.atleast_2d(x))
        elif self._alpha.shape[0] != x.shape[0]:
            x = x.T
        y_pred = (self._alpha.dot(x) + self._b).ravel()
        if not get_raw_results:
            return np.sign(y_pred)
        return y_pred


class TFKernelBase(KernelBase, TFClassifierBase):
    """ Kernel classifier with tensorflow """

    def __init__(self, **kwargs):
        super(TFKernelBase, self).__init__(**kwargs)
        self._loss = None


if TorchAutoClassifierBase is not None:
    class TorchKernelBase(KernelBase, TorchAutoClassifierBase):
        """ Kernel classifier with torch """

        def __init__(self, **kwargs):
            super(TorchKernelBase, self).__init__(**kwargs)
            self._loss_function = None
            self._is_torch = True

        def _torch_transform(self, y_cv):
            return self._arr_to_variable(False, y_cv, self._x, self._y)
else:
    TorchKernelBase = None


class RegressorBase(ModelBase):
    def predict(self, x, **kwargs):
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
        x_base = np.linspace(x_min, x_max, dense)  # type: np.ndarray
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
