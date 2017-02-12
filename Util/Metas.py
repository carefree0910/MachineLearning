import time
import numpy as np
from abc import ABCMeta
import matplotlib.pyplot as plt

from Util.Timing import Timing


class TimingMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        try:
            _timing = attr[name + "Timing"]
        except KeyError:
            _timing = Timing()
            attr[name + "Timing"] = _timing

        for _name, _value in attr.items():
            if "__" in _name or "timing" in _name or "estimate" in _name:
                continue
            _str_val = str(_value)
            if "<" not in _str_val and ">" not in _str_val:
                continue
            if _str_val.find("function") >= 0 or _str_val.find("staticmethod") >= 0 or _str_val.find("property") >= 0:
                attr[_name] = _timing.timeit(level=2)(_value)

        def feed_timing(self, timing):
            setattr(self, name + "Timing", timing)

        def show_timing_log(self, level=2):
            getattr(self, name + "Timing").show_timing_log(level)

        attr["feed_timing"] = feed_timing
        attr["show_timing_log"] = show_timing_log

        return type(name, bases, attr)


class ClassifierMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        try:
            clf_timing = attr[name + "Timing"]
        except KeyError:
            clf_timing = Timing()
            attr[name + "Timing"] = clf_timing

        def __str__(self):
            try:
                return self.name
            except AttributeError:
                return name

        def __repr__(self):
            return str(self)

        def __getitem__(self, item):
            if isinstance(item, str):
                return getattr(self, "_" + item)

        def acc(y, y_pred, weights=None):
            if weights is not None:
                return np.sum((np.array(y) == np.array(y_pred)) * weights) / len(y)
            return np.sum(np.array(y) == np.array(y_pred)) / len(y)

        attr["_metrics"] = [acc]

        @clf_timing.timeit(level=1, prefix="[API] ")
        def estimate(self, x, y, metrics=None, tar=0, prefix="Acc"):
            if metrics is None:
                metrics = self._metrics
            else:
                for i in range(len(metrics) - 1, -1, -1):
                    metric = metrics[i]
                    if isinstance(metric, str):
                        if metric not in self._available_metrics:
                            metrics.pop(i)
                        else:
                            metrics[i] = self._available_metrics[metric]
            logs, y_pred = [], self.predict(x)
            for metric in metrics:
                logs.append(metric(y, y_pred))
            if isinstance(tar, int):
                print(prefix + ": {:12.8}".format(logs[tar]))
            return logs

        def visualize2d(self, x, y, dense=100):
            axis, labels = np.array(x).T, np.array(y)

            print("=" * 30 + "\n" + str(self))
            decision_function = lambda _xx: self.predict(_xx)

            nx, ny, margin = dense, dense, 0.1
            x_min, x_max = np.min(axis[0]), np.max(axis[0])
            y_min, y_max = np.min(axis[1]), np.max(axis[1])
            x_margin = max(abs(x_min), abs(x_max)) * margin
            y_margin = max(abs(y_min), abs(y_max)) * margin
            x_min -= x_margin
            x_max += x_margin
            y_min -= y_margin
            y_max += y_margin

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

            plt.figure()
            plt.pcolormesh(xy_xf, xy_yf, z > 0, cmap=plt.cm.Paired)
            plt.contour(xf, yf, z, c='k-', levels=[0])
            plt.scatter(axis[0], axis[1], c=colors)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.show()

            print("Done.")

        def feed_timing(self, timing):
            setattr(self, name + "Timing", timing)

        def show_timing_log(self, level=2):
            getattr(self, name + "Timing").show_timing_log(level)

        for key, value in locals().items():
            if str(value).find("function") >= 0 or str(value).find("property"):
                attr[key] = value

        return type(name, bases, attr)


class SubClassTimingMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        try:
            timing = getattr(bases[0], bases[0].__name__ + "Timing")
        except AttributeError:
            timing = Timing()
            setattr(bases[0], bases[0].__name__ + "Timing", timing)
        for _name, _value in attr.items():
            if "__" in _name or "timing" in _name or "estimate" in _name:
                continue
            _str_val = str(_value)
            if "<" not in _str_val and ">" not in _str_val:
                continue
            if _str_val.find("function") >= 0 or _str_val.find("staticmethod") >= 0 or _str_val.find("property") >= 0:
                attr[_name] = timing.timeit(level=2)(_value)
        return type(name, bases, attr)


class SklearnCompatibleMeta(ABCMeta, ClassifierMeta):
    pass
