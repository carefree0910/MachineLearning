import time
import wrapt
import pickle
import numpy as np
from math import sqrt, ceil
import matplotlib.pyplot as plt


class DataUtil:

    @staticmethod
    def gen_xor(size, scale=1):
        x = np.random.randn(size) * scale
        y = np.random.randn(size) * scale
        z = np.zeros((size, 2))
        z[x * y >= 0, :] = [0, 1]
        z[x * y < 0, :] = [1, 0]
        return np.c_[x, y].astype(np.float32), z

    @staticmethod
    def gen_spin(size, n=7, n_class=7):
        xs = np.zeros((size * n, 2), dtype=np.float32)
        ys = np.zeros(size * n, dtype=np.int8)
        for j in range(n):
            ix = range(size * j, size * (j + 1))
            r = np.linspace(0.0, 1, size+1)[1:]
            t = np.array(
                np.linspace(j * (n + 1), (j + 1) * (n + 1), size) +
                np.array(np.random.random(size=size)) * 0.2)
            xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            ys[ix] = j % n_class
        z = []
        for yy in ys:
            tmp = [0 if i != yy else 1 for i in range(n_class)]
            z.append(tmp)
        return xs, np.array(z)


class VisUtil:

    @staticmethod
    def get_line_info(weight, weight_min, weight_max, weight_average, max_thickness=5):
        mask = weight >= weight_average
        min_avg_gap = (weight_average - weight_min)
        max_avg_gap = (weight_max - weight_average)
        weight -= weight_average
        max_mask = mask / max_avg_gap
        min_mask = ~mask / min_avg_gap
        weight *= max_mask + min_mask
        colors = np.array([
                              [(130 - 125 * n, 130, 130 + 125 * n) for n in line] for line in weight
                              ])
        thicknesses = np.array([
                                   [int((max_thickness - 1) * abs(n)) + 1 for n in line] for line in weight
                                   ])
        max_thickness = int(max_thickness)
        if max_thickness <= 1:
            max_thickness = 2
        if np.sum(thicknesses == max_thickness) == thicknesses.shape[0] * thicknesses.shape[1]:
            thicknesses = np.ones(thicknesses.shape, dtype=np.uint8)
        return colors, thicknesses

    @staticmethod
    def get_graphs_from_logs():
        with open("Results/logs.dat", "rb") as file:
            logs = pickle.load(file)
        for (hus, ep, bt), log in logs.items():
            hus = list(map(lambda _c: str(_c), hus))
            title = "hus: {} ep: {} bt: {}".format(
                "- " + " -> ".join(hus) + " -", ep, bt
            )
            fb_log, acc_log = log["fb_log"], log["acc_log"]
            xs = np.arange(len(fb_log)) + 1
            plt.figure()
            plt.title(title)
            plt.plot(xs, fb_log)
            plt.plot(xs, acc_log, c="g")
            plt.savefig("Results/img/" + "{}_{}_{}".format(
                "-".join(hus), ep, bt
            ))
            plt.close()

    @staticmethod
    def show_img(img, title, normalize=True):
        if normalize:
            img_max, img_min = np.max(img), np.min(img)
            img = 255.0 * (img - img_min) / (img_max - img_min)
        plt.figure()
        plt.title(title)
        plt.imshow(img.astype('uint8'), cmap=plt.cm.gray)
        plt.gca().axis('off')
        plt.show()

    @staticmethod
    def show_batch_img(batch_img, title, normalize=True):
        _n, height, width = batch_img.shape
        a = int(ceil(sqrt(_n)))
        g = np.ones((a * height + a, a * width + a), batch_img.dtype)
        g *= np.min(batch_img)
        _i = 0
        for y in range(a):
            for x in range(a):
                if _i < _n:
                    g[y * height + y:(y + 1) * height + y, x * width + x:(x + 1) * width + x] = batch_img[_i, :, :]
                    _i += 1
        max_g = g.max()
        min_g = g.min()
        g = (g - min_g) / (max_g - min_g)
        VisUtil.show_img(g, title, normalize)

    @staticmethod
    def trans_img(img, shape=None):
        if shape is not None:
            img = img.reshape(shape)
        if img.shape[0] == 1:
            return img.reshape(img.shape[1:])
        return img.transpose(1, 2, 0)


class Util:

    @staticmethod
    def get_and_pop(dic, key, default):
        try:
            val = dic[key]
            dic.pop(key)
        except KeyError:
            val = default
        return val


class ProgressBar:
    def __init__(self, min_value=0, max_value=None, min_refresh_period=0.5, width=30, name=""):
        self._min, self._max = min_value, max_value
        self._task_length = int(max_value - min_value) if (
            min_value is not None and max_value is not None
        ) else None
        self._counter = min_value
        self._min_period = min_refresh_period
        self._bar_width = int(width)
        self._bar_name = " " if not name else " # {:^12s} # ".format(name)
        self._terminated = False
        self._started = False
        self._ended = False
        self._current = 0
        self._clock = 0
        self._cost = 0

    def _flush(self):
        if self._ended:
            return False
        if not self._started:
            print("Progress bar not started yet.")
            return False
        if self._terminated:
            self._cost = time.time() - self._clock
            tmp_hour = int(self._cost / 3600)
            tmp_min = int((self._cost - tmp_hour * 3600) / 60)
            tmp_sec = self._cost % 60
            tmp_avg = self._cost / (self._counter - self._min)
            tmp_avg_hour = int(tmp_avg / 3600)
            tmp_avg_min = int((tmp_avg - tmp_avg_hour * 3600) / 60)
            tmp_avg_sec = tmp_avg % 60
            print(
                "\r" + "##{}({:d} : {:d} -> {:d}) Task Finished. "
                       "Time Cost: {:3d} h {:3d} min {:6.4} s; Average: {:3d} h {:3d} min {:6.4} s ".format(
                            self._bar_name, self._task_length, self._min, self._max,
                            tmp_hour, tmp_min, tmp_sec, tmp_avg_hour, tmp_avg_min, tmp_avg_sec
                        ) + " ##\n", end=""
            )
            self._ended = True
            return True
        if self._counter >= self._max:
            self._terminated = True
            return self._flush()
        if self._counter != self._min and time.time() - self._current <= self._min_period:
            return False
        self._current = time.time()
        self._cost = time.time() - self._clock
        if self._counter > self._min:
            tmp_hour = int(self._cost / 3600)
            tmp_min = int((self._cost - tmp_hour * 3600) / 60)
            tmp_sec = self._cost % 60
            tmp_avg = self._cost / (self._counter - self._min)
            tmp_avg_hour = int(tmp_avg / 3600)
            tmp_avg_min = int((tmp_avg - tmp_avg_hour * 3600) / 60)
            tmp_avg_sec = tmp_avg % 60
        else:
            print()
            tmp_hour = 0
            tmp_min = 0
            tmp_sec = 0
            tmp_avg_hour = 0
            tmp_avg_min = 0
            tmp_avg_sec = 0
        passed = int(self._counter * self._bar_width / self._max)
        print("\r" + "##{}[".format(
            self._bar_name
        ) + "-" * passed + " " * (self._bar_width - passed) + "] : {} / {}".format(
            self._counter, self._max
        ) + " ##  Time Cost: {:3d} h {:3d} min {:6.4} s; Average: {:3d} h {:3d} min {:6.4} s ".format(
            tmp_hour, tmp_min, tmp_sec, tmp_avg_hour, tmp_avg_min, tmp_avg_sec
        ) if self._counter != self._min else "##{}Progress bar initialized  ##".format(
            self._bar_name), end=""
        )
        return True

    def set_min(self, min_val):
        if self._max is not None:
            if self._max <= min_val:
                print("Target min_val: {} is larger than current max_val: {}".format(min_val, self._max))
                return
            self._task_length = self._max - min_val
        self._counter = self._min = min_val

    def set_max(self, max_val):
        if self._min is not None:
            if self._min >= max_val:
                print("Target max_val: {} is smaller than current min_val: {}".format(max_val, self._min))
                return
            self._task_length = max_val - self._min
        self._max = max_val

    def update(self, new_value=None):
        if new_value is None:
            new_value = self._counter + 1
        if new_value != self._min:
            self._counter = self._max if new_value >= self._max else int(new_value)
            return self._flush()

    def start(self):
        if self._task_length is None:
            print("Error: Progress bar not initialized properly.")
            return
        self._current = self._clock = time.time()
        self._started = True
        self._flush()


class Timing:
    _timings = {}
    _enabled = False

    def __init__(self, enabled=False):
        Timing._enabled = enabled

    @staticmethod
    def timeit(level=0, name=None, cls_name=None, prefix="[Private Method] "):
        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            if not Timing._enabled:
                return func(*args, **kwargs)
            if instance is not None:
                instance_name = "{:>18s}".format(str(instance))
            else:
                instance_name = " " * 18 if cls_name is None else "{:>18s}".format(cls_name)
            _prefix = "{:>26s}".format(prefix)
            func_name = "{:>28}".format(func.__name__ if name is None else name)
            _name = instance_name + _prefix + func_name
            _t = time.time()
            rs = func(*args, **kwargs)
            _t = time.time() - _t
            try:
                Timing._timings[_name]["timing"] += _t
                Timing._timings[_name]["call_time"] += 1
            except KeyError:
                Timing._timings[_name] = {
                    "level": level,
                    "timing": _t,
                    "call_time": 1
                }
            return rs

        return wrapper

    @property
    def timings(self):
        return self._timings

    def show_timing_log(self, level):
        print()
        print("=" * 110 + "\n" + "Timing log\n" + "-" * 110)
        if not self.timings:
            print("None")
        else:
            for key in sorted(self.timings.keys()):
                timing_info = self.timings[key]
                if level >= timing_info["level"]:
                    print("{:<42s} :  {:12.7} s (Call Time: {:6d})".format(
                        key, timing_info["timing"], timing_info["call_time"]))
        print("-" * 110)
