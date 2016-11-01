import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from config import *


def clear_cache():

    def core(path):
        if os.path.isfile(path):
            os.remove(path)

    if CLEAR_CACHE:
        core("Data/cache.dat")


def get_cache():
    try:
        with open("Data/cache.dat", "rb") as file:
            nn_data = pickle.load(file)
        return nn_data
    except FileNotFoundError:
        return None


def do_cache(data):
    with open("Data/cache.dat", "wb") as file:
        pickle.dump(data, file)


def data_cleaning(line):
    # line = line.replace('"', "")
    return list(map(lambda c: c.strip(), line.split(",")))


def get_data():

    categories = None

    x = []
    with open("Data/data.txt", "r") as file:
        flag = None
        for line in file:
            if SKIP_FIRST and flag is None:
                flag = True
                continue

            line = data_cleaning(line)

            tmp_x = []
            if not DATA_CLEANED:
                if categories is None:
                    categories = [{ "flag": 1, _l: 0 } for _l in line]
                for i, _l in enumerate(line):
                    if not WHETHER_NUMERICAL[i]:
                        if _l in categories[i]:
                            tmp_x.append(categories[i][_l])
                        else:
                            tmp_x.append(categories[i]["flag"])
                            categories[i][_l] = categories[i]["flag"]
                            categories[i]["flag"] += 1
                    else:
                        tmp_x.append(float(_l))
            else:
                for i, _l in enumerate(line):
                    if i == TAR_IDX:
                        tmp_x.append(int(_l))
                    elif not WHETHER_EXPAND[i]:
                        tmp_x.append(float(_l))
                    else:
                        _l = int(_l)
                        for _i in range(EXPAND_NUM_LST[i]):
                            if _i == _l - 1:
                                tmp_x.append(1)
                            else:
                                tmp_x.append(0)

            x.append(tmp_x)

    classes_num = categories[TAR_IDX]["flag"] if CLASSES_NUM is None else CLASSES_NUM
    expand_sum = sum(EXPAND_NUM_LST[:TAR_IDX])
    expand_seq = np.array(EXPAND_NUM_LST[:TAR_IDX]) > 0
    assert isinstance(expand_seq, np.ndarray), "Never mind. You'll never see this error"
    expand_num = np.sum(expand_seq)
    expand_total = expand_sum - expand_num
    y = np.array([xx.pop(TAR_IDX + expand_total) for xx in x])
    y = np.array([[0 if i != yy else 1 for i in range(classes_num)] for yy in y])

    return np.array(x), y


def get_and_cache_data():

    clear_cache()
    _data = get_cache()

    if _data is None:
        x, y = get_data()
        do_cache((x, y))
    else:
        x, y = _data

    return x, y


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


def gen_xor(size, scale):
    quarter_size = int(size / 4)
    with open("Data/data.txt", "w") as file:
        seq = np.random.random(size=quarter_size) * scale
        y0 = [(str(_s), str(_s)) for _s in seq]
        y1 = [(str(-_s), str(_s)) for _s in seq]
        y2 = [(str(-_s), str(-_s)) for _s in seq]
        y3 = [(str(_s), str(-_s)) for _s in seq]
        for i, y in enumerate((y0, y1, y2, y3)):
            file.write("\n".join([",".join(_y) + ",{}".format(i % 2) for _y in y]) + "\n")


class ProgressBar:

    def __init__(self, min_value=None, max_value=None, width=30):
        self._min, self._max = min_value, max_value
        self._task_length = int(max_value - min_value) if (
            min_value is not None and max_value is not None
        ) else None
        self._counter = min_value
        self._bar_width = int(width)
        self._terminated = False
        self._started = False
        self._clock = 0
        self._cost = 0

    def _flush(self):

        if not self._started:
            print("Progress bar not started yet.")
        elif self._terminated:
            sys.stdout.write(
                "\r" + "## ({:d} : {:d} -> {:d}) Task Finished. Time cost: {:8.6}; Average: {:8.6}".format(
                    self._task_length, self._min, self._max, self._cost, self._cost / self._task_length
                ) + " ##        "
            )
        else:

            self._cost = time.time() - self._clock
            if self._counter > self._min:
                tmp_hour = int(self._cost / 3600)
                tmp_min = int(self._cost / 60)
                tmp_sec = self._cost % 60
                tmp_avg = self._cost / (self._counter - self._min)
                tmp_avg_hour = int(tmp_avg / 3600)
                tmp_avg_min = int(tmp_avg / 60)
                tmp_avg_sec = tmp_avg % 60
            else:
                tmp_hour = 0
                tmp_min = 0
                tmp_sec = 0
                tmp_avg_hour = 0
                tmp_avg_min = 0
                tmp_avg_sec = 0

            passed = int(self._counter * self._bar_width / self._max)
            sys.stdout.write("\r" + "## [" + "-" * passed + " " * (self._bar_width - passed) + "] : {} / {}".format(
                self._counter, self._max
            ) + " ##   Time Cost: {:3d} h {:4d} min {:8.6} s; Average: {:3d} h {:4d} min {:8.6} s ".format(
                tmp_hour, tmp_min, tmp_sec, tmp_avg_hour, tmp_avg_min, tmp_avg_sec
            ) if self._counter != self._min else "##  Progress bar initialized  ##")

            sys.stdout.flush()

            if self._counter >= self._max:
                self._terminated = True
                self._flush()

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

    def update(self, new_value):
        if new_value != self._min:
            self._counter = self._max if new_value >= self._max else int(new_value)
            self._flush()

    def start(self):
        if self._task_length is None:
            print("Error: Progress bar not initialized properly.")
            return
        self._clock = time.time()
        self._started = True
        self._flush()

if __name__ == '__main__':
    get_graphs_from_logs()
