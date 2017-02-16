import time
import numpy as np
from math import pi

from d_Ensemble.AdaBoost import AdaBoost
from d_Ensemble.RandomForest import RandomForest
from _SKlearn.Ensemble import SKAdaBoost, SKRandomForest

from Util.Util import DataUtil

_clf_dic = {
    "AdaBoost": AdaBoost, "RF": RandomForest,
    "SKAdaBoost": SKAdaBoost, "SKRandomForest": SKRandomForest
}


def gen_random(size=100):
    xy = np.random.rand(size, 2)
    z = np.random.randint(2, size=size)
    z[z == 0] = -1
    return xy, z


def gen_xor(size=100):
    x = np.random.randn(size)
    y = np.random.randn(size)
    z = np.ones(size)
    z[x * y < 0] = -1
    return np.c_[x, y].astype(np.float32), z


def gen_spin(size=30):
    xs = np.zeros((size * 4, 2), dtype=np.float32)
    ys = np.zeros(size * 4, dtype=np.int8)
    for i in range(4):
        ix = range(size * i, size * (i + 1))
        r = np.linspace(0.0, 1, size + 1)[1:]
        t = np.linspace(2 * i * pi / 4, 2 * (i + 4) * pi / 4, size) + np.random.random(size=size) * 0.1
        xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        ys[ix] = 2 * (i % 2) - 1
    return xs, ys


def test(x, y, algorithm="AdaBoost", clf="Cart", epoch=10, **kwargs):
    ensemble = _clf_dic[algorithm]()
    if "SK" in algorithm:
        ensemble.fit(x, y)
    else:
        ensemble.fit(x, y, clf, epoch, **kwargs)
    ensemble.visualize2d(x, y)
    ensemble.estimate(x, y)


def cv_test(x, y, xt, yt, algorithm="AdaBoost", clf="Cart", epoch=10, **kwargs):
    print("=" * 30)
    print("Testing {} ({})...".format(algorithm, clf))
    print("-" * 30)
    _t = time.time()
    ensemble = _clf_dic[algorithm]()
    ensemble.fit(x, y, clf, epoch, **kwargs)
    ensemble.estimate(xt, yt)
    print("Time cost: {:8.6} s".format(time.time() - _t))

if __name__ == '__main__':
    # _x, _y = gen_random()
    # test(_x, _y, algorithm="RF", epoch=1)
    # test(_x, _y, algorithm="RF", epoch=10)
    # test(_x, _y, algorithm="RF", epoch=50)
    # test(_x, _y, algorithm="SKRandomForest")
    # test(_x, _y, epoch=1)
    # test(_x, _y, epoch=1)
    # test(_x, _y, epoch=10)
    # _x, _y = gen_xor()
    # test(_x, _y, algorithm="RF", epoch=1)
    # test(_x, _y, algorithm="RF", epoch=10)
    # test(_x, _y, algorithm="RF", epoch=1000)
    # test(_x, _y, algorithm="SKAdaBoost")
    # _x, _y = gen_spin()
    # test(_x, _y, algorithm="RF", epoch=10)
    # test(_x, _y, algorithm="RF", epoch=1000)
    # test(_x, _y, algorithm="SKAdaBoost")

    _x, _y, *_ = DataUtil.get_dataset("mushroom", "../_Data/mushroom.txt", tar_idx=0, quantize=True)
    _y[_y == 0] = -1
    train_num = 6000
    x_train = _x[:train_num]
    y_train = _y[:train_num]
    x_test = _x[train_num:]
    y_test = _y[train_num:]

    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=1)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=5)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=10)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=15)

    AdaBoost().show_timing_log()
