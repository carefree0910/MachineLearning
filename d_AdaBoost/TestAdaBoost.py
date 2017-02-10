import time

from d_AdaBoost.AdaBoost import *
from Util import DataUtil


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


def gen_spin(size=20):
    xs = np.zeros((size * 4, 2), dtype=np.float32)
    ys = np.zeros(size * 4, dtype=np.int8)
    for i in range(4):
        ix = range(size * i, size * (i + 1))
        r = np.linspace(0.0, 1, size + 1)[1:]
        t = np.array(
            np.linspace(i * (4 + 1), (i + 1) * (4 + 1), size) +
            np.array(np.random.random(size=size)) * 0.2)
        xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        ys[ix] = 2 * (i % 2) - 1
    return xs, ys


def test(x, y, clf=None, epoch=10, **kwargs):
    ada = AdaBoost()
    ada.fit(x, y, clf, epoch, **kwargs)
    ada.visualize2d(x, y)
    ada.estimate(x, y)
    # ada.draw()


def cv_test(x, y, xt, yt, clf, epoch=10, **kwargs):
    print("=" * 30)
    print("Testing {}...".format(clf))
    print("-" * 30)
    _t = time.time()
    ada = AdaBoost()
    ada.fit(x, y, clf, epoch, **kwargs)
    ada.estimate(xt, yt)
    # ada.visualize()
    print("Time cost: {:8.6} s".format(time.time() - _t))

if __name__ == '__main__':
    # _x, _y = gen_random()
    # test(_x, _y)
    # test(_x, _y, clf="CvDTree")
    # test(_x, _y, clf="CvDTree", max_depth=1)
    # _x, _y = gen_xor()
    # test(_x, _y, clf="Cart")
    # test(_x, _y, clf="CvDTree")
    # test(_x, _y, clf="Cart", max_depth=2, epoch=1)
    # test(_x, _y, clf="CvDTree", max_depth=2, epoch=1)
    # test(_x, _y)
    # test(_x, _y, clf="CvDTree")
    # _x, _y = gen_spin()
    # test(_x, _y, clf="CvDTree")

    _x, _y = DataUtil.get_dataset("mushroom", "../_Data/mushroom.txt", tar_idx=0)
    _dic = {"e": -1, "p": 1}
    _y = np.array([_dic[c] for c in _y])
    _dic = [{_k: i for i, _k in enumerate(set(xx))} for xx in _x]
    train_num = 6000
    x_train = _x[:train_num]
    y_train = _y[:train_num]
    x_test = _x[train_num:]
    y_test = _y[train_num:]
    # cv_test(x_train, y_train, x_test, y_test, clf="Cart", epoch=1, max_depth=1)
    # cv_test(x_train, y_train, x_test, y_test, clf="ID3", epoch=1, max_depth=1)
    # cv_test(x_train, y_train, x_test, y_test, clf="C45", epoch=1, max_depth=1)
    # cv_test(x_train, y_train, x_test, y_test, clf="Cart", max_depth=1)
    # cv_test(x_train, y_train, x_test, y_test, clf="ID3", max_depth=1)
    # cv_test(x_train, y_train, x_test, y_test, clf="C45", max_depth=1)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=1)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=5)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=10)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=15)
