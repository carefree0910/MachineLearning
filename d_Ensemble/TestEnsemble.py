import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import time

from d_Ensemble.AdaBoost import AdaBoost
from d_Ensemble.RandomForest import RandomForest
from _SKlearn.Ensemble import SKAdaBoost, SKRandomForest

from Util.Util import DataUtil

clf_dict = {
    "AdaBoost": AdaBoost, "RF": RandomForest,
    "SKAdaBoost": SKAdaBoost, "SKRandomForest": SKRandomForest
}


def test(x, y, algorithm="AdaBoost", clf="Cart", epoch=10, **kwargs):
    ensemble = clf_dict[algorithm]()
    if "SK" in algorithm:
        ensemble.fit(x, y)
    else:
        ensemble.fit(x, y, None, clf, epoch, **kwargs)
    ensemble.visualize2d(x, y, **kwargs)
    ensemble.evaluate(x, y, **kwargs)


def cv_test(x, y, xt, yt, algorithm="AdaBoost", clf="Cart", epoch=10, **kwargs):
    print("=" * 30)
    print("Testing {} ({})...".format(algorithm, clf))
    print("-" * 30)
    t = time.time()
    ensemble = clf_dict[algorithm]()
    ensemble.fit(x, y, None, clf, epoch, **kwargs)
    ensemble.evaluate(xt, yt)
    print("Time cost: {:8.6} s".format(time.time() - t))

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
    _x, _y = DataUtil.gen_spiral(size=20, n=4, n_class=2, one_hot=False)
    _y[_y == 0] = -1
    # test(_x, _y, clf="SKTree", epoch=10)
    # test(_x, _y, clf="SKTree", epoch=1000)
    # test(_x, _y, algorithm="RF", epoch=10)
    test(_x, _y, algorithm="RF", epoch=30, n_cores=4)
    test(_x, _y, algorithm="SKAdaBoost")

    train_num = 6000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", n_train=train_num, quantize=True, tar_idx=0)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=1)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=5)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=10)
    cv_test(x_train, y_train, x_test, y_test, clf="MNB", epoch=15)

    AdaBoost().show_timing_log()
