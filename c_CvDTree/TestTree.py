import time

from c_CvDTree.Tree import *

from Util.Util import DataUtil


def main():
    _x, _y = DataUtil.get_dataset("balloon1.0(en)", "../_Data/balloon1.0(en).txt")
    _fit_time = time.time()
    _tree = ID3Tree()
    _tree.fit(_x, _y, train_only=True)
    _fit_time = time.time() - _fit_time
    _tree.view()
    _estimate_time = time.time()
    _tree.estimate(_x, _y)
    _estimate_time = time.time() - _estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            _fit_time, _estimate_time,
            _fit_time + _estimate_time
        )
    )
    _tree.visualize()

    train_num = 6000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", tar_idx=0, train_num=train_num)
    _fit_time = time.time()
    _tree = C45Tree()
    _tree.fit(x_train, y_train)
    _fit_time = time.time() - _fit_time
    _tree.view()
    _estimate_time = time.time()
    _tree.estimate(x_train, y_train)
    _tree.estimate(x_test, y_test)
    _estimate_time = time.time() - _estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            _fit_time, _estimate_time,
            _fit_time + _estimate_time
        )
    )
    _tree.visualize()

    _x, _y = DataUtil.gen_xor(one_hot=False)
    _fit_time = time.time()
    _tree = CartTree()
    _tree.fit(_x, _y, train_only=True)
    _fit_time = time.time() - _fit_time
    _tree.view()
    _estimate_time = time.time()
    _tree.estimate(_x, _y)
    _estimate_time = time.time() - _estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            _fit_time, _estimate_time,
            _fit_time + _estimate_time
        )
    )
    _tree.visualize2d(_x, _y)
    _tree.visualize()

    _wc = [False] * 16
    _continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for _cl in _continuous_lst:
        _wc[_cl] = True

    train_num = 2000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "bank1.0", "../_Data/bank1.0.txt", train_num=train_num, quantize=True)
    _fit_time = time.time()
    _tree = CartTree()
    _tree.fit(x_train, y_train)
    _fit_time = time.time() - _fit_time
    _tree.view()
    _estimate_time = time.time()
    _tree.estimate(x_test, y_test)
    _estimate_time = time.time() - _estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            _fit_time, _estimate_time,
            _fit_time + _estimate_time
        )
    )
    _tree.visualize()

    _tree.show_timing_log()

if __name__ == '__main__':
    main()
