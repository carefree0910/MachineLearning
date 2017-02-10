import time

from c_CvDTree.Tree import *
from Util import DataUtil, SklearnCompatibleMeta

from sklearn.tree import DecisionTreeClassifier

np.random.seed(31416)


class SKTree(DecisionTreeClassifier, metaclass=SklearnCompatibleMeta):
    SKTreeTiming = Timing()

    @SKTreeTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, check_input=True, x_idx_sorted=None, *args, **kwargs):
        return DecisionTreeClassifier.fit(self, x, y, sample_weight, check_input, x_idx_sorted)


def main():
    # _x = DataUtil.get_dataset("balloon1.0(en)", "../_Data/balloon1.0(en).txt")
    # np.random.shuffle(_x)
    # _y = [xx.pop() for xx in _x]
    # _x, _y = np.array(_x), np.array(_y)
    # _fit_time = time.time()
    # _tree = C45Tree()
    # _tree.fit(_x, _y, train_only=True)
    # _fit_time = time.time() - _fit_time
    # _tree.view()
    # _estimate_time = time.time()
    # _tree.estimate(_x, _y)
    # _estimate_time = time.time() - _estimate_time
    # print(
    #     "Model building  : {:12.6} s\n"
    #     "Estimation      : {:12.6} s\n"
    #     "Total           : {:12.6} s".format(
    #         _fit_time, _estimate_time,
    #         _fit_time + _estimate_time
    #     )
    # )
    # _tree.visualize()

    # _x = DataUtil.get_dataset("mushroom", "../_Data/mushroom.txt")
    # np.random.shuffle(_x)
    # _y = [xx.pop(0) for xx in _x]
    # _x, _y = np.array(_x), np.array(_y)
    # # _x, _y, _, _, _ = DataUtil.quantize_data(_x, _y)
    # train_num = 6000
    # x_train = _x[:train_num]
    # y_train = _y[:train_num]
    # x_test = _x[train_num:]
    # y_test = _y[train_num:]
    # _fit_time = time.time()
    # _tree = ID3Tree()
    # _tree.fit(x_train, y_train)
    # _fit_time = time.time() - _fit_time
    # # _tree.view()
    # _estimate_time = time.time()
    # _tree.estimate(x_train, y_train)
    # _tree.estimate(x_test, y_test)
    # _estimate_time = time.time() - _estimate_time
    # print(
    #     "Model building  : {:12.6} s\n"
    #     "Estimation      : {:12.6} s\n"
    #     "Total           : {:12.6} s".format(
    #         _fit_time, _estimate_time,
    #         _fit_time + _estimate_time
    #     )
    # )
    # _tree.visualize()

    # _x, _y = DataUtil.gen_xor()
    # _y = np.argmax(_y, axis=1)
    # _fit_time = time.time()
    # _tree = ID3Tree()
    # _tree.fit(_x, _y, train_only=True)
    # _fit_time = time.time() - _fit_time
    # # _tree.view()
    # _estimate_time = time.time()
    # _tree.estimate(_x, _y)
    # _estimate_time = time.time() - _estimate_time
    # print(
    #     "Model building  : {:12.6} s\n"
    #     "Estimation      : {:12.6} s\n"
    #     "Total           : {:12.6} s".format(
    #         _fit_time, _estimate_time,
    #         _fit_time + _estimate_time
    #     )
    # )
    # _tree.show_timing_log()
    # _tree.visualize2d(_x, _y)
    # _tree.visualize()

    _wc = [False] * 16
    _continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for _cl in _continuous_lst:
        _wc[_cl] = True

    train_num = 2000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "bank1.0", "../_Data/bank1.0.txt", train_num=train_num, quantize=True)
    _fit_time = time.time()
    _tree = SKTree()
    _tree.fit(x_train, y_train)
    _fit_time = time.time() - _fit_time
    # _tree.view()
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
    _tree.show_timing_log()
    # _tree.visualize()

if __name__ == '__main__':
    main()
