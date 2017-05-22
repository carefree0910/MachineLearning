import time

from c_CvDTree.Tree import *

from Util.Util import DataUtil


def main():
    # # x, y = DataUtil.get_dataset("balloon1.0(en)", "../_Data/balloon1.0(en).txt")
    # x, y = DataUtil.get_dataset("test", "../_Data/test.txt")
    # fit_time = time.time()
    # tree = CartTree(whether_continuous=[False] * 4)
    # tree.fit(x, y, train_only=True)
    # fit_time = time.time() - fit_time
    # tree.view()
    # estimate_time = time.time()
    # tree.evaluate(x, y)
    # estimate_time = time.time() - estimate_time
    # print(
    #     "Model building  : {:12.6} s\n"
    #     "Estimation      : {:12.6} s\n"
    #     "Total           : {:12.6} s".format(
    #         fit_time, estimate_time,
    #         fit_time + estimate_time
    #     )
    # )
    # tree.visualize()

    # train_num = 6000
    # (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
    #     "mushroom", "../_Data/mushroom.txt", tar_idx=0, train_num=train_num)
    # fit_time = time.time()
    # tree = C45Tree()
    # tree.fit(x_train, y_train)
    # fit_time = time.time() - fit_time
    # tree.view()
    # estimate_time = time.time()
    # tree.evaluate(x_train, y_train)
    # tree.evaluate(x_test, y_test)
    # estimate_time = time.time() - estimate_time
    # print(
    #     "Model building  : {:12.6} s\n"
    #     "Estimation      : {:12.6} s\n"
    #     "Total           : {:12.6} s".format(
    #         fit_time, estimate_time,
    #         fit_time + estimate_time
    #     )
    # )
    # tree.visualize()

    x, y = DataUtil.gen_xor(one_hot=False)
    fit_time = time.time()
    tree = CartTree()
    tree.fit(x, y, train_only=True)
    fit_time = time.time() - fit_time
    tree.view()
    estimate_time = time.time()
    tree.evaluate(x, y, n_cores=1)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    tree.visualize2d(x, y, dense=1000, n_cores=2)
    tree.visualize()

    wc = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for _cl in continuous_lst:
        wc[_cl] = True

    train_num = 2000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "bank1.0", "../_Data/bank1.0.txt", train_num=train_num, quantize=True)
    fit_time = time.time()
    tree = CartTree()
    tree.fit(x_train, y_train)
    fit_time = time.time() - fit_time
    tree.view()
    estimate_time = time.time()
    tree.evaluate(x_test, y_test)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    tree.visualize()

    tree.show_timing_log()

if __name__ == '__main__':
    main()
