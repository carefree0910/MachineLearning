import matplotlib.pyplot as plt

from e_SVM.SVM import SVM, TFSVM
from _SKlearn.SVM import SKSVM

from Util.Util import DataUtil


def main():

    # x, y = DataUtil.gen_xor(100, one_hot=False)
    x, y = DataUtil.gen_spiral(20, 4, 2, 2, one_hot=False)
    # x, y = DataUtil.gen_two_clusters(n_dim=2, one_hot=False)
    y[y == 0] = -1
    #
    # svm = SKSVM()
    # # svm = SKSVM(kernel="poly", degree=12)
    # svm.fit(x, y)
    # svm.evaluate(x, y)
    # svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm.support_)
    #
    # svm = TFSVM()
    # svm.fit(x, y, lr=0.0001)
    # svm.evaluate(x, y)
    # svm.visualize2d(x, y, padding=0.1, dense=400)
    #
    svm = SVM()
    svm.fit(x, y, kernel="poly", p=12)
    # _logs = [_log[0] for _log in svm.fit(x, y, metrics=["acc"])]
    svm.evaluate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm["alpha"] > 0)

    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", train_num=100, quantize=True, tar_idx=0)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    svm = SKSVM()
    svm.fit(x_train, y_train)
    svm.evaluate(x_train, y_train)
    svm.evaluate(x_test, y_test)

    svm = TFSVM()
    _logs = [_log[0] for _log in svm.fit(
        x_train, y_train, metrics=["acc"], x_test=x_test, y_test=y_test
    )]
    svm.evaluate(x_train, y_train)
    svm.evaluate(x_test, y_test)

    plt.figure()
    plt.title(svm.title)
    plt.plot(range(len(_logs)), _logs)
    plt.show()

    svm = SVM()
    _logs = [_log[0] for _log in svm.fit(
        x_train, y_train, metrics=["acc"], x_test=x_test, y_test=y_test
    )]
    svm.evaluate(x_train, y_train)
    svm.evaluate(x_test, y_test)

    plt.figure()
    plt.title(svm.title)
    plt.plot(range(len(_logs)), _logs)
    plt.show()

    svm.show_timing_log()

if __name__ == '__main__':
    main()
