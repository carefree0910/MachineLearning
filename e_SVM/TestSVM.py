import matplotlib.pyplot as plt

from e_SVM.SVM import SVM
from _SKlearn.SVM import SKSVM

from Util.Util import DataUtil


def main():

    # # # x, y = DataUtil.gen_xor(100, one_hot=False)
    # x, y = DataUtil.gen_spin(20, 4, 2, 2, one_hot=False)
    # # x, y = DataUtil.gen_two_clusters(n_dim=2, one_hot=False)
    # y[y == 0] = -1
    # #
    # # svm = SKSVM(max_iter=10 ** 5, tol=1e-8)
    # # # svm = SKSVM(kernel="poly", degree=12, max_iter=10 ** 5, tol=1e-8)
    # # svm.fit(x, y)
    # # svm.estimate(x, y)
    # # svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm.support_)
    #
    # svm = SVM()
    # # svm.fit(x, y, kernel="rbf", epoch=10 ** 5)
    # # svm.fit(x, y, kernel="poly", p=12, epoch=10 ** 5)
    # _logs = [_log[0] for _log in svm.fit(
    #     x, y, kernel="poly", p=12, metrics=["acc"], epoch=10 ** 5
    # )]
    # svm.estimate(x, y)
    # svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm["alpha"] > 0)

    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", train_num=100, quantize=True, tar_idx=0)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    svm = SKSVM(kernel="poly", max_iter=10 ** 4)
    svm.fit(x_train, y_train)
    svm.estimate(x_train, y_train)
    svm.estimate(x_test, y_test)

    svm = SVM()
    _logs = [_log[0] for _log in svm.fit(
        x_train, y_train, kernel="poly", metrics=["acc"], x_test=x_test, y_test=y_test
    )]
    svm.estimate(x_train, y_train)
    svm.estimate(x_test, y_test)

    plt.figure()
    plt.title(svm.title)
    plt.plot(range(len(_logs)), _logs)
    plt.show()

    svm.show_timing_log()

if __name__ == '__main__':
    main()
