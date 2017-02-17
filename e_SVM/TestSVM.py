from e_SVM.SVM import *
from _SKlearn.SVM import SKSVM

from Util.Util import DataUtil


def main():

    # x, y = DataUtil.gen_spin(20, 4, 2, 2, one_hot=False)
    x, y = DataUtil.gen_two_clusters(one_hot=False)
    y[y == 0] = -1

    svm = SVM()
    # svm.fit(x, y, kernel="gaussian")
    svm.fit(x, y, kernel="poly", p=3)
    svm.estimate(x, y)
    svm.visualize2d(x, y, dense=400)

    svm = SKSVM(kernel="poly", degree=3)
    svm.fit(x, y)
    svm.estimate(x, y)
    svm.visualize2d(x, y, dense=400)

    svm.show_timing_log()


if __name__ == '__main__':
    main()
