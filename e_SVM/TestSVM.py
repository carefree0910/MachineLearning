from e_SVM.SVM import *

from Util.Util import DataUtil


def main():

    x, y = DataUtil.gen_spin(20, 4, 2, 2, one_hot=False)
    y[y == 0] = -1

    svm = SVM()
    # svm.fit(x, y)
    svm.fit(x, y, kernel="poly", p=16)
    svm.estimate(x, y)
    svm.visualize2d(x, y, dense=400)

    # svm = SklearnSVM()
    # svm.fit(x, y)
    # svm.estimate(x, y)
    # svm.visualize2d(x, y, dense=400)

    svm.show_timing_log()


if __name__ == '__main__':
    main()
