from e_SVM.SVM import *

from Util.Util import DataUtil
from Util.Bases import ClassifierBase
from Util.Metas import SklearnCompatibleMeta

from sklearn.svm import SVC

np.random.seed(142857)  # for reproducibility


class SklearnSVM(SVC, ClassifierBase, metaclass=SklearnCompatibleMeta):
    pass


def main():

    x, y = DataUtil.gen_spin(10, 4, 2, 1)
    y = np.array([-1 if yy[0] == 0 else 1 for yy in y])

    svm = SVM()
    svm.fit(x, y)
    svm.estimate(x, y)
    svm.visualize2d(x, y, dense=300)

    svm = SklearnSVM()
    svm.fit(x, y)
    svm.estimate(x, y)
    svm.visualize2d(x, y, dense=300)

    svm.show_timing_log()


if __name__ == '__main__':
    main()
