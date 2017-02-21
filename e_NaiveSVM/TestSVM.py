from e_NaiveSVM.SVM import SVM
from _SKlearn.SVM import SKSVM

from Util.Util import DataUtil


def main():

    # x, y = DataUtil.gen_spin(20, 4, 2, 2, one_hot=False)
    # x, y = DataUtil.gen_two_clusters(n_dim=3, one_hot=False)
    x, y = DataUtil.gen_xor(100, one_hot=False)
    y[y == 0] = -1

    svm = SVM()
    # svm.fit(x, y, kernel="rbf")
    svm.fit(x, y, kernel="poly", p=12, epoch=10 ** 5)
    svm.estimate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm["alpha"] > 0)

    # svm = SKSVM()
    svm = SKSVM(kernel="poly", degree=12, max_iter=10 ** 5, tol=1e-8)
    svm.fit(x, y)
    svm.estimate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm.support_)

    svm.show_timing_log()

if __name__ == '__main__':
    main()
