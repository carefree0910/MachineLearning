from e_SVM.Perceptron import Perceptron
from e_SVM.LinearSVM import LinearSVM

from Util.Util import DataUtil


def main():

    x, y = DataUtil.gen_two_clusters(n_dim=2, dis=2.5, center=5, one_hot=False)
    y[y == 0] = -1

    svm = LinearSVM()
    svm.fit(x, y, epoch=10 ** 5, lr=1e-3)
    svm.evaluate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400)

    perceptron = Perceptron()
    perceptron.fit(x, y)
    perceptron.evaluate(x, y)
    perceptron.visualize2d(x, y)

    perceptron.show_timing_log()

if __name__ == '__main__':
    main()
