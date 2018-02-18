import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from e_SVM.Perceptron import Perceptron
from e_SVM.LinearSVM import LinearSVM, TFLinearSVM, TorchLinearSVM

from Util.Util import DataUtil


def main():

    x, y = DataUtil.gen_two_clusters(n_dim=2, dis=2.5, center=5, one_hot=False)
    y[y == 0] = -1

    animation_params = {
        "show": False, "period": 50, "mp4": False,
        "dense": 400, "draw_background": True
    }

    svm = LinearSVM(animation_params=animation_params)
    svm.fit(x, y)
    svm.evaluate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400)

    svm = TFLinearSVM(animation_params=animation_params)
    svm.fit(x, y)
    svm.evaluate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400)

    if TorchLinearSVM is not None:
        svm = TorchLinearSVM(animation_params=animation_params)
        svm.fit(x, y)
        svm.evaluate(x, y)
        svm.visualize2d(x, y, padding=0.1, dense=400)

    perceptron = Perceptron()
    perceptron.fit(x, y)
    perceptron.evaluate(x, y)
    perceptron.visualize2d(x, y, padding=0.1, dense=400)

    perceptron.show_timing_log()


if __name__ == '__main__':
    main()
