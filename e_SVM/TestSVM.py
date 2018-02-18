import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import matplotlib.pyplot as plt

from e_SVM.SVM import SVM, GDSVM, TFSVM, TorchSVM
from _SKlearn.SVM import SKSVM

from Util.Util import DataUtil


def main():

    # x, y = DataUtil.gen_xor(100, one_hot=False)
    x, y = DataUtil.gen_spiral(20, 4, 2, 2, one_hot=False)
    # x, y = DataUtil.gen_two_clusters(n_dim=2, one_hot=False)
    y[y == 0] = -1

    animation_params = {
        "show": False, "mp4": False, "period": 50,
        "dense": 400, "draw_background": True
    }

    svm = SVM(animation_params=animation_params)
    svm.fit(x, y, kernel="poly", p=12, epoch=600)
    svm.evaluate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm["alpha"] > 0)

    svm = GDSVM(animation_params=animation_params)
    svm.fit(x, y, kernel="poly", p=12, epoch=10000)
    svm.evaluate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm["alpha"] > 0)

    if TorchSVM is not None:
        svm = TorchSVM(animation_params=animation_params)
        svm.fit(x, y, kernel="poly", p=12)
        svm.evaluate(x, y)
        svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm["alpha"] > 0)

    svm = TFSVM()
    svm.fit(x, y)
    svm.evaluate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400)

    svm = SKSVM()
    # svm = SKSVM(kernel="poly", degree=12)
    svm.fit(x, y)
    svm.evaluate(x, y)
    svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm.support_)

    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", n_train=100, quantize=True, tar_idx=0)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    svm = SKSVM()
    svm.fit(x_train, y_train)
    svm.evaluate(x_train, y_train)
    svm.evaluate(x_test, y_test)

    svm = TFSVM()
    svm.fit(x_train, y_train)
    svm.evaluate(x_train, y_train)
    svm.evaluate(x_test, y_test)

    if TorchSVM is not None:
        svm = TorchSVM()
        svm.fit(x_train, y_train)
        svm.evaluate(x_train, y_train)
        svm.evaluate(x_test, y_test)

        svm = TorchSVM()
        logs = [log[0] for log in svm.fit(
            x_train, y_train, metrics=["acc"], x_test=x_test, y_test=y_test
        )]
        svm.evaluate(x_train, y_train)
        svm.evaluate(x_test, y_test)

        plt.figure()
        plt.title(svm.title)
        plt.plot(range(len(logs)), logs)
        plt.show()

    svm = SVM()
    logs = [log[0] for log in svm.fit(
        x_train, y_train, metrics=["acc"], x_test=x_test, y_test=y_test
    )]
    svm.evaluate(x_train, y_train)
    svm.evaluate(x_test, y_test)

    plt.figure()
    plt.title(svm.title)
    plt.plot(range(len(logs)), logs)
    plt.show()

    svm = GDSVM()
    logs = [log[0] for log in svm.fit(
        x_train, y_train, metrics=["acc"], x_test=x_test, y_test=y_test
    )]
    svm.evaluate(x_train, y_train)
    svm.evaluate(x_test, y_test)

    plt.figure()
    plt.title(svm.title)
    plt.plot(range(len(logs)), logs)
    plt.show()

    svm.show_timing_log()


if __name__ == '__main__':
    main()
