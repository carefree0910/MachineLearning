import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from g_CNN.Networks import *

from Util.Util import DataUtil


def main():

    nn = NN()
    epoch = 50

    x, y = DataUtil.get_dataset("cifar10", "../_Data/cifar10.txt", quantized=True, one_hot=True)

    x = x.reshape(len(x), 3, 32, 32)
    nn.add("ConvReLU", (x.shape[1:], (32, 3, 3)))
    nn.add("ConvReLU", ((32, 3, 3),))
    nn.add("MaxPool", ((3, 3),), 2)
    nn.add("ConvNorm")
    nn.add("ConvDrop")
    nn.add("ConvReLU", ((64, 3, 3),))
    nn.add("ConvReLU", ((64, 3, 3),))
    nn.add("AvgPool", ((3, 3),), 2)
    nn.add("ConvNorm")
    nn.add("ConvDrop")
    nn.add("ConvReLU", ((32, 3, 3),))
    nn.add("ConvReLU", ((32, 3, 3),))
    nn.add("AvgPool", ((3, 3),), 2)
    nn.add("ReLU", (512,))
    nn.add("Identical", (64,), apply_bias=False)
    nn.add("Normalize", activation="ReLU")
    nn.add("Dropout")
    nn.add("CrossEntropy", (y.shape[1],))

    # nn.disable_timing()
    nn.fit(x, y, lr=0.001, epoch=epoch, train_rate=0.8,
           metrics=["acc"], record_period=1, verbose=4)
    nn.evaluate(x, y)
    nn.show_timing_log()
    nn.draw_logs()

if __name__ == '__main__':
    main()
