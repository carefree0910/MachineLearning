from NN.TF.Networks import *

from Util.Util import DataUtil


def main():
    x, y = DataUtil.get_dataset("mnist", "../../../_Data/mnist.txt", quantized=True, one_hot=True)
    x = x.reshape(len(x), 1, 28, 28)

    nn = NNDist()

    # nn.add("ReLU", (x.shape[1], 24))
    # nn.add("ReLU", (24, ))
    # nn.add("CrossEntropy", (y.shape[1], ))

    nn.add("ConvReLU", (x.shape[1:], (32, 3, 3)))
    nn.add("ConvReLU", ((32, 3, 3),))
    nn.add("MaxPool", ((3, 3),), 2)
    nn.add("ConvNorm")
    nn.add("ConvDrop")
    nn.add("ConvReLU", ((64, 3, 3),), std=0.01)
    nn.add("ConvReLU", ((64, 3, 3),), std=0.01)
    nn.add("AvgPool", ((3, 3),), 2)
    nn.add("ConvNorm")
    nn.add("ConvDrop")
    nn.add("ConvReLU", ((32, 3, 3),))
    nn.add("ConvReLU", ((32, 3, 3),))
    nn.add("AvgPool", ((3, 3),), 2)
    nn.add("ReLU", (512,))
    nn.add("Identical", (64,))
    nn.add("Normalize", activation="ReLU")
    nn.add("Dropout")
    nn.add("CrossEntropy", (y.shape[1],))

    nn.optimizer = "Adam"
    nn.preview()
    nn.fit(x, y, verbose=2, do_log=True)
    nn.evaluate(x, y)
    nn.draw_results()
    nn.show_timing_log()

if __name__ == '__main__':
    main()
