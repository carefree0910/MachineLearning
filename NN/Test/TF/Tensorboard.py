from NN.TF.Networks import *

from Util.Util import DataUtil


def main():
    save = False
    load = False
    show_loss = True
    train_only = False
    verbose = 2

    lr = 0.001
    lb = 0.001
    epoch = 10
    record_period = 1

    x, y = DataUtil.get_dataset("mnist", "../../../_Data/mnist.txt", quantized=True, one_hot=True)
    x = x.reshape(len(x), 1, 28, 28)

    if not load:
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
        nn.preview(verbose=verbose)
        nn.fit(x, y, lr=lr, lb=lb,
               epoch=epoch, batch_size=256, record_period=record_period,
               show_loss=show_loss, train_only=train_only, do_log=True, tensorboard_verbose=1, verbose=verbose)
        if save:
            nn.save()
    else:
        nn = NNFrozen()
        nn.load()
        nn.preview()
        nn.evaluate(x, y)

    nn.show_timing_log()

if __name__ == '__main__':
    main()
