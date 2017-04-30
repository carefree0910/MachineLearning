from NN.Basic.Networks import *

from Util.Util import DataUtil


def main():
    nn = NNDist()
    save = False
    load = False
    show_loss = True
    train_only = False
    visualize = False
    verbose = 2

    lr = 0.001
    lb = 0.001
    epoch = 10
    record_period = 1

    timing = Timing(enabled=True)
    timing_level = 1

    x, y = DataUtil.get_dataset("mnist", "../../_Data/mnist.txt", quantized=True, one_hot=True)
    x = x.reshape(len(x), 1, 28, 28)

    if not load:
        nn.add("ConvReLU", (x.shape[1:], (32, 3, 3)))
        nn.add("ConvReLU", ((32, 3, 3),))
        nn.add("MaxPool", ((3, 3),), 1)
        nn.add("ConvNorm")
        nn.add("ConvDrop")
        nn.add("ConvReLU", ((64, 3, 3),))
        nn.add("ConvReLU", ((64, 3, 3),))
        nn.add("MaxPool", ((3, 3),), 1)
        nn.add("ConvNorm")
        nn.add("ConvDrop")
        nn.add("ConvReLU", ((32, 3, 3),))
        nn.add("ConvReLU", ((32, 3, 3),))
        nn.add("MaxPool", ((3, 3),), 1)
        nn.add("ReLU", (512,))
        nn.add("Identical", (64,))
        nn.add("Normalize")
        nn.add("Dropout")
        nn.add("CrossEntropy", (y.shape[1],))
        nn.optimizer = "Adam"
        nn.preview()
        nn.feed_timing(timing)
        nn.fit(x, y, lr=lr, lb=lb,
               epoch=epoch, batch_size=32, record_period=record_period,
               show_loss=show_loss, train_only=train_only,
               do_log=True, verbose=verbose, visualize=visualize)
        if save:
            nn.save()
        nn.draw_results()
    else:
        nn.load()
        nn.preview()
        nn.evaluate(x, y)

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
