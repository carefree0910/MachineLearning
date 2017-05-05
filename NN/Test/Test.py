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
    epoch = 5
    record_period = 1

    timing = Timing()
    timing_level = 1

    x, y = DataUtil.get_dataset("mnist", "../../_Data/mnist.txt", quantized=True, one_hot=True)

    if not load:
        nn.add("ReLU", (x.shape[1], 1024))
        nn.add("ReLU", (1024,))
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
