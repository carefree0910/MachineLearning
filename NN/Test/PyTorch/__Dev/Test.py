from NN.PyTorch.__Dev.Networks import *

from Util.Util import DataUtil


def main():
    nn = NNDist()
    save = False
    load = False
    show_loss = True
    train_only = False
    verbose = 2

    lr = 0.001
    lb = 0.001
    epoch = 5
    record_period = 1
    use_cnn = False

    x, y = DataUtil.get_dataset("mnist", "../../../../_Data/mnist.txt", quantized=True, one_hot=True)
    if use_cnn:
        x = x.reshape(len(x), 1, 28, 28)
        batch_size = 32
    else:
        batch_size = 128

    if not load:
        if use_cnn:
            nn.add("ConvReLU", (x.shape[1:], (16, 3, 3)))
            nn.add("MaxPool", ((3, 3),), 1)
            nn.add("ConvNorm")
            nn.add("ConvDrop")
        else:
            nn.add("ReLU", (x.shape[1], 1024))
            nn.add("ReLU", (1024,))
        nn.add("CrossEntropy", (y.shape[1],))
        nn.optimizer = "Adam"
        nn.preview()
        nn.fit(x, y, lr=lr, lb=lb,
               epoch=epoch, batch_size=batch_size, record_period=record_period,
               show_loss=show_loss, train_only=train_only, do_log=True, verbose=verbose)
        if save:
            nn.save()
        nn.draw_results()
    else:
        nn.load()
        nn.preview()
        print(nn.evaluate(x, y)[0])

    nn.show_timing_log()

if __name__ == '__main__':
    main()
