from NN.TF.Networks import *

from Util.Util import DataUtil


def main():
    log = ""

    nn = NNDist()
    save = False
    load = False
    show_loss = True
    train_only = False
    do_log = True
    verbose = 4

    lr = 0.001
    lb = 0.001
    epoch = 10
    record_period = 1
    weight_scale = 0.001
    optimizer = "Adam"
    nn.optimizer = optimizer

    x, y = DataUtil.get_dataset("cifar10", "../../../_Data/cifar10.txt", quantized=True, one_hot=True)

    draw = True
    img_shape = (3, 32, 32)
    x = x.reshape(len(x), *img_shape)

    if not load:
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
        nn.add("ReLU", (512, ))
        nn.add("Identical", (64, ), apply_bias=False)
        nn.add("Normalize", activation="ReLU")
        nn.add("Dropout")
        nn.add("CrossEntropy", (y.shape[1], ))

        nn.preview()
        nn.fit(x, y,
               lr=lr, lb=0, epoch=epoch, weight_scale=weight_scale,
               record_period=record_period, show_loss=show_loss, train_only=train_only,
               do_log=do_log, verbose=verbose)
        nn.draw_results()

        if save:
            nn.save()
        if draw:
            # nn.draw_conv_weights()
            nn.draw_conv_series(x[:3], img_shape)
    else:
        nn.load()
        print("Optimizer: " + nn.optimizer)
        nn.preview()
        nn.fit(x, y, epoch=1, lr=lr, lb=lb, verbose=verbose)
        # nn.fit(x, y, x_cv, y_cv, x_test, y_test, epoch=1, lr=lr, lb=lb, verbose=verbose)
        if draw:
            # nn.draw_conv_weights()
            nn.draw_conv_series(x[:3], img_shape)
        nn.draw_results()

        acc = nn.evaluate(x, y)[0]
        log += "Test set Accuracy  : {:12.6} %".format(100 * acc) + "\n"
        print("=" * 30 + "\n" + "Results\n" + "-" * 30)
        print(log)

    nn.show_timing_log()

if __name__ == '__main__':
    main()
