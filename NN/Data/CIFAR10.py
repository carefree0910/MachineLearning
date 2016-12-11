from NN import *

np.random.seed(142857)  # for reproducibility


def get_cifar10():

    import os
    import pickle

    def load_cifar_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding="latin")
            _x = datadict['data']
            _y = datadict['labels']
            _x = _x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            _y = np.array(_y)
            return _x, _y

    def load_cifar10(root):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(root, 'data_batch_%d' % (b,))
            _x, _y = load_cifar_batch(f)
            xs.append(_x)
            ys.append(_y)
        _x = np.c_[xs].transpose(0, 3, 1, 2)
        _y = np.c_[ys]
        _y = np.array([[1 if i == yy else 0 for i in range(10)] for yy in _y])
        with open("cifar10.dat", "wb") as file:
            pickle.dump((_x, _y), file)
        with open("mini_cifar10.dat", "wb") as file:
            pickle.dump((_x[:100], _y[:100]), file)

    load_cifar10("cifar-10-batches-py")


def main():
    log = ""

    nn = NN()
    save = True
    load = False
    show_loss = True
    train_only = False
    visualize = False
    verbose = 2

    lr = 0.001
    lb = 0.001
    epoch = 10
    record_period = 1
    weight_scale = 0.001
    optimizer = "Adam"
    nn.optimizer = optimizer

    timing = Timing(enabled=True)
    timing_level = 1

    import pickle

    with open("mini_cifar10.dat", "rb") as file:
        x, y = pickle.load(file)
    # x.shape = (len(x), -1)
    x_cv, y_cv, x_test, y_test = x[:1000], y[:1000], x[1000:2000], y[1000:2000]
    x, y = x[2000:], y[2000:]

    draw = True
    img_shape = (3, 32, 32)

    if not load:

        nn.add("ConvReLU", (x.shape[1:], (9, 5, 5)), padding=1)
        nn.add("ConvReLU", ((9, 5, 5), ), padding=1)
        nn.add("MaxPool", ((2, 2), ), 2)
        nn.add("ConvNorm")
        nn.add("ConvReLU", ((16, 3, 3), ))
        nn.add("ConvReLU", ((16, 3, 3), ))
        nn.add("MaxPool", ((2, 2), ), 2)
        nn.add("ConvNorm", momentum=0.8)
        nn.add("ReLU", (400, ))
        nn.add("CrossEntropy", (y.shape[1], ))

        nn.preview()
        nn.feed_timing(timing)

        nn.fit(x, y, x_cv, y_cv, x_test, y_test,
               lr=lr, lb=lb, batch_size=256, epoch=epoch, weight_scale=weight_scale,
               record_period=record_period, show_loss=show_loss, train_only=train_only,
               do_log=True, verbose=verbose, visualize=visualize)
        nn.draw_results()

        if save:
            nn.save()
        if draw:
            # nn.draw_conv_weights()
            nn.draw_conv_series(x[:3], img_shape)

    else:

        nn.load("Models/Model.nn")
        nn.feed(x, y)
        print("Optimizer: " + nn.optimizer)
        nn.preview()
        if visualize:
            nn.visualize_2d()
        if draw:
            # nn.draw_conv_weights()
            nn.draw_conv_series(x[:3], img_shape)
        nn.draw_results()

        acc = nn.evaluate(x, y)[0]
        log += "Test set Accuracy  : {:12.6} %".format(100 * acc) + "\n"
        print("=" * 30 + "\n" + "Results\n" + "-" * 30)
        print(log)

    timing.show_timing_log(timing_level)


if __name__ == '__main__':
    main()
