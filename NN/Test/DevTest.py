from NN.TF.Networks import *

np.random.seed(142857)  # for reproducibility


def main():
    log = ""

    nn = NNDist()
    save = False
    load = False
    show_loss = True
    train_only = False
    visualize = False
    verbose = 2

    lr = 0.001
    lb = 0.001
    epoch = 50
    record_period = 1

    timing = Timing(enabled=True)
    timing_level = 1

    import pickle

    with open("../Data/mini_cifar10.dat", "rb") as file:
        x, y = pickle.load(file)

    x = x.reshape(len(x), 3, 32, 32)
    # x = x.reshape(len(x), -1)

    if not load:

        def add_layers(_nn):
            _nn.add("Pipe", 3)
            _nn.add_pipe_layer(0, "ConvReLU", ((32, 1, 3),))
            _nn.add_pipe_layer(0, "ConvReLU", ((32, 3, 1),))
            _nn.add_pipe_layer(1, "ConvReLU", ((32, 2, 3),))
            _nn.add_pipe_layer(1, "ConvReLU", ((32, 3, 2),))
            _nn.add_pipe_layer(2, "ConvReLU", ((32, 1, 1),))
            _nn.add_pipe_layer(2, "Pipe", 2)
            _pipe = _nn.get_current_pipe(2)
            _pipe.add_pipe_layer(0, "ConvReLU", ((16, 1, 3),))
            _pipe.add_pipe_layer(1, "ConvReLU", ((16, 3, 1),))

        nn.add("ConvReLU", (x.shape[1:], (32, 3, 3)))
        nn.add("ConvReLU", ((32, 3, 3),))
        nn.add("MaxPool", ((3, 3),), 2)
        nn.add("ConvNorm")
        add_layers(nn)
        nn.add("MaxPool", ((3, 3),), 2)
        nn.add("ConvNorm")
        add_layers(nn)
        nn.add("AvgPool", ((3, 3),), 2)
        nn.add("ConvNorm")
        add_layers(nn)
        nn.add("ReLU", (512,))
        nn.add("ReLU", (64,))
        nn.add("Normalize")
        nn.add("CrossEntropy", (y.shape[1], ))

        nn.optimizer = "Adam"

        nn.preview()
        nn.feed_timing(timing)

        nn.fit(x, y, lr=lr, lb=lb,
               epoch=epoch, batch_size=256, record_period=record_period,
               show_loss=show_loss, train_only=train_only,
               do_log=True, verbose=verbose, visualize=visualize)
        if save:
            nn.save()
        nn.draw_conv_series(x[:3], (3, 32, 32))
        nn.draw_results()

    else:

        nn.load("Models/Model")
        nn.feed(x, y)
        nn.preview()
        nn.fit(epoch=5, lr=lr, lb=lb, verbose=verbose)
        if visualize:
            nn.visualize_2d()
        nn.draw_results()

        acc = nn.evaluate(x, y)[0]
        log += "Whole set Accuracy  : {:12.6} %".format(100 * acc) + "\n"

        print()
        print("=" * 30 + "\n" + "Results\n" + "-" * 30)
        print(log)

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
