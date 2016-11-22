# encoding: utf8

from Util import *
from Networks import *

np.random.seed(142857)  # for reproducibility


def main():

    log = ""

    def precision(_y, y_pred):
        y_true, y_pred = np.argmax(_y, axis=1), np.argmax(y_pred, axis=1)
        tp = np.sum(y_true * y_pred)
        if tp == 0:
            return .0
        fp = np.sum((1 - y_true) * y_pred)
        return tp / (tp + fp)

    def recall(_y, y_pred):
        y_true, y_pred = np.argmax(_y, axis=1), np.argmax(y_pred, axis=1)
        tp = np.sum(y_true * y_pred)
        if tp == 0:
            return .0
        fn = np.sum(y_true * (1 - y_pred))
        return tp / (tp + fn)

    nn = NN()
    save = True
    load = False
    debug = False
    show_loss = True
    train_only = True
    apply_bias = True
    whether_gen_xor = False
    whether_gen_spin = True
    whether_gen_random = False
    custom_data_scale = 10 ** 0
    visualize = False
    draw_network = False
    draw_detailed_network = False
    weight_average = None
    draw_weights = False
    show_figure = True
    print_log = False
    do_log = True
    data_path = None
    # data_path = "Data/Training Set/data.txt"

    lr = 0.01
    lb = 0.001
    epoch = 1000
    batch_size = 512
    record_period = 100
    optimizer = "Adam"
    # w_optimizer, b_optimizer = "Adam", "Adam"
    w_optimizer, b_optimizer = None, None

    timing = Timing(enabled=True)
    timing_level = 1
    nn.feed_timing(timing)

    if whether_gen_xor:
        gen_xor(10 ** 2, custom_data_scale, data_path)
    elif whether_gen_spin:
        gen_spin(10 ** 2, 7, data_path)
    elif whether_gen_random:
        gen_random(10 ** 2, custom_data_scale, CLASSES_NUM)
    x, y = get_and_cache_data(data_path)

    if not load:

        # nn.add(Softmax((x.shape[1], y.shape[1])))

        # nn.build([x.shape[1], y.shape[1]])

        # nn.add(Sigmoid((x.shape[1], 24)))
        nn.add("Sigmoid", (x.shape[1], 24))
        nn.add("Sigmoid", (24, ))
        nn.add("Normalize")
        nn.add("Sigmoid", (24, ))
        nn.add(Softmax((y.shape[1], )))

        # nn.layer_names = ["Tanh", "Softmax"]
        # nn.layer_shapes = [(x.shape[1], 48), (y.shape[1], )]
        # nn.build()

        # nn.build([x.shape[1], 48, y.shape[1]])

        nn.preview()

        logs = (
            nn.fit(x, y, optimizer=optimizer, w_optimizer=w_optimizer, b_optimizer=b_optimizer, lr=lr, lb=lb,
                   epoch=epoch, record_period=record_period, batch_size=batch_size,
                   metrics=["acc", "f1", precision, recall], apply_bias=apply_bias,
                   show_loss=show_loss, train_only=train_only,
                   do_log=do_log, print_log=print_log, debug=debug, visualize=visualize,
                   draw_network=draw_network, draw_detailed_network=draw_detailed_network,
                   draw_weights=draw_weights, weight_average=weight_average))

        acc_log, f1_log, precision_log, recall_log, loss_log = logs
        nn.do_visualization()
        if save:
            nn.save()

        if do_log:

            test_fb, test_acc, test_precision, test_recall = (
                f1_log.pop(), acc_log.pop(), precision_log.pop(), recall_log.pop())
            if show_loss:
                loss_log.pop()

            xs = np.arange(len(f1_log)) + 1
            plt.figure()
            plt.plot(xs, acc_log, label="accuracy")
            plt.plot(xs, f1_log, c="c", label="f1 score")
            plt.plot(xs, precision_log, c="g", label="precision")
            plt.plot(xs, recall_log, c="y", label="recall")
            plt.title("Boosted: {}".format(BOOST_LESS_SAMPLES))
            if show_figure:
                plt.legend()
                plt.show()
                if show_loss:
                    plt.figure()
                    plt.plot(xs, loss_log, c="r", label="loss")
                    plt.legend()
                    plt.show()

            log += "CV set Accuracy    : {:12.6} %".format(100 * acc_log[-1]) + "\n"
            log += "CV set F1 Score    : {:12.6}".format(f1_log[-1]) + "\n"
            log += "CV set Precision   : {:12.6}".format(precision_log[-1]) + "\n"
            log += "CV set Recall      : {:12.6}".format(recall_log[-1]) + "\n"
            log += "Test set Accuracy  : {:12.6} %".format(100 * test_acc) + "\n"
            log += "Test set F1 Score  : {:12.6}".format(test_fb) + "\n"
            log += "Test set Precision : {:12.6}".format(test_precision) + "\n"
            log += "Test set Recall    : {:12.6}".format(test_recall) + "\n"

            print()
            print("=" * 30 + "\n" + "Results\n" + "-" * 30)
            print(log)

    else:

        nn.load("Models/Model.nn")
        nn.preview()
        nn.feed(x, y)
        nn.fit(epoch=200, draw_detailed_network=True, train_only=True)
        nn.do_visualization()
        if save:
            nn.save()

        if do_log:

            f1, acc, _precision, _recall = nn.evaluate(x, y, metrics=["f1", "acc", precision, recall])
            log += "Test set Accuracy  : {:12.6} %".format(100 * acc) + "\n"
            log += "Test set F1 Score  : {:12.6}".format(f1) + "\n"
            log += "Test set Precision : {:12.6}".format(_precision) + "\n"
            log += "Test set Recall    : {:12.6}".format(_recall)

            print()
            print("=" * 30 + "\n" + "Results\n" + "-" * 30)
            print(log)

    show_timing_log(timing, timing_level)

if __name__ == '__main__':
    main()
