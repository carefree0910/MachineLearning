from Util import *
from NeuralNetwork import *


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
    save = False
    load = False
    debug = False
    apply_bias = False
    whether_gen_xor = False
    xor_scale = 10 ** -1
    visualize = True

    if whether_gen_xor:
        gen_xor(10 ** 4, xor_scale)
    x, y = get_and_cache_data()

    train_clock = time.time()

    if not load:

        # nn.add(Sigmoid((x.shape[1], y.shape[1])))

        # nn.build([x.shape[1], y.shape[1]])

        nn.add(Sigmoid((x.shape[1], 6)))
        nn.add(Sigmoid((6, )))
        nn.add(Sigmoid((y.shape[1], )))
        # nn.add("Dropout")

        # nn.build([x.shape[1], 6, 6, y.shape[1]])

        nn.preview()

        (acc_log, f1_log, precision_log, recall_log) = (
            nn.fit(x, y, metrics=["acc", "f1", precision, recall],
                   apply_bias=apply_bias, print_log=True, debug=debug, visualize=visualize))
        (test_fb, test_acc, test_precision, test_recall) = (
            f1_log.pop(), acc_log.pop(), precision_log.pop(), recall_log.pop())

        if save:
            nn.save()

        train_clock = time.time() - train_clock

        draw_clock = time.time()
        xs = np.arange(len(f1_log)) + 1
        plt.figure()
        plt.plot(xs, acc_log, label="accuracy")
        plt.plot(xs, f1_log, c="g", label="f1 score")
        plt.plot(xs, precision_log, c="r", label="precision")
        plt.plot(xs, recall_log, c="y", label="recall")
        plt.title("Boosted: {}".format(BOOST_LESS_SAMPLES))
        draw_clock = time.time() - draw_clock
        if SHOW_FIGURE:
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
        log += "Training time      : {:12.6} s".format(train_clock) + "\n"
        log += "Drawing time       : {:12.6} s".format(draw_clock) + "\n"

        print()
        print("=" * 30 + "\n" + "Results\n" + "-" * 30)
        print(log)

    else:

        nn.load("Models/NN_Model")
        nn.preview()
        print(nn.evaluate(x, y))

if __name__ == '__main__':
    main()
