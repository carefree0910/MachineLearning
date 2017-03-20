from Util.Util import DataUtil
from NN.TF.Networks import *


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

    nn = NNDist()
    save = False
    load = False
    do_log = True
    show_loss = True
    train_only = True
    visualize = False
    draw_detailed_network = False
    weight_average = None
    verbose = 1

    lr = 0.01
    lb = 0.001
    epoch = 1000
    batch_size = 512
    record_period = 100

    timing = Timing(enabled=True)
    timing_level = 1
    nn.feed_timing(timing)

    x, y = DataUtil.gen_spin()

    if not load:

        nn.add("ReLU", (x.shape[1], 24))
        nn.add("CrossEntropy", (y.shape[1],))

        nn.preview()

        nn.fit(x, y, lr=lr, lb=lb,
               epoch=epoch, record_period=record_period, batch_size=batch_size,
               # metrics=["acc", "f1", precision, recall],
               show_loss=show_loss, train_only=train_only,
               do_log=do_log, verbose=verbose, visualize=visualize,
               draw_detailed_network=draw_detailed_network, weight_average=weight_average)
        nn.draw_results()
        nn.visualize2d()

        if save:
            nn.save()

    else:

        nn.load("Models/Model")
        nn.preview()
        nn.feed(x, y)
        nn.fit(epoch=20, train_only=True, record_period=20, verbose=2)
        nn.visualize2d()
        nn.draw_results()

        f1, acc, _precision, _recall = nn.evaluate(x, y, metrics=["f1", "acc", precision, recall])
        log += "Test set Accuracy  : {:12.6} %".format(100 * acc) + "\n"
        log += "Test set F1 Score  : {:12.6}".format(f1) + "\n"
        log += "Test set Precision : {:12.6}".format(_precision) + "\n"
        log += "Test set Recall    : {:12.6}".format(_recall)

        print()
        print("=" * 30 + "\n" + "Results\n" + "-" * 30)
        print(log)

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
