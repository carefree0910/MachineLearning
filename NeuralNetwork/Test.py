# encoding: utf8

from Util import *
from NeuralNetwork import *

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
    save = True  # Set True to save trained model
    load = False  # Set True to load from pre-trained model
    visualize = True
    data_path = "Data/Homework/data.txt"

    x, y = get_and_cache_data(data_path)

    train_clock = time.time()
    
    if not load:
    
        # nn.add(Softmax((x.shape[1], y.shape[1])))

        # nn.build([x.shape[1], y.shape[1]])

        nn.add(ReLU((x.shape[1], 6)))
        nn.add(Softmax((y.shape[1], )))

        # nn.layer_names = ["Tanh", "Sigmoid"]
        # nn.layer_shapes = [(x.shape[1], 2), (y.shape[1], )]
        # nn.build()

        # nn.build([x.shape[1], 6, 6, y.shape[1]])

        nn.preview()

        logs = (
            nn.fit(x, y, optimizer="Momentum", epoch=40, record_period=8,
                   metrics=["acc", "f1", precision, recall],
                   print_log=True, show_loss=True, train_only=True, visualize=visualize))
        acc_log, f1_log, precision_log, recall_log, loss_log = logs

        test_fb, test_acc, test_precision, test_recall = (
            f1_log.pop(), acc_log.pop(), precision_log.pop(), recall_log.pop())
        loss_log.pop()
            
        if save:
            nn.save()

        train_clock = time.time() - train_clock

        draw_clock = time.time()
        xs = np.arange(len(f1_log)) + 1
        plt.figure()
        plt.plot(xs, acc_log, label="accuracy")
        plt.plot(xs, f1_log, c="c", label="f1 score")
        plt.plot(xs, precision_log, c="g", label="precision")
        plt.plot(xs, recall_log, c="y", label="recall")
        plt.title("Boosted: {}".format(BOOST_LESS_SAMPLES))
        draw_clock = time.time() - draw_clock
        plt.legend()
        plt.show()
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
        log += "Training time      : {:12.6} s".format(train_clock) + "\n"
        log += "Drawing time       : {:12.6} s".format(draw_clock) + "\n"

        print()
        print("=" * 30 + "\n" + "Results\n" + "-" * 30)
        print(log)
        
    else:
        
        nn.load("Models/Model.nn")
        nn.feed(x, y)
        nn.preview()
        nn.do_visualization()

        f1, acc, _precision, _recall = nn.evaluate(x, y, metrics=["f1", "acc", precision, recall])
        log += "Test set Accuracy  : {:12.6} %".format(100 * acc) + "\n"
        log += "Test set F1 Score  : {:12.6}".format(f1) + "\n"
        log += "Test set Precision : {:12.6}".format(_precision) + "\n"
        log += "Test set Recall    : {:12.6}".format(_recall)

        print("=" * 30 + "\n" + "Results\n" + "-" * 30)
        print(log)

if __name__ == '__main__':
    main()
