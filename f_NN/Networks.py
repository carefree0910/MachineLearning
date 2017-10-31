import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import matplotlib.pyplot as plt

from f_NN.Layers import *
from f_NN.Optimizers import *

from Util.Bases import ClassifierBase
from Util.ProgressBar import ProgressBar


class NNVerbose:
    NONE = 0
    EPOCH = 1
    METRICS = 2
    METRICS_DETAIL = 3
    DETAIL = 4
    DEBUG = 5


class NaiveNN(ClassifierBase):
    NaiveNNTiming = Timing()

    def __init__(self, **kwargs):
        super(NaiveNN, self).__init__(**kwargs)
        self._layers, self._weights, self._bias = [], [], []
        self._w_optimizer = self._b_optimizer = None
        self._current_dimension = 0

        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["epoch"] = kwargs.get("epoch", 10)
        self._params["optimizer"] = kwargs.get("optimizer", "Adam")

    # Utils

    @NaiveNNTiming.timeit(level=4)
    def _add_params(self, shape):
        self._weights.append(np.random.randn(*shape))
        self._bias.append(np.zeros((1, shape[1])))

    @NaiveNNTiming.timeit(level=4)
    def _add_layer(self, layer, *args):
        current, nxt = args
        self._add_params((current, nxt))
        self._current_dimension = nxt
        self._layers.append(layer)

    @NaiveNNTiming.timeit(level=1)
    def _get_activations(self, x):
        activations = [self._layers[0].activate(x, self._weights[0], self._bias[0])]
        for i, layer in enumerate(self._layers[1:]):
            activations.append(layer.activate(
                activations[-1], self._weights[i + 1], self._bias[i + 1]))
        return activations

    @NaiveNNTiming.timeit(level=1)
    def _get_prediction(self, x):
        return self._get_activations(x)[-1]

    # Optimizing Process

    @NaiveNNTiming.timeit(level=4)
    def _init_optimizers(self, optimizer, lr, epoch):
        opt_fac = OptFactory()
        self._w_optimizer = opt_fac.get_optimizer_by_name(
            optimizer, self._weights, lr, epoch)
        self._b_optimizer = opt_fac.get_optimizer_by_name(
            optimizer, self._bias, lr, epoch)

    @NaiveNNTiming.timeit(level=1)
    def _opt(self, i, _activation, _delta):
        self._weights[i] += self._w_optimizer.run(
            i, _activation.T.dot(_delta)
        )
        self._bias[i] += self._b_optimizer.run(
            i, np.sum(_delta, axis=0, keepdims=True)
        )

    # API

    @NaiveNNTiming.timeit(level=4, prefix="[API] ")
    def add(self, layer):
        if not self._layers:
            self._layers, self._current_dimension = [layer], layer.shape[1]
            self._add_params(layer.shape)
        else:
            nxt = layer.shape[0]
            layer.shape = (self._current_dimension, nxt)
            self._add_layer(layer, self._current_dimension, nxt)

    @NaiveNNTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, lr=None, epoch=None, optimizer=None):
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        if optimizer is None:
            optimizer = self._params["optimizer"]
        self._init_optimizers(optimizer, lr, epoch)
        layer_width = len(self._layers)
        for counter in range(epoch):
            self._w_optimizer.update()
            self._b_optimizer.update()
            activations = self._get_activations(x)
            deltas = [self._layers[-1].bp_first(y, activations[-1])]
            for i in range(-1, -len(activations), -1):
                deltas.append(self._layers[i - 1].bp(
                    activations[i - 1], self._weights[i], deltas[-1]
                ))
            for i in range(layer_width - 1, 0, -1):
                self._opt(i, activations[i - 1], deltas[layer_width - i - 1])
            self._opt(0, x, deltas[-1])

    @NaiveNNTiming.timeit(level=4, prefix="[API] ")
    def predict(self, x, get_raw_results=False, **kwargs):
        y_pred = self._get_prediction(np.atleast_2d(x))
        if get_raw_results:
            return y_pred
        return np.argmax(y_pred, axis=1)


class NN(NaiveNN):
    NNTiming = Timing()

    def __init__(self, **kwargs):
        super(NN, self).__init__(**kwargs)
        self._available_metrics = {
            key: value for key, value in zip(["acc", "f1-score"], [NN.acc, NN.f1_score])
        }
        self._metrics, self._metric_names, self._logs = [], [], {}
        self.verbose = None

        self._params["batch_size"] = kwargs.get("batch_size", 256)
        self._params["train_rate"] = kwargs.get("train_rate", None)
        self._params["metrics"] = kwargs.get("metrics", None)
        self._params["record_period"] = kwargs.get("record_period", 100)
        self._params["verbose"] = kwargs.get("verbose", 1)

    # Utils

    @NNTiming.timeit(level=1)
    def _get_prediction(self, x, name=None, batch_size=1e6, verbose=None):
        if verbose is None:
            verbose = self.verbose
        single_batch = batch_size / np.prod(x.shape[1:])  # type: float
        single_batch = int(single_batch)
        if not single_batch:
            single_batch = 1
        if single_batch >= len(x):
            return self._get_activations(x).pop()
        epoch = int(len(x) / single_batch)
        if not len(x) % single_batch:
            epoch += 1
        name = "Prediction" if name is None else "Prediction ({})".format(name)
        sub_bar = ProgressBar(max_value=epoch, name=name, start=False)
        if verbose >= NNVerbose.METRICS:
            sub_bar.start()
        rs, count = [self._get_activations(x[:single_batch]).pop()], single_batch
        if verbose >= NNVerbose.METRICS:
            sub_bar.update()
        while count < len(x):
            count += single_batch
            if count >= len(x):
                rs.append(self._get_activations(x[count - single_batch:]).pop())
            else:
                rs.append(self._get_activations(x[count - single_batch:count]).pop())
            if verbose >= NNVerbose.METRICS:
                sub_bar.update()
        return np.vstack(rs)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def _preview(self):
        if not self._layers:
            rs = "None"
        else:
            rs = (
                "Input  :  {:<10s} - {}\n".format("Dimension", self._layers[0].shape[0]) +
                "\n".join(
                    ["Layer  :  {:<10s} - {}".format(
                        _layer.name, _layer.shape[1]
                    ) for _layer in self._layers[:-1]]
                ) + "\nCost   :  {:<10s}".format(self._layers[-1].name)
            )
        print("=" * 30 + "\n" + "Structure\n" + "-" * 30 + "\n" + rs + "\n" + "=" * 30)
        print("Optimizer")
        print("-" * 30)
        print(self._w_optimizer)
        print("=" * 30)

    @NNTiming.timeit(level=2)
    def _append_log(self, x, y, y_classes, name):
        y_pred = self._get_prediction(x, name)
        y_pred_classes = np.argmax(y_pred, axis=1)
        for i, metric in enumerate(self._metrics):
            self._logs[name][i].append(metric(y_classes, y_pred_classes))
        self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred) / len(y))

    @NNTiming.timeit(level=3)
    def _print_metric_logs(self, data_type):
        print()
        print("=" * 47)
        for i, name in enumerate(self._metric_names):
            print("{:<16s} {:<16s}: {:12.8}".format(
                data_type, name, self._logs[data_type][i][-1]))
        print("{:<16s} {:<16s}: {:12.8}".format(
            data_type, "loss", self._logs[data_type][-1][-1]))
        print("=" * 47)

    @NNTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, lr=None, epoch=None, batch_size=None, train_rate=None,
            optimizer=None, metrics=None, record_period=None, verbose=None):
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        if optimizer is None:
            optimizer = self._params["optimizer"]
        if batch_size is None:
            batch_size = self._params["batch_size"]
        if train_rate is None:
            train_rate = self._params["train_rate"]
        if metrics is None:
            metrics = self._params["metrics"]
        if record_period is None:
            record_period = self._params["record_period"]
        if verbose is None:
            verbose = self._params["verbose"]
        self.verbose = verbose
        self._init_optimizers(optimizer, lr, epoch)
        layer_width = len(self._layers)
        self._preview()

        if train_rate is not None:
            train_rate = float(train_rate)
            train_len = int(len(x) * train_rate)
            shuffle_suffix = np.random.permutation(len(x))
            x, y = x[shuffle_suffix], y[shuffle_suffix]
            x_train, y_train = x[:train_len], y[:train_len]
            x_test, y_test = x[train_len:], y[train_len:]
        else:
            x_train = x_test = x
            y_train = y_test = y
        y_train_classes = np.argmax(y_train, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        train_len = len(x_train)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len > batch_size
        train_repeat = 1 if not do_random_batch else int(train_len / batch_size) + 1

        if metrics is None:
            metrics = []
        self._metrics = self.get_metrics(metrics)
        self._metric_names = [_m.__name__ for _m in metrics]
        self._logs = {
            name: [[] for _ in range(len(metrics) + 1)] for name in ("train", "test")
        }

        bar = ProgressBar(max_value=max(1, epoch // record_period), name="Epoch", start=False)
        if self.verbose >= NNVerbose.EPOCH:
            bar.start()

        sub_bar = ProgressBar(max_value=train_repeat * record_period - 1, name="Iteration", start=False)
        for counter in range(epoch):
            if self.verbose >= NNVerbose.EPOCH and counter % record_period == 0:
                sub_bar.start()
            for _ in range(train_repeat):
                if do_random_batch:
                    batch = np.random.choice(train_len, batch_size)
                    x_batch, y_batch = x_train[batch], y_train[batch]
                else:
                    x_batch, y_batch = x_train, y_train
                self._w_optimizer.update()
                self._b_optimizer.update()
                activations = self._get_activations(x_batch)
                deltas = [self._layers[-1].bp_first(y_batch, activations[-1])]
                for i in range(-1, -len(activations), -1):
                    deltas.append(
                        self._layers[i - 1].bp(activations[i - 1], self._weights[i], deltas[-1])
                    )
                for i in range(layer_width - 1, 0, -1):
                    self._opt(i, activations[i - 1], deltas[layer_width - i - 1])
                self._opt(0, x_batch, deltas[-1])
                if self.verbose >= NNVerbose.EPOCH:
                    if sub_bar.update() and self.verbose >= NNVerbose.METRICS_DETAIL:
                        self._append_log(x_train, y_train, y_train_classes, "train")
                        self._append_log(x_test, y_test, y_test_classes, "test")
                        self._print_metric_logs("train")
                        self._print_metric_logs("test")
            if self.verbose >= NNVerbose.EPOCH:
                sub_bar.update()
            if (counter + 1) % record_period == 0:
                self._append_log(x_train, y_train, y_train_classes, "train")
                self._append_log(x_test, y_test, y_test_classes, "test")
                if self.verbose >= NNVerbose.METRICS:
                    self._print_metric_logs("train")
                    self._print_metric_logs("test")
                if self.verbose >= NNVerbose.EPOCH:
                    bar.update(counter // record_period + 1)
                    sub_bar = ProgressBar(max_value=train_repeat * record_period - 1, name="Iteration", start=False)

    def draw_logs(self):
        metrics_log, loss_log = {}, {}
        for key, value in sorted(self._logs.items()):
            metrics_log[key], loss_log[key] = value[:-1], value[-1]
        for i, name in enumerate(sorted(self._metric_names)):
            plt.figure()
            plt.title("Metric Type: {}".format(name))
            for key, log in sorted(metrics_log.items()):
                xs = np.arange(len(log[i])) + 1
                plt.plot(xs, log[i], label="Data Type: {}".format(key))
            plt.legend(loc=4)
            plt.show()
            plt.close()
        plt.figure()
        plt.title("Cost")
        for key, loss in sorted(loss_log.items()):
            xs = np.arange(len(loss)) + 1
            plt.plot(xs, loss, label="Data Type: {}".format(key))
        plt.legend()
        plt.show()
