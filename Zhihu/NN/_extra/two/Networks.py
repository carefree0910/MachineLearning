import matplotlib.pyplot as plt

from Zhihu.NN._extra.Layers import *
from Zhihu.NN._extra.Optimizers import *

from Util.Timing import Timing
from Util.ProgressBar import ProgressBar


class NNVerbose:
    NONE = 0
    EPOCH = 1
    METRICS = 2
    METRICS_DETAIL = 3
    DETAIL = 4
    DEBUG = 5


# Neural Network

class NNDist:
    NNTiming = Timing()

    def __init__(self):
        self._layers, self._weights, self._bias = [], [], []
        self._w_optimizer = self._b_optimizer = None
        self._current_dimension = 0

        self.verbose = 0
        self._logs = {}
        self._metrics, self._metric_names = [], []
        self._available_metrics = {
            "acc": NNDist._acc, "_acc": NNDist._acc,
            "f1_score": NNDist._f1_score, "_f1_score": NNDist._f1_score
        }

    @NNTiming.timeit(level=4, prefix="[API] ")
    def feed_timing(self, timing):
        if isinstance(timing, Timing):
            self.NNTiming = timing
            for layer in self._layers:
                layer.feed_timing(timing)

    def __str__(self):
        return "Neural Network"

    __repr__ = __str__

    # Utils

    @NNTiming.timeit(level=4)
    def _add_weight(self, shape):
        self._weights.append(np.random.randn(*shape))
        self._bias.append(np.zeros((1, shape[1])))

    @NNTiming.timeit(level=4)
    def _add_layer(self, layer, *args):
        _parent = self._layers[-1]
        _current, _next = args
        self._layers.append(layer)
        if isinstance(layer, CostLayer):
            _parent.child = layer
            self.parent = _parent
            self._weights.append(np.eye(_current))
            self._bias.append(np.zeros((1, _current)))
            self._current_dimension = _next
        else:
            self._add_weight((_current, _next))
            self._current_dimension = _next

    @NNTiming.timeit(level=4)
    def _add_cost_layer(self):
        _last_layer = self._layers[-1]
        if _last_layer.name == "Sigmoid":
            _cost_func = "Cross Entropy"
        elif _last_layer.name == "Softmax":
            _cost_func = "Log Likelihood"
        else:
            _cost_func = "MSE"
        _cost_layer = CostLayer(_last_layer, (self._current_dimension,), _cost_func)
        self.add(_cost_layer)

    @NNTiming.timeit(level=1)
    def _get_prediction(self, x, name=None, batch_size=1e6, verbose=None):
        if verbose is None:
            verbose = self.verbose
        fc_shape = np.prod(x.shape[1:])  # type: float
        single_batch = int(batch_size / fc_shape)
        if not single_batch:
            single_batch = 1
        if single_batch >= len(x):
            return self._get_activations(x, predict=True).pop()
        epoch = int(len(x) / single_batch)
        if not len(x) % single_batch:
            epoch += 1
        name = "Prediction" if name is None else "Prediction ({})".format(name)
        sub_bar = ProgressBar(max_value=epoch, name=name, start=False)
        if verbose >= NNVerbose.METRICS:
            sub_bar.start()
        rs, count = [self._get_activations(x[:single_batch], predict=True).pop()], single_batch
        if verbose >= NNVerbose.METRICS:
            sub_bar.update()
        while count < len(x):
            count += single_batch
            if count >= len(x):
                rs.append(self._get_activations(x[count - single_batch:], predict=True).pop())
            else:
                rs.append(self._get_activations(x[count - single_batch:count], predict=True).pop())
            if verbose >= NNVerbose.METRICS:
                sub_bar.update()
        return np.vstack(rs)

    @NNTiming.timeit(level=1)
    def _get_activations(self, x, predict=False):
        _activations = [self._layers[0].activate(x, self._weights[0], self._bias[0], predict)]
        for i, layer in enumerate(self._layers[1:]):
            _activations.append(layer.activate(
                _activations[-1], self._weights[i + 1], self._bias[i + 1], predict))
        return _activations

    @NNTiming.timeit(level=3)
    def _append_log(self, x, y, name):
        y_pred = self._get_prediction(x, name)
        for i, metric in enumerate(self._metrics):
            self._logs[name][i].append(metric(y, y_pred))
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

    # Metrics

    @staticmethod
    @NNTiming.timeit(level=2, prefix="[Private StaticMethod] ")
    def _acc(y, y_pred):
        y_arg, y_pred_arg = np.argmax(y, axis=1), np.argmax(y_pred, axis=1)
        return np.sum(y_arg == y_pred_arg) / len(y_arg)

    @staticmethod
    @NNTiming.timeit(level=2, prefix="[Private StaticMethod] ")
    def _f1_score(y, y_pred):
        y_true, y_pred = np.argmax(y, axis=1), np.argmax(y_pred, axis=1)
        tp = np.sum(y_true * y_pred)
        if tp == 0:
            return .0
        fp = np.sum((1 - y_true) * y_pred)
        fn = np.sum(y_true * (1 - y_pred))
        return 2 * tp / (2 * tp + fn + fp)

    # Optimizing Process

    def _init_optimizers(self, lr):
        self._w_optimizer, self._b_optimizer = Adam(lr), Adam(lr)
        self._w_optimizer.feed_variables(self._weights)
        self._b_optimizer.feed_variables(self._bias)

    @NNTiming.timeit(level=1)
    def _opt(self, i, _activation, _delta):
        self._weights[i] += self._w_optimizer.run(
            i, _activation.reshape(_activation.shape[0], -1).T.dot(_delta)
        )
        self._bias[i] += self._b_optimizer.run(
            i, np.sum(_delta, axis=0, keepdims=True)
        )

    # API

    @NNTiming.timeit(level=4, prefix="[API] ")
    def add(self, layer):
        if not self._layers:
            self._layers, self._current_dimension = [layer], layer.shape[1]
            self._add_weight(layer.shape)
        else:
            _next = layer.shape[0]
            layer.shape = (self._current_dimension, _next)
            self._add_layer(layer, self._current_dimension, _next)

    @NNTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x=None, y=None, lr=0.01, epoch=10, batch_size=128, train_rate=None,
            verbose=0, metrics=None, record_period=100):

        # Initialize
        self.verbose = verbose
        self._add_cost_layer()
        self._init_optimizers(lr)
        layer_width = len(self._layers)

        if train_rate is not None:
            train_rate = float(train_rate)
            train_len = int(len(x) * train_rate)
            shuffle_suffix = np.random.permutation(int(len(x)))
            x, y = x[shuffle_suffix], y[shuffle_suffix]
            x_train, y_train = x[:train_len], y[:train_len]
            x_test, y_test = x[train_len:], y[train_len:]
        else:
            x_train = x_test = x
            y_train = y_test = y

        train_len = len(x_train)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len >= batch_size
        train_repeat = int(train_len / batch_size) + 1

        self._metrics = ["acc"] if metrics is None else metrics
        for i, metric in enumerate(self._metrics):
            if isinstance(metric, str):
                self._metrics[i] = self._available_metrics[metric]
        self._metric_names = [_m.__name__ for _m in self._metrics]
        self._logs = {
            name: [[] for _ in range(len(self._metrics) + 1)] for name in ("train", "test")
        }

        bar = ProgressBar(max_value=max(1, epoch // record_period), name="Epoch", start=False)
        if self.verbose >= NNVerbose.EPOCH:
            bar.start()

        # Train
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
                self._w_optimizer.update(); self._b_optimizer.update()
                _activations = self._get_activations(x_batch)
                _deltas = [self._layers[-1].bp_first(y_batch, _activations[-1])]
                for i in range(-1, -len(_activations), -1):
                    _deltas.append(
                        self._layers[i - 1].bp(_activations[i - 1], self._weights[i], _deltas[-1])
                    )
                for i in range(layer_width - 2, 0, -1):
                    self._opt(i, _activations[i - 1], _deltas[layer_width - i - 1])
                self._opt(0, x_batch, _deltas[-1])
                if self.verbose >= NNVerbose.EPOCH:
                    if sub_bar.update() and self.verbose >= NNVerbose.METRICS_DETAIL:
                        self._append_log(x_train, y_train, "train")
                        self._append_log(x_test, y_test, "test")
                        self._print_metric_logs("train")
                        self._print_metric_logs("test")
            if self.verbose >= NNVerbose.EPOCH:
                sub_bar.update()
            if (counter + 1) % record_period == 0:
                self._append_log(x_train, y_train, "train")
                self._append_log(x_test, y_test, "test")
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
        plt.title("Loss")
        for key, loss in sorted(loss_log.items()):
            xs = np.arange(len(loss)) + 1
            plt.plot(xs, loss, label="Data Type: {}".format(key))
        plt.legend()
        plt.show()

    @NNTiming.timeit(level=4, prefix="[API] ")
    def predict(self, x):
        return self._get_prediction(x)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def predict_classes(self, x):
        x = np.array(x)
        return np.argmax(self._get_prediction(x), axis=1)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def evaluate(self, x, y):
        y_pred = self.predict_classes(x)
        y_arg = np.argmax(y, axis=1)
        print("Acc: {:8.6}".format(np.sum(y_arg == y_pred) / len(y_arg)))

    def visualize_2d(self, x, y, plot_scale=2, plot_precision=0.01):

        plot_num = int(1 / plot_precision)

        xf = np.linspace(np.min(x) * plot_scale, np.max(x) * plot_scale, plot_num)
        yf = np.linspace(np.min(x) * plot_scale, np.max(x) * plot_scale, plot_num)
        input_x, input_y = np.meshgrid(xf, yf)
        input_xs = np.c_[input_x.ravel(), input_y.ravel()]

        output_ys_2d = np.argmax(self.predict(input_xs), axis=1).reshape(len(xf), len(yf))

        plt.contourf(input_x, input_y, output_ys_2d, cmap=plt.cm.Spectral)
        plt.scatter(x[:, 0], x[:, 1], c=np.argmax(y, axis=1), s=40, cmap=plt.cm.Spectral)
        plt.axis("off")
        plt.show()
