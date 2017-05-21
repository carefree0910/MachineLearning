import matplotlib.pyplot as plt

from g_CNN.Layers import *
from g_CNN.Optimizers import *

from Util.Timing import Timing
from Util.Bases import TFClassifierBase
from Util.ProgressBar import ProgressBar

# TODO: Optimize the codes


class NNVerbose:
    NONE = 0
    EPOCH = 1
    METRICS = 2
    METRICS_DETAIL = 3
    DETAIL = 4
    DEBUG = 5


class NN(TFClassifierBase):
    NNTiming = Timing()

    def __init__(self, **kwargs):
        super(NN, self).__init__(**kwargs)
        self._layers = []
        self._optimizer = None
        self._current_dimension = 0
        self._available_metrics = {
            key: value for key, value in zip(["acc", "f1-score"], [NN.acc, NN.f1_score])
        }
        self._metrics, self._metric_names, self._logs = [], [], {}

        self.verbose = 0
        self._layer_factory = LayerFactory()
        self._cost = self._train_step = None
        self._tf_weights, self._tf_bias = [], []

        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["epoch"] = kwargs.get("epoch", 10)
        self._params["optimizer"] = kwargs.get("optimizer", "Adam")
        self._params["batch_size"] = kwargs.get("batch_size", 256)
        self._params["train_rate"] = kwargs.get("train_rate", None)
        self._params["metrics"] = kwargs.get("metrics", None)
        self._params["record_period"] = kwargs.get("record_period", 100)
        self._params["verbose"] = kwargs.get("verbose", 1)
        self._params["preview"] = kwargs.get("preview", True)

    @NNTiming.timeit(level=1)
    def _get_prediction(self, x, name=None, batch_size=1e6, verbose=None, out_of_sess=False):
        if verbose is None:
            verbose = self.verbose
        single_batch = batch_size / np.prod(x.shape[1:])  # type: float
        single_batch = int(single_batch)
        if not single_batch:
            single_batch = 1
        if single_batch >= len(x):
            if not out_of_sess:
                return self._y_pred.eval(feed_dict={self._tfx: x})
            with self._sess.as_default():
                x = x.astype(np.float32)
                return self._get_rs(x).eval(feed_dict={self._tfx: x})
        epoch = int(len(x) / single_batch)
        if not len(x) % single_batch:
            epoch += 1
        name = "Prediction" if name is None else "Prediction ({})".format(name)
        sub_bar = ProgressBar(max_value=epoch, name=name, start=False)
        if verbose >= NNVerbose.METRICS:
            sub_bar.start()
        if not out_of_sess:
            rs = [self._y_pred.eval(feed_dict={self._tfx: x[:single_batch]})]
        else:
            rs = [self._get_rs(x[:single_batch])]
        count = single_batch
        if verbose >= NNVerbose.METRICS:
            sub_bar.update()
        while count < len(x):
            count += single_batch
            if count >= len(x):
                if not out_of_sess:
                    rs.append(self._y_pred.eval(feed_dict={self._tfx: x[count - single_batch:]}))
                else:
                    rs.append(self._get_rs(x[count - single_batch:]))
            else:
                if not out_of_sess:
                    rs.append(self._y_pred.eval(feed_dict={self._tfx: x[count - single_batch:count]}))
                else:
                    rs.append(self._get_rs(x[count - single_batch:count]))
            if verbose >= NNVerbose.METRICS:
                sub_bar.update()
        if out_of_sess:
            with self._sess.as_default():
                rs = [_rs.eval() for _rs in rs]
        return np.vstack(rs)

    @staticmethod
    @NNTiming.timeit(level=4)
    def _get_w(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="w")

    @staticmethod
    @NNTiming.timeit(level=4)
    def _get_b(shape):
        return tf.Variable(np.zeros(shape, dtype=np.float32) + 0.1, name="b")

    @NNTiming.timeit(level=4)
    def _add_params(self, shape, conv_channel=None, fc_shape=None, apply_bias=True):
        if fc_shape is not None:
            w_shape = (fc_shape, shape[1])
            b_shape = shape[1],
        elif conv_channel is not None:
            if len(shape[1]) <= 2:
                w_shape = shape[1][0], shape[1][1], conv_channel, conv_channel
            else:
                w_shape = (shape[1][1], shape[1][2], conv_channel, shape[1][0])
            b_shape = shape[1][0],
        else:
            w_shape = shape
            b_shape = shape[1],
        self._tf_weights.append(self._get_w(w_shape))
        if apply_bias:
            self._tf_bias.append(self._get_b(b_shape))
        else:
            self._tf_bias.append(None)

    @NNTiming.timeit(level=4)
    def _add_param_placeholder(self):
        self._tf_weights.append(tf.constant([.0]))
        self._tf_bias.append(tf.constant([.0]))

    @NNTiming.timeit(level=4)
    def _add_layer(self, layer, *args, **kwargs):
        if not self._layers and isinstance(layer, str):
            _layer = self._layer_factory.get_root_layer_by_name(layer, *args, **kwargs)
            if _layer:
                self.add(_layer)
                return
        _parent = self._layers[-1]
        if isinstance(layer, str):
            layer, shape = self._layer_factory.get_layer_by_name(
                layer, _parent, self._current_dimension, *args, **kwargs
            )
            if shape is None:
                self.add(layer)
                return
            _current, _next = shape
        else:
            _current, _next = args
        if isinstance(layer, SubLayer):
            self.parent = _parent
            self._layers.append(layer)
            self._add_param_placeholder()
            self._current_dimension = _next
        else:
            fc_shape, conv_channel, last_layer = None, None, self._layers[-1]
            if isinstance(last_layer, ConvLayer):
                if isinstance(layer, ConvLayer):
                    conv_channel = last_layer.n_filters
                    _current = (conv_channel, last_layer.out_h, last_layer.out_w)
                    layer.feed_shape((_current, _next))
                else:
                    layer.is_fc = True
                    last_layer.is_fc_base = True
                    fc_shape = last_layer.out_h * last_layer.out_w * last_layer.n_filters
            self._layers.append(layer)
            self._add_params((_current, _next), conv_channel, fc_shape, layer.apply_bias)
            self._current_dimension = _next

    @NNTiming.timeit(level=1)
    def _get_rs(self, x, predict=True, idx=-1):
        _cache = self._layers[0].activate(x, self._tf_weights[0], self._tf_bias[0], predict)
        idx = idx + 1 if idx >= 0 else len(self._layers) + idx + 1
        for i, layer in enumerate(self._layers[1:idx]):
            if i == len(self._layers) - 2:
                if isinstance(self._layers[-2], ConvLayer):
                    _shape = np.prod(_cache.get_shape()[1:])  # type: int
                    _cache = tf.reshape(_cache, [-1, _shape])
                if self._tf_bias[-1] is not None:
                    return tf.matmul(_cache, self._tf_weights[-1]) + self._tf_bias[-1]
                return tf.matmul(_cache, self._tf_weights[-1])
            _cache = layer.activate(_cache, self._tf_weights[i + 1], self._tf_bias[i + 1], predict)
        return _cache

    @NNTiming.timeit(level=2)
    def _append_log(self, x, y, y_classes, name, out_of_sess=False):
        y_pred = self._get_prediction(x, name, out_of_sess=out_of_sess)
        y_pred_class = np.argmax(y_pred, axis=1)
        for i, metric in enumerate(self._metrics):
            self._logs[name][i].append(metric(y_classes, y_pred_class))
        if not out_of_sess:
            self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred).eval())
        else:
            with self._sess.as_default():
                self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred).eval())

    @NNTiming.timeit(level=2)
    def _print_metric_logs(self, data_type):
        print()
        print("=" * 47)
        for i, name in enumerate(self._metric_names):
            print("{:<16s} {:<16s}: {:12.8}".format(
                data_type, name, self._logs[data_type][i][-1]))
        print("{:<16s} {:<16s}: {:12.8}".format(
            data_type, "loss", self._logs[data_type][-1][-1]))
        print("=" * 47)

    @staticmethod
    @NNTiming.timeit(level=4, prefix="[Private StaticMethod] ")
    def _transfer_x(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(x.shape) == 4:
            x = x.transpose(0, 2, 3, 1)
        return x.astype(np.float32)

    @NNTiming.timeit(level=4)
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
        print("\n" + "=" * 30 + "\n" + "Structure\n" + "-" * 30 + "\n" + rs + "\n" + "=" * 30)
        print("Optimizer")
        print("-" * 30)
        print(self._optimizer)
        print("=" * 30)

    @NNTiming.timeit(level=2)
    def _batch_work(self, i, x_train, y_train, y_train_classes, x_test, y_test, y_test_classes, condition):
        if condition:
            self._append_log(x_train, y_train, y_train_classes, "train")
            self._append_log(x_test, y_test, y_test_classes, "test")
            self._print_metric_logs("train")
            self._print_metric_logs("test")

    @NNTiming.timeit(level=4, prefix="[API] ")
    def add(self, layer, *args, **kwargs):
        if isinstance(layer, str):
            self._add_layer(layer, *args, **kwargs)
        else:
            if not self._layers:
                self._layers, self._current_dimension = [layer], layer.shape[1]
                if isinstance(layer, ConvLayer):
                    self._add_params(layer.shape, layer.n_channels, apply_bias=layer.apply_bias)
                else:
                    self._add_params(layer.shape, apply_bias=layer.apply_bias)
            else:
                if len(layer.shape) == 2:
                    _current, _next = layer.shape
                else:
                    _current, _next = self._current_dimension, layer.shape[0]
                    layer.shape = (_current, _next)
                self._add_layer(layer, _current, _next)

    @NNTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, lr=None, epoch=None, batch_size=None, train_rate=None,
            optimizer=None, metrics=None, record_period=None, verbose=None, preview=None):
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
        if preview is None:
            preview = self._params["preview"]

        x = NN._transfer_x(x)
        self.verbose = verbose
        self._optimizer = OptFactory().get_optimizer_by_name(optimizer, lr)
        self._tfx = tf.placeholder(tf.float32, shape=[None, *x.shape[1:]])
        self._tfy = tf.placeholder(tf.float32, shape=[None, y.shape[1]])

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
        y_train_classes = np.argmax(y_train, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

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

        if preview:
            self._preview()

        args = (
            (x_train, y_train, y_train_classes,
             x_test, y_test, y_test_classes,
             self.verbose >= NNVerbose.EPOCH and self.verbose >= NNVerbose.METRICS_DETAIL),
            (None, x_train, y_train, y_train_classes, x_test, y_test, y_test_classes,
             self.verbose >= NNVerbose.METRICS)
        )

        with self._sess.as_default() as sess:
            self._y_pred = self._get_rs(self._tfx, predict=False)
            self._cost = self._layers[-1].calculate(self._tfy, self._y_pred)
            self._train_step = self._optimizer.minimize(self._cost)
            sess.run(tf.global_variables_initializer())
            for counter in range(epoch):
                self.batch_training(x_train, y_train, batch_size, self._cost, self._train_step, *args[0])
                if (counter + 1) % record_period == 0:
                    self._batch_work(*args[1])
                    if self.verbose >= NNVerbose.EPOCH:
                        bar.update(counter // record_period + 1)

    @NNTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False):
        y_pred = self._get_prediction(NN._transfer_x(x).astype(np.float32), out_of_sess=True)
        if get_raw_results:
            return y_pred
        return np.argmax(y_pred, axis=1)

    def draw_logs(self):
        metrics_log, cost_log = {}, {}
        for key, value in sorted(self._logs.items()):
            metrics_log[key], cost_log[key] = value[:-1], value[-1]
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
        for key, loss in sorted(cost_log.items()):
            xs = np.arange(len(loss)) + 1
            plt.plot(xs, loss, label="Data Type: {}".format(key))
        plt.legend()
        plt.show()
