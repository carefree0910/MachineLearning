import matplotlib.pyplot as plt

from Zhihu.NN.Layers import *
from Zhihu.NN.Optimizers import *

from Util.ProgressBar import ProgressBar


class NNVerbose:
    NONE = 0
    EPOCH = 1
    METRICS = 2
    METRICS_DETAIL = 3
    DETAIL = 4
    DEBUG = 5


# Neural Network

class NNBase:

    def __init__(self):
        self._layers = []
        self._optimizer = None
        self._current_dimension = 0

        self._tfx = self._tfy = None
        self._tf_weights, self._tf_bias = [], []
        self._cost = self._y_pred = None

        self._train_step = None
        self.verbose = 0

    def __str__(self):
        return "Neural Network"

    __repr__ = __str__

    @staticmethod
    def _get_w(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="w")

    @staticmethod
    def _get_b(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="b")

    def _add_weight(self, shape):
        w_shape = shape
        b_shape = shape[1],
        self._tf_weights.append(self._get_w(w_shape))
        self._tf_bias.append(self._get_b(b_shape))

    def get_rs(self, x, y=None):
        _cache = self._layers[0].activate(x, self._tf_weights[0], self._tf_bias[0])
        for i, layer in enumerate(self._layers[1:]):
            if i == len(self._layers) - 2:
                if y is None:
                    return tf.matmul(_cache, self._tf_weights[-1]) + self._tf_bias[-1]
                return layer.activate(_cache, self._tf_weights[i + 1], self._tf_bias[i + 1], y)
            _cache = layer.activate(_cache, self._tf_weights[i + 1], self._tf_bias[i + 1])
        return _cache

    def add(self, layer):
        if not self._layers:
            self._layers, self._current_dimension = [layer], layer.shape[1]
            self._add_weight(layer.shape)
        else:
            _next = layer.shape[0]
            self._layers.append(layer)
            self._add_weight((self._current_dimension, _next))
            self._current_dimension = _next


class NNDist(NNBase):

    def __init__(self):
        NNBase.__init__(self)
        self._logs = {}
        self._sess = tf.Session()
        self._metrics, self._metric_names = [], []
        self._available_metrics = {
            "acc": NNDist._acc, "_acc": NNDist._acc,
            "f1_score": NNDist._f1_score, "_f1_score": NNDist._f1_score
        }

    # Utils

    def _get_prediction(self, x, name=None, batch_size=1e6, verbose=None, out_of_sess=False):
        if verbose is None:
            verbose = self.verbose
        fc_shape = np.prod(x.shape[1:])  # type: float
        single_batch = int(batch_size / fc_shape)
        if not single_batch:
            single_batch = 1
        if single_batch >= len(x):
            if not out_of_sess:
                return self._y_pred.eval(feed_dict={self._tfx: x})
            with self._sess.as_default():
                x = x.astype(np.float32)
                return self.get_rs(x).eval(feed_dict={self._tfx: x})
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
            rs = [self.get_rs(x[:single_batch])]
        count = single_batch
        if verbose >= NNVerbose.METRICS:
            sub_bar.update()
        while count < len(x):
            count += single_batch
            if count >= len(x):
                if not out_of_sess:
                    rs.append(self._y_pred.eval(feed_dict={self._tfx: x[count - single_batch:]}))
                else:
                    rs.append(self.get_rs(x[count - single_batch:]))
            else:
                if not out_of_sess:
                    rs.append(self._y_pred.eval(feed_dict={self._tfx: x[count - single_batch:count]}))
                else:
                    rs.append(self.get_rs(x[count - single_batch:count]))
            if verbose >= NNVerbose.METRICS:
                sub_bar.update()
        if out_of_sess:
            with self._sess.as_default():
                rs = [_rs.eval() for _rs in rs]
        return np.vstack(rs)

    def _append_log(self, x, y, name, out_of_sess=False):
        y_pred = self._get_prediction(x, name, out_of_sess=out_of_sess)
        for i, metric in enumerate(self._metrics):
            self._logs[name][i].append(metric(y, y_pred))
        if not out_of_sess:
            self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred).eval())
        else:
            with self._sess.as_default():
                self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred).eval())

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
    def _acc(y, y_pred):
        y_arg, y_pred_arg = np.argmax(y, axis=1), np.argmax(y_pred, axis=1)
        return np.sum(y_arg == y_pred_arg) / len(y_arg)

    @staticmethod
    def _f1_score(y, y_pred):
        y_true, y_pred = np.argmax(y, axis=1), np.argmax(y_pred, axis=1)
        tp = np.sum(y_true * y_pred)
        if tp == 0:
            return .0
        fp = np.sum((1 - y_true) * y_pred)
        fn = np.sum(y_true * (1 - y_pred))
        return 2 * tp / (2 * tp + fn + fp)

    # API

    def fit(self, x=None, y=None, lr=0.01, epoch=10, batch_size=128, train_rate=None,
            verbose=0, metrics=None, record_period=100):

        self.verbose = verbose
        self._optimizer = Adam(lr)
        self._tfx = tf.placeholder(tf.float32, shape=[None, x.shape[1]])
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

        with self._sess.as_default() as sess:

            # Define session
            self._cost = self.get_rs(self._tfx, self._tfy)
            self._y_pred = self.get_rs(self._tfx)
            self._train_step = self._optimizer.minimize(self._cost)
            sess.run(tf.global_variables_initializer())

            # Train
            sub_bar = ProgressBar(max_value=train_repeat * record_period - 1, name="Iteration", start=False)
            for counter in range(epoch):
                if self.verbose >= NNVerbose.EPOCH and counter % record_period == 0:
                    sub_bar.start()
                for _i in range(train_repeat):
                    if do_random_batch:
                        batch = np.random.choice(train_len, batch_size)
                        x_batch, y_batch = x_train[batch], y_train[batch]
                    else:
                        x_batch, y_batch = x_train, y_train
                    self._train_step.run(feed_dict={self._tfx: x_batch, self._tfy: y_batch})
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

    def predict(self, x):
        return self._get_prediction(x, out_of_sess=True)

    def predict_classes(self, x):
        x = np.array(x)
        return np.argmax(self._get_prediction(x, out_of_sess=True), axis=1)

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
