from Models.Zhihu.NN.Layers import *
from Models.Zhihu.NN.Optimizers import *
from Models.Zhihu.NN.Util import Timing, Util, ProgressBar

np.random.seed(142857)  # for reproducibility


class NNVerbose:
    NONE = 0
    EPOCH = 1
    METRICS = 2
    METRICS_DETAIL = 3
    DETAIL = 4
    DEBUG = 5


class NNConfig:
    BOOST_LESS_SAMPLES = False
    TRAINING_SCALE = 5 / 6


# Neural Network

class NNBase:
    NNTiming = Timing()

    def __init__(self):
        self._layers = []
        self._layer_names, self._layer_params = [], []
        self._lr = 0
        self._w_stds, self._b_inits = [], []
        self._optimizer = None
        self._data_size = 0
        self.verbose = 0

        self._current_dimension = 0

        self._logs = {}
        self._timings = {}
        self._metrics, self._metric_names = [], []

        self._x = self._y = None
        self._x_min = self._x_max = self._y_min = self._y_max = 0
        self._transferred_flags = {"train": False, "test": False}

        self._tfx = self._tfy = None
        self._tf_weights, self._tf_bias = [], []
        self._cost = self._y_pred = self._activations = None

        self._loaded = False
        self._train_step = None

        self._layer_factory = LayerFactory()

    def __getitem__(self, item):
        if isinstance(item, int):
            if item < 0 or item >= len(self._layers):
                return
            bias = self._tf_bias[item]
            return {
                "name": self._layers[item].name,
                "weight": self._tf_weights[item],
                "bias": bias
            }
        if isinstance(item, str):
            return getattr(self, "_" + item)
        return

    def __str__(self):
        return "Neural Network"

    __repr__ = __str__

    @NNTiming.timeit(level=4, prefix="[API] ")
    def feed_timing(self, timing):
        if isinstance(timing, Timing):
            self.NNTiming = timing
            for layer in self._layers:
                layer.feed_timing(timing)

    @NNTiming.timeit(level=4, prefix="[Private StaticMethod] ")
    def _get_w(self, shape):
        initial = tf.truncated_normal(shape, stddev=self._w_stds[-1])
        return tf.Variable(initial, name="w")

    @NNTiming.timeit(level=4, prefix="[Private StaticMethod] ")
    def _get_b(self, shape):
        initial = tf.constant(self._b_inits[-1], shape=shape)
        return tf.Variable(initial, name="b")

    @NNTiming.timeit(level=4)
    def _add_weight(self, shape):
        w_shape = shape
        b_shape = shape[1],
        self._tf_weights.append(self._get_w(w_shape))
        self._tf_bias.append(self._get_b(b_shape))

    @NNTiming.timeit(level=4)
    def _add_layer(self, layer, *args, **kwargs):
        if not self._layers and isinstance(layer, str):
            _layer = self._layer_factory.handle_str_main_layers(layer, *args, **kwargs)
            if _layer:
                self.add(_layer, pop_last_init=True)
                return
        _parent = self._layers[-1]
        if isinstance(_parent, CostLayer):
            raise BuildLayerError("Adding layer after CostLayer is not permitted")
        if isinstance(layer, str):
            layer, shape = self._layer_factory.get_layer_by_name(
                layer, parent=_parent, *args, **kwargs
            )
            if shape is None:
                self.add(layer, pop_last_init=True)
                return
            _current, _next = shape
        else:
            _current, _next = args
        self._layers.append(layer)
        self._add_weight((_current, _next))
        self._current_dimension = _next

    @NNTiming.timeit(level=1, prefix="[API] ")
    def get_rs(self, x, y=None, predict=False, pipe=False):
        if y is None:
            predict = True
        _cache = self._layers[0].activate(x, self._tf_weights[0], self._tf_bias[0], predict)
        for i, layer in enumerate(self._layers[1:]):
            if i == len(self._layers) - 2:
                if y is None:
                    if not pipe:
                        if self._tf_bias[-1] is not None:
                            return tf.matmul(_cache, self._tf_weights[-1]) + self._tf_bias[-1]
                        return tf.matmul(_cache, self._tf_weights[-1])
                    else:
                        return layer.activate(_cache, self._tf_weights[i + 1], self._tf_bias[i + 1], predict)
                predict = y
            _cache = layer.activate(_cache, self._tf_weights[i + 1], self._tf_bias[i + 1], predict)
        return _cache

    @NNTiming.timeit(level=4, prefix="[API] ")
    def add(self, layer, *args, **kwargs):
        self._w_stds.append(Util.get_and_pop(kwargs, "std", 0.1))
        self._b_inits.append(Util.get_and_pop(kwargs, "init", 0.1))
        if Util.get_and_pop(kwargs, "pop_last_init", False):
            self._w_stds.pop()
            self._b_inits.pop()
        if isinstance(layer, str):
            # noinspection PyTypeChecker
            self._add_layer(layer, *args, **kwargs)
        else:
            if not isinstance(layer, Layer):
                raise BuildLayerError("Invalid Layer provided (should be subclass of Layer)")
            if not self._layers:
                if len(layer.shape) != 2:
                    raise BuildLayerError("Invalid input Layer provided (shape should be {}, {} found)".format(
                        2, len(layer.shape)
                    ))
                self._layers, self._current_dimension = [layer], layer.shape[1]
                self._add_weight(layer.shape)
            else:
                if len(layer.shape) > 2:
                    raise BuildLayerError("Invalid Layer provided (shape should be {}, {} found)".format(
                        2, len(layer.shape)
                    ))
                if len(layer.shape) == 2:
                    _current, _next = layer.shape
                    self._add_layer(layer, _current, _next)
                elif len(layer.shape) == 1:
                    _next = layer.shape[0]
                    layer.shape = (self._current_dimension, _next)
                    self._add_layer(layer, self._current_dimension, _next)
                else:
                    raise LayerError("Invalid Layer provided (invalid shape '{}' found)".format(layer.shape))


class NNDist(NNBase):
    NNTiming = Timing()

    def __init__(self):
        NNBase.__init__(self)

        self._sess = tf.Session()
        self._optimizer_factory = OptFactory()

        self._available_metrics = {
            "acc": NNDist._acc, "_acc": NNDist._acc,
            "f1": NNDist._f1_score, "_f1_score": NNDist._f1_score
        }

    @NNTiming.timeit(level=4, prefix="[Initialize] ")
    def initialize(self):
        self._layers = []
        self._layer_names, self._layer_params = [], []
        self._lr = 0
        self._w_stds, self._b_inits = [], []
        self._optimizer = None
        self._data_size = 0
        self.verbose = 0

        self._current_dimension = 0

        self._logs = {}
        self._timings = {}
        self._metrics, self._metric_names = [], []

        self._x = self._y = None
        self._x_min = self._x_max = self._y_min = self._y_max = 0
        self._transferred_flags = {"train": False, "test": False}

        self._tfx = self._tfy = None
        self._tf_weights, self._tf_bias = [], []
        self._cost = self._y_pred = self._activations = None

        self._loaded = False
        self._train_step = None

        self._sess = tf.Session()

    # Property

    @property
    def layer_names(self):
        return [layer.name for layer in self._layers]

    @layer_names.setter
    def layer_names(self, value):
        self._layer_names = value

    @property
    def layer_special_params(self):
        return [layer.get_special_params(self._sess) for layer in self._layers]

    @layer_special_params.setter
    def layer_special_params(self, value):
        for layer, sp_param in zip(self._layers, value):
            if sp_param is not None:
                layer.set_special_params(sp_param)

    @property
    def optimizer(self):
        return self._optimizer.name

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    # Utils

    @NNTiming.timeit(level=4)
    def _feed_data(self, x, y):
        if x is None:
            if self._x is None:
                raise BuildNetworkError("Please provide input matrix")
            x = self._x
        else:
            if not self._transferred_flags["train"]:
                self._transferred_flags["train"] = True
        if y is None:
            if self._y is None:
                raise BuildNetworkError("Please provide input matrix")
            y = self._y
        if len(x) != len(y):
            raise BuildNetworkError("Data fed to network should be identical in length, x: {} and y: {} found".format(
                len(x), len(y)
            ))
        self._x, self._y = x, y
        self._x_min, self._x_max = np.min(x), np.max(x)
        self._y_min, self._y_max = np.min(y), np.max(y)
        self._data_size = len(x)
        return x, y

    @NNTiming.timeit(level=2)
    def _get_prediction(self, x, name=None, batch_size=1e6, verbose=None, out_of_sess=False):
        if verbose is None:
            verbose = self.verbose
        single_batch = int(batch_size / np.prod(x.shape[1:]))
        if not single_batch:
            single_batch = 1
        if single_batch >= len(x):
            if not out_of_sess:
                return self._y_pred.eval(feed_dict={self._tfx: x})
            with self._sess.as_default():
                return self.get_rs(x).eval(feed_dict={self._tfx: x})
        epoch = int(len(x) / single_batch)
        if not len(x) % single_batch:
            epoch += 1
        name = "Prediction" if name is None else "Prediction ({})".format(name)
        sub_bar = ProgressBar(min_value=0, max_value=epoch, name=name)
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

    @NNTiming.timeit(level=4)
    def _get_activations(self, x):
        _activations = [self._layers[0].activate(x, self._tf_weights[0], self._tf_bias[0], True)]
        for i, layer in enumerate(self._layers[1:]):
            if i == len(self._layers) - 2:
                if self._tf_bias[-1] is not None:
                    _activations.append(tf.matmul(_activations[-1], self._tf_weights[-1]) + self._tf_bias[-1])
                else:
                    _activations.append(tf.matmul(_activations[-1], self._tf_weights[-1]))
            else:
                _activations.append(layer.activate(
                    _activations[-1], self._tf_weights[i + 1], self._tf_bias[i + 1], True))
        return _activations

    @NNTiming.timeit(level=1)
    def _get_l2_loss(self, lb):
        if lb <= 0:
            return 0
        return lb * tf.reduce_sum([tf.nn.l2_loss(_w) for i, _w in enumerate(self._tf_weights)])

    @NNTiming.timeit(level=1)
    def _get_acts(self, x):
        with self._sess.as_default():
            _activations = [_ac.eval() for _ac in self._get_activations(x)]
        return _activations

    @NNTiming.timeit(level=3)
    def _append_log(self, x, y, name, get_loss=True, out_of_sess=False):
        y_pred = self._get_prediction(x, name, out_of_sess=out_of_sess)
        for i, metric in enumerate(self._metrics):
            self._logs[name][i].append(metric(y, y_pred))
        if get_loss:
            if not out_of_sess:
                self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred).eval())
            else:
                with self._sess.as_default():
                    self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred).eval())

    @NNTiming.timeit(level=3)
    def _print_metric_logs(self, show_loss, data_type):
        print()
        print("=" * 47)
        for i, name in enumerate(self._metric_names):
            print("{:<16s} {:<16s}: {:12.8}".format(
                data_type, name, self._logs[data_type][i][-1]))
        if show_loss:
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

    # Init

    @NNTiming.timeit(level=4)
    def _init_optimizer(self, optimizer=None):
        if optimizer is None:
            if isinstance(self._optimizer, str):
                optimizer = self._optimizer
            else:
                if self._optimizer is None:
                    self._optimizer = Adam(self._lr)
                if isinstance(self._optimizer, Optimizers):
                    return
                raise BuildNetworkError("Invalid optimizer '{}' provided".format(self._optimizer))
        if isinstance(optimizer, str):
            self._optimizer = self._optimizer_factory.get_optimizer_by_name(
                optimizer, self.NNTiming, self._lr)
        elif isinstance(optimizer, Optimizers):
            self._optimizer = optimizer
        else:
            raise BuildNetworkError("Invalid optimizer '{}' provided".format(optimizer))

    @NNTiming.timeit(level=4)
    def _init_train_step(self, sess):
        if not self._loaded:
            self._train_step = self._optimizer.minimize(self._cost)
            sess.run(tf.global_variables_initializer())
        else:
            _var_cache = set(tf.global_variables())
            self._train_step = self._optimizer.minimize(self._cost)
            sess.run(tf.variables_initializer(set(tf.global_variables()) - _var_cache))

    # API

    @NNTiming.timeit(level=4, prefix="[API] ")
    def feed(self, x, y):
        self._feed_data(x, y)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def build(self, units):
        try:
            units = np.array(units).flatten().astype(np.int)
        except ValueError as err:
            raise BuildLayerError(err)
        if len(units) < 2:
            raise BuildLayerError("At least 2 layers are needed")
        _input_shape = (units[0], units[1])
        self.initialize()
        self.add(Sigmoid(_input_shape))
        for unit_num in units[2:]:
            self.add(Sigmoid((unit_num,)))
        self.add(CrossEntropy((units[-1],)))

    @NNTiming.timeit(level=4, prefix="[API] ")
    def split_data(self, x, y, x_test, y_test,
                   train_only, training_scale=NNConfig.TRAINING_SCALE):
        if train_only:
            if x_test is not None and y_test is not None:
                if not self._transferred_flags["test"]:
                    x, y = np.vstack((x, x_test)), np.vstack((y, y_test))
                    self._transferred_flags["test"] = True
            x_train, y_train = np.array(x), np.array(y)
            x_test, y_test = x_train, y_train
        else:
            shuffle_suffix = np.random.permutation(len(x))
            x, y = x[shuffle_suffix], y[shuffle_suffix]
            if x_test is None or y_test is None:
                train_len = int(len(x) * training_scale)
                x_train, y_train = np.array(x[:train_len]), np.array(y[:train_len])
                x_test, y_test = np.array(x[train_len:]), np.array(y[train_len:])
            elif x_test is None or y_test is None:
                raise BuildNetworkError("Please provide test sets if you want to split data on your own")
            else:
                x_train, y_train = np.array(x), np.array(y)
                if not self._transferred_flags["test"]:
                    x_test, y_test = np.array(x_test), np.array(y_test)
                    self._transferred_flags["test"] = True
        if NNConfig.BOOST_LESS_SAMPLES:
            if y_train.shape[1] != 2:
                raise BuildNetworkError("It is not permitted to boost less samples in multiple classification")
            y_train_arg = np.argmax(y_train, axis=1)
            y0 = y_train_arg == 0
            y1 = ~y0
            y_len, y0_len = len(y_train), int(np.sum(y0))
            if y0_len > 0.5 * y_len:
                y0, y1 = y1, y0
                y0_len = y_len - y0_len
            boost_suffix = np.random.randint(y0_len, size=y_len - y0_len)
            x_train = np.vstack((x_train[y1], x_train[y0][boost_suffix]))
            y_train = np.vstack((y_train[y1], y_train[y0][boost_suffix]))
            shuffle_suffix = np.random.permutation(len(x_train))
            x_train, y_train = x_train[shuffle_suffix], y_train[shuffle_suffix]
        return (x_train, x_test), (y_train, y_test)

    @NNTiming.timeit(level=1, prefix="[API] ")
    def fit(self,
            x=None, y=None, x_test=None, y_test=None,
            lr=0.001, lb=0.001, epoch=10, weight_scale=1, apply_bias=True,
            batch_size=512, record_period=1, train_only=False, optimizer=None,
            show_loss=True, metrics=None, do_log=True, verbose=None):

        x, y = self._feed_data(x, y)
        self._lr = lr
        self._init_optimizer(optimizer)
        print("Optimizer: ", self._optimizer.name)
        print("-" * 30)

        if not self._layers:
            raise BuildNetworkError("Please provide layers before fitting data")

        if y.shape[1] != self._current_dimension:
            raise BuildNetworkError("Output layer's shape should be {}, {} found".format(
                self._current_dimension, y.shape[1]))

        (x_train, x_test), (y_train, y_test) = self.split_data(x, y, x_test, y_test, train_only)
        train_len = len(x_train)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len >= batch_size
        train_repeat = int(train_len / batch_size) + 1
        self._feed_data(x_train, y_train)

        self._tfx = tf.placeholder(tf.float32, shape=[None, *x.shape[1:]])
        self._tfy = tf.placeholder(tf.float32, shape=[None, y.shape[1]])

        self._metrics = ["acc"] if metrics is None else metrics
        for i, metric in enumerate(self._metrics):
            if isinstance(metric, str):
                if metric not in self._available_metrics:
                    raise BuildNetworkError("Metric '{}' is not implemented".format(metric))
                self._metrics[i] = self._available_metrics[metric]
        self._metric_names = [_m.__name__ for _m in self._metrics]

        self._logs = {
            name: [[] for _ in range(len(self._metrics) + 1)] for name in ("train", "test")
            }
        if verbose is not None:
            self.verbose = verbose

        bar = ProgressBar(min_value=0, max_value=max(1, epoch // record_period), name="Epoch")
        if self.verbose >= NNVerbose.EPOCH:
            bar.start()

        with self._sess.as_default() as sess:

            # Session
            self._cost = self.get_rs(self._tfx, self._tfy) + self._get_l2_loss(lb)
            self._y_pred = self.get_rs(self._tfx)
            self._activations = self._get_activations(self._tfx)
            self._init_train_step(sess)
            for weight in self._tf_weights:
                weight *= weight_scale
            if not apply_bias:
                self._tf_bias = [None] * len(self._tf_bias)

            sub_bar = ProgressBar(min_value=0, max_value=train_repeat * record_period - 1, name="Iteration")
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

                    if self.verbose >= NNVerbose.DEBUG:
                        pass
                    if self.verbose >= NNVerbose.EPOCH:
                        if sub_bar.update() and self.verbose >= NNVerbose.METRICS_DETAIL:
                            self._append_log(x, y, "train", get_loss=show_loss)
                            self._append_log(x_test, y_test, "test", get_loss=show_loss)
                            self._print_metric_logs(show_loss, "train")
                            self._print_metric_logs(show_loss, "test")
                if self.verbose >= NNVerbose.EPOCH:
                    sub_bar.update()

                if (counter + 1) % record_period == 0:
                    if do_log:
                        self._append_log(x, y, "train", get_loss=show_loss)
                        self._append_log(x_test, y_test, "test", get_loss=show_loss)
                        if self.verbose >= NNVerbose.METRICS:
                            self._print_metric_logs(show_loss, "train")
                            self._print_metric_logs(show_loss, "test")
                    if self.verbose >= NNVerbose.EPOCH:
                        bar.update(counter // record_period + 1)
                        sub_bar = ProgressBar(min_value=0, max_value=train_repeat * record_period - 1, name="Iteration")

        return self._logs

    @NNTiming.timeit(level=4, prefix="[API] ")
    def predict(self, x):
        x = np.array(x)
        return self._get_prediction(x, out_of_sess=True)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def predict_classes(self, x, flatten=True):
        x = np.array(x)
        if flatten:
            return np.argmax(self._get_prediction(x, out_of_sess=True), axis=1)
        return np.argmax([self._get_prediction(x, out_of_sess=True)], axis=2).T

    @NNTiming.timeit(level=4, prefix="[API] ")
    def evaluate(self, x, y, metrics=None):
        x = np.array(x)
        if metrics is None:
            metrics = self._metrics
        else:
            for i in range(len(metrics) - 1, -1, -1):
                metric = metrics[i]
                if isinstance(metric, str):
                    if metric not in self._available_metrics:
                        metrics.pop(i)
                    else:
                        metrics[i] = self._available_metrics[metric]
        logs, y_pred = [], self._get_prediction(x, verbose=2, out_of_sess=True)
        for metric in metrics:
            logs.append(metric(y, y_pred))
        return logs
