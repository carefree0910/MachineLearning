from Zhihu.NN.Layers import *
from Zhihu.NN.Optimizers import *
from Zhihu.NN.Util import Timing, Util

np.random.seed(142857)  # for reproducibility


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

        self._tfx = self._tfy = None
        self._tf_weights, self._tf_bias = [], []
        self._cost = self._y_pred = self._activations = None

        self._loaded = False
        self._train_step = None

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
    def _add_layer(self, layer, *args):
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
    def add(self, layer, **kwargs):
        self._w_stds.append(Util.get_and_pop(kwargs, "std", 0.1))
        self._b_inits.append(Util.get_and_pop(kwargs, "init", 0.1))
        if not self._layers:
            self._layers, self._current_dimension = [layer], layer.shape[1]
            self._add_weight(layer.shape)
        else:
            _next = layer.shape[0]
            layer.shape = (self._current_dimension, _next)
            self._add_layer(layer, self._current_dimension, _next)


class NNDist(NNBase):
    NNTiming = Timing()

    def __init__(self):
        NNBase.__init__(self)

        self._sess = tf.Session()
        self._optimizer_factory = OptFactory()

    # Property

    @property
    def optimizer(self):
        return self._optimizer.name

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    # Utils

    @NNTiming.timeit(level=4)
    def _feed_data(self, x, y):
        self._x, self._y = x, y
        self._x_min, self._x_max = np.min(x), np.max(x)
        self._y_min, self._y_max = np.min(y), np.max(y)
        self._data_size = len(x)
        return x, y

    @NNTiming.timeit(level=2)
    def _get_prediction(self, x, out_of_sess=False):
        if not out_of_sess:
            return self._y_pred.eval(feed_dict={self._tfx: x})
        with self._sess.as_default():
            return self.get_rs(x).eval(feed_dict={self._tfx: x})

    @NNTiming.timeit(level=4)
    def _get_activations(self, x):
        _activations = [self._layers[0].activate(x, self._tf_weights[0], self._tf_bias[0], True)]
        for i, layer in enumerate(self._layers[1:]):
            if i == len(self._layers) - 2:
                _activations.append(tf.matmul(_activations[-1], self._tf_weights[-1]) + self._tf_bias[-1])
            else:
                _activations.append(layer.activate(
                    _activations[-1], self._tf_weights[i + 1], self._tf_bias[i + 1], True))
        return _activations

    @NNTiming.timeit(level=4)
    def _get_l2_loss(self, lb):
        if lb <= 0:
            return 0
        return lb * tf.reduce_sum([tf.nn.l2_loss(_w) for i, _w in enumerate(self._tf_weights)])

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

    @NNTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x=None, y=None, lr=0.001, lb=0.001, epoch=10,
            batch_size=512, optimizer=None):

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

        train_len = len(x)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len >= batch_size
        train_repeat = int(train_len / batch_size) + 1

        self._tfx = tf.placeholder(tf.float32, shape=[None, *x.shape[1:]])
        self._tfy = tf.placeholder(tf.float32, shape=[None, y.shape[1]])

        with self._sess.as_default() as sess:

            # Session
            self._cost = self.get_rs(self._tfx, self._tfy) + self._get_l2_loss(lb)
            self._y_pred = self.get_rs(self._tfx)
            self._activations = self._get_activations(self._tfx)
            self._init_train_step(sess)

            for counter in range(epoch):
                for _i in range(train_repeat):
                    if do_random_batch:
                        batch = np.random.choice(train_len, batch_size)
                        x_batch, y_batch = x[batch], y[batch]
                    else:
                        x_batch, y_batch = x, y

                    self._train_step.run(feed_dict={self._tfx: x_batch, self._tfy: y_batch})

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
    def evaluate(self, x, y):
        y_pred = self.predict_classes(x)
        y_arg = np.argmax(y, axis=1)
        print("Acc: {:8.6}".format(np.sum(y_arg == y_pred) / len(y_arg)))
