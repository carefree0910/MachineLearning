from Zhihu.NN.Layers import *
from Zhihu.NN.Optimizers import *

np.random.seed(142857)  # for reproducibility


# Neural Network

class NNBase:
    NNTiming = Timing()

    def __init__(self):
        self._layers = []
        self._optimizer = None
        self._current_dimension = 0

        self._tfx = self._tfy = None
        self._tf_weights, self._tf_bias = [], []
        self._cost = self._y_pred = None

        self._train_step = None

    def __str__(self):
        return "Neural Network"

    __repr__ = __str__

    def feed_timing(self, timing):
        if isinstance(timing, Timing):
            self.NNTiming = timing
            for layer in self._layers:
                layer.feed_timing(timing)

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

    def _get_rs(self, x, y=None):
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
        self._sess = tf.Session()

    # Utils

    def _get_prediction(self, x):
        with self._sess.as_default():
            return self._get_rs(x).eval(feed_dict={self._tfx: x})

    # API

    def fit(self, x=None, y=None, lr=0.001, epoch=10):
        self._optimizer = Adam(lr)
        self._tfx = tf.placeholder(tf.float32, shape=[None, x.shape[1]])
        self._tfy = tf.placeholder(tf.float32, shape=[None, y.shape[1]])
        with self._sess.as_default() as sess:
            # Define session
            self._cost = self._get_rs(self._tfx, self._tfy)
            self._y_pred = self._get_rs(self._tfx)
            self._train_step = self._optimizer.minimize(self._cost)
            sess.run(tf.global_variables_initializer())
            # Train
            for counter in range(epoch):
                self._train_step.run(feed_dict={self._tfx: x, self._tfy: y})

    def predict_classes(self, x):
        x = np.array(x)
        return np.argmax(self._get_prediction(x), axis=1)

    def evaluate(self, x, y):
        y_pred = self.predict_classes(x)
        y_arg = np.argmax(y, axis=1)
        print("Acc: {:8.6}".format(np.sum(y_arg == y_pred) / len(y_arg)))
