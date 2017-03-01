import matplotlib.pyplot as plt

from Zhihu.NN._extra.Layers import *
from Zhihu.NN._extra.Optimizers import *


class NNDist:
    NNTiming = Timing()

    def __init__(self):
        self._layers, self._weights, self._bias = [], [], []
        self._w_optimizer = self._b_optimizer = None
        self._current_dimension = 0

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
            self._add_weight((1, 1))
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
    def _get_prediction(self, x):
        return self._get_activations(x).pop()

    @NNTiming.timeit(level=1)
    def _get_activations(self, x):
        _activations = [self._layers[0].activate(x, self._weights[0], self._bias[0])]
        for i, layer in enumerate(self._layers[1:]):
            _activations.append(layer.activate(
                _activations[-1], self._weights[i + 1], self._bias[i + 1]))
        return _activations

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
    def fit(self, x=None, y=None, lr=0.01, epoch=10):
        # Initialize
        self._add_cost_layer()
        self._init_optimizers(lr)
        layer_width = len(self._layers)
        # Train
        for counter in range(epoch):
            self._w_optimizer.update(); self._b_optimizer.update()
            _activations = self._get_activations(x)
            _deltas = [self._layers[-1].bp_first(y, _activations[-1])]
            for i in range(-1, -len(_activations), -1):
                _deltas.append(self._layers[i - 1].bp(
                    _activations[i - 1], self._weights[i], _deltas[-1]
                ))
            for i in range(layer_width - 2, 0, -1):
                self._opt(i, _activations[i - 1], _deltas[layer_width-i-1])
            self._opt(0, x, _deltas[-1])

    @NNTiming.timeit(level=4, prefix="[API] ")
    def predict(self, x):
        return self._get_prediction(x)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def evaluate(self, x, y):
        y_pred = self.predict(x)
        y_arg = np.argmax(y, axis=1)
        y_pred_arg = np.argmax(y_pred, axis=1)
        print("Acc: {:8.6}".format(np.sum(y_arg == y_pred_arg) / len(y_arg)))

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
