import os
import time
import pickle
import platform
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import *
from Util import ProgressBar

np.random.seed(142857)  # for reproducibility


# Errors

class LayerError(Exception):
    pass


class BuildLayerError(Exception):
    pass


class BuildNetworkError(Exception):
    pass


# Abstract Layers

class Layer:

    def __init__(self, shape):
        """
        :param shape: shape[0] = units of previous layer
                      shape[1] = units of current layer (self)
        """
        self._shape = shape
        self.parent = None
        self.child = None
        self.is_last_root = False
        self._last_sub_layer = None

    @property
    def name(self):
        return str(self)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def root(self):
        return self

    @root.setter
    def root(self, value):
        raise BuildLayerError("Setting Layer's root is not permitted")

    @property
    def last_sub_layer(self):
        _child = self.child
        if not _child:
            return None
        while _child.child:
            _child = _child.child
        return _child

    @last_sub_layer.setter
    def last_sub_layer(self, value):
            self._last_sub_layer = value

    # Core

    def activate(self, x, w, bias=None, predict=False):
        raise NotImplementedError("Please implement activation function for {}".format(self.name))

    def derivative(self, x):
        raise NotImplementedError("Please implement derivative function for {}".format(self.name))

    def bp(self, x, w, prev_delta):
        if isinstance(self._last_sub_layer, CostLayer):
            return prev_delta.dot(w.T)
        return prev_delta.dot(w.T) * self.derivative(x)

    # Util

    @staticmethod
    def safe_exp(y):
        exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    def __str__(self):
        raise NotImplementedError("Please provide a name for your layer")

    __repr__ = __str__


class SubLayer(Layer):

    def __init__(self, shape, parent):

        Layer.__init__(self, shape)
        self.parent = parent
        parent.child = self
        self._root = None

    @property
    def root(self):
        _parent = self.parent
        while _parent.parent:
            _parent = _parent.parent
        return _parent

    @root.setter
    def root(self, value):
        self._root = value

    def activate(self, x, w, bias=None, predict=False):
        raise NotImplementedError("Please implement activation function for a SubLayer")

    def derivative(self, x):
        raise NotImplementedError("Please implement derivative function for a SubLayer")

    def __str__(self):
        raise NotImplementedError("Please provide a name for your layer")


# Activation Layers

class Tanh(Layer):

    def activate(self, x, w, bias=None, predict=False):
        if bias is None:
            return np.tanh(x.dot(w))
        return np.tanh(x.dot(w) + bias)

    def derivative(self, x):
        return 1 - x ** 2

    def __str__(self):
        return "Tanh"


class Sigmoid(Layer):

    def activate(self, x, w, bias=None, predict=False):
        if bias is None:
            return 1 / (1 + np.exp(-x.dot(w)))
        return 1 / (1 + np.exp(-x.dot(w) - bias))

    def derivative(self, x):
        return x * (1 - x)

    def __str__(self):
        return "Sigmoid"


class ELU(Layer):

    def activate(self, x, w, bias=None, predict=False):
        _rs = x.dot(w) if bias is None else x.dot(w) + bias
        _rs0 = _rs < 0
        _rs[_rs0] = np.exp(_rs[_rs0]) - 1
        return _rs

    def derivative(self, x):
        _rs, _arg0 = np.zeros(x.shape), x < 0
        _rs[_arg0], _rs[~_arg0] = x[_arg0] + 1, 1
        return _rs

    def __str__(self):
        return "ELU"


class ReLU(Layer):

    def activate(self, x, w, bias=None, predict=False):
        if bias is None:
            return np.maximum(0, x.dot(w))
        return np.maximum(0, x.dot(w) + bias)

    def derivative(self, x):
        return x > 0

    def __str__(self):
        return "ReLU"


class Softplus(Layer):

    def activate(self, x, w, bias=None, predict=False):
        if bias is None:
            return np.log(1 + np.exp(x.dot(w)))
        return np.log(1 + np.exp(x.dot(w) + bias))

    def derivative(self, x):
        return 1 / (1 + 1 / (np.exp(x) - 1))

    def __str__(self):
        return "Softplus"


class Softmax(Layer):

    def activate(self, x, w, bias=None, predict=False):
        if bias is None:
            return Layer.safe_exp(x.dot(w))
        return Layer.safe_exp(x.dot(w) + bias)

    def derivative(self, x):
        return x * (1 - x)

    def __str__(self):
        return "Softmax"


class Identical(Layer):

    def activate(self, x, w, bias=None, predict=False):
        if bias is None:
            return x.dot(w)
        return x.dot(w) + bias

    def derivative(self, x):
        return 1

    def __str__(self):
        return "Identical"


# Special Layer

class Dropout(SubLayer):

    def __init__(self, shape, parent, prob=0.5):

        if prob < 0 or prob >= 1:
            raise BuildLayerError("Probability of Dropout should be a positive float smaller than 1")

        SubLayer.__init__(self, shape, parent)
        self._prob = prob
        self._div_prob = 1 / (1 - self._prob)

    def activate(self, x, w, bias=None, predict=False):
        if not predict:
            _rand_diag = np.random.random(x.shape[1]) >= self._prob
            _diag = np.diag(_rand_diag) * self._div_prob
        else:
            _diag = np.eye(x.shape[1])
        return x.dot(_diag)

    def derivative(self, x):
        return self._div_prob

    def __str__(self):
        return "Dropout"


# Cost Layer

class CostLayer(SubLayer):

    # Optimization
    _batch_range = None

    def __init__(self, shape, parent, cost_function="MSE"):

        SubLayer.__init__(self, shape, parent)
        self._available_cost_functions = {
            "MSE": CostLayer._mse,
            "Cross Entropy": CostLayer._cross_entropy,
            "Log Likelihood": CostLayer._log_likelihood
        }

        if cost_function not in self._available_cost_functions:
            raise LayerError("Cost function '{}' not implemented".format(cost_function))
        self._cost_function_name = cost_function
        self._cost_function = self._available_cost_functions[cost_function]

    def activate(self, x, w, bias=None, predict=False):
        return x

    def derivative(self, x):
        raise LayerError("derivative function should not be called in CostLayer")

    def __str__(self):
        return self._cost_function_name

    def bp_first(self, y, y_pred):
        if self._root.name == "Sigmoid" and self.cost_function == "Cross Entropy":
            return y * (1 - y_pred) - (1 - y) * y_pred
        if self.cost_function == "Log Likelihood":
            return -self._cost_function(y, y_pred) / 4
        return -self._cost_function(y, y_pred) * self._root.derivative(y_pred)

    @property
    def calculate(self):
        return lambda y, y_pred: self._cost_function(y, y_pred, False)

    @property
    def cost_function(self):
        return self._cost_function_name

    @cost_function.setter
    def cost_function(self, value):
        if value not in self._available_cost_functions:
            raise LayerError("'{}' is not implemented".format(value))
        self._cost_function_name = value
        self._cost_function = self._available_cost_functions[value]

    def set_cost_function_derivative(self, func, name=None):
        name = "Custom Cost Function" if name is None else name
        self._cost_function_name = name
        self._cost_function = func

    # Cost Functions

    @staticmethod
    def _mse(y, y_pred, diff=True):
        if diff:
            return -y + y_pred
        assert_string = "y or y_pred should be np.ndarray in cost function"
        assert isinstance(y, np.ndarray) or isinstance(y_pred, np.ndarray), assert_string
        return 0.5 * np.average((y - y_pred) ** 2)

    @staticmethod
    def _cross_entropy(y, y_pred, diff=True):
        if diff:
            return -y / y_pred + (1 - y) / (1 - y_pred)
        assert_string = "y or y_pred should be np.ndarray in cost function"
        assert isinstance(y, np.ndarray) or isinstance(y_pred, np.ndarray), assert_string
        return np.average(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

    @classmethod
    def _log_likelihood(cls, y, y_pred, diff=True):
        if cls._batch_range is None:
            cls._batch_range = np.arange(len(y_pred))
        y_arg_max = np.argmax(y, axis=1)
        if diff:
            y_pred[cls._batch_range, y_arg_max] -= 1
            return y_pred
        return np.sum(-np.log(Layer.safe_exp(y_pred)[:, y_arg_max])) / len(y)


# Neural Network

class NN:

    def __init__(self):
        self._layers, self._weights, self._bias = [], [], []
        self._layer_names, self._layer_shapes = [], []
        self._data_size = 0

        self._whether_apply_bias = False
        self._current_dimension = 0
        self._cost_layer = "Undefined"

        self._logs = []
        self._metrics, self._metric_names = [], []

        self._x, self._y = None, None
        self._x_min, self._x_max = 0, 0
        self._y_min, self._y_max = 0, 0

        # Sets & Dictionaries

        self._available_metrics = {
            "acc": NN._acc, "_acc": NN._acc,
            "f1": NN._f1_score, "_f1_score": NN._f1_score
        }
        self._available_root_layers = {
            "Tanh": Tanh, "Sigmoid": Sigmoid,
            "ELU": ELU, "ReLU": ReLU, "Softplus": Softplus,
            "Softmax": Softmax
        }
        self._available_sub_layers = {
            "Dropout", "MSE", "Cross Entropy", "Log Likelihood"
        }
        self._available_cost_functions = {
            "MSE", "Cross Entropy", "Log Likelihood"
        }
        self._available_special_layers = {
            "Dropout": Dropout
        }

    def initialize(self):
        self._layers, self._weights, self._bias = [], [], []
        self._layer_names, self._layer_shapes = [], []
        self._whether_apply_bias = False
        self._current_dimension = 0

        self._logs = []
        self._metrics, self._metric_names = [], []

    @property
    def name(self):
        return (
            "-".join([str(_layer.shape[1]) for _layer in self._layers]) +
            " at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        )

    @property
    def layer_names(self):
        return [layer.name for layer in self._layers]

    @layer_names.setter
    def layer_names(self, value):
        self._layer_names = value

    @property
    def layer_shapes(self):
        return [layer.shape for layer in self._layers]

    @layer_shapes.setter
    def layer_shapes(self, value):
        self._layer_shapes = value

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

    # Utils

    def _feed_data(self, x, y):
        if len(x) != len(y):
            raise BuildNetworkError("Data fed to network should be identical in length, x: {} and y: {} found".format(
                len(x), len(y)
            ))
        if x is None:
            if self._x is None:
                raise BuildNetworkError("Please provide input matrix")
            x = self._x
        if y is None:
            if self._y is None:
                raise BuildNetworkError("Please provide input matrix")
            y = self._y
        self._x, self._y = x, y
        self._x_min, self._x_max = np.min(x), np.max(x)
        self._y_min, self._y_max = np.min(y), np.max(y)
        self._data_size = len(x)
        return x, y

    def _add_weight(self, shape):
        self._weights.append(2 * np.random.random(shape) - 1)
        self._bias.append(np.zeros((1, shape[1])))

    def _add_layer(self, layer, *args):
        _parent = self._layers[-1]
        if isinstance(_parent, CostLayer):
            raise BuildLayerError("Adding layer after CostLayer is not permitted")
        if isinstance(layer, str):
            if layer not in self._available_sub_layers:
                raise BuildLayerError("Invalid SubLayer '{}' provided".format(layer))
            _current, _next = _parent.shape[1], self._current_dimension
            if layer in self._available_cost_functions:
                layer = CostLayer((_current, _next), _parent, layer)
            else:
                if args:
                    if layer == "Dropout":
                        try:
                            prob = float(args[0])
                            layer = Dropout((_current, _next), _parent, prob)
                        except ValueError as err:
                            raise BuildLayerError("Invalid parameter for Dropout: '{}'".format(err))
                        except BuildLayerError as err:
                            raise BuildLayerError("Invalid parameter for Dropout: {}".format(err))
                else:
                    layer = self._available_special_layers[layer]((_current, _next), _parent)
        else:
            _current, _next = args
        if isinstance(layer, SubLayer):
            if not isinstance(layer, CostLayer) and _current != _parent.shape[1]:
                raise BuildLayerError("Output shape should be identical with input shape "
                                      "if chosen SubLayer is not a CostLayer")
            _parent.child = layer
            layer.root = layer.root
            layer.root.last_sub_layer = layer
            if isinstance(layer, CostLayer):
                layer.root.is_last_root = True
            self.parent = _parent
            self._layers.append(layer)
            self._weights.append(np.eye(_current))
            self._bias.append(np.zeros((1, _current)))
            self._current_dimension = _next
            return
        self._layers.append(layer)
        self._add_weight((_current, _next))
        self._current_dimension = _next

    def _add_cost_layer(self):
        _last_layer = self._layers[-1]
        _last_layer_root = _last_layer.root
        if not isinstance(_last_layer, CostLayer):
            if _last_layer_root.name == "Sigmoid":
                self._cost_layer = "Cross Entropy"
            elif _last_layer_root.name == "Softmax":
                self._cost_layer = "Log Likelihood"
            else:
                self._cost_layer = "MSE"
            self.add(self._cost_layer)

    def _get_activations(self, x, predict=False):
        _activations = [self._layers[0].activate(x, self._weights[0], self._bias[0], predict)]
        for i, layer in enumerate(self._layers[1:]):
            _activations.append(layer.activate(
                _activations[-1], self._weights[i + 1], self._bias[i + 1], predict))
        return _activations

    def _get_prediction(self, x):
        return self._get_activations(x, predict=True).pop()

    def _get_accuracy(self, x, y):
        y_pred = self._get_prediction(x)
        return NN._acc(y, y_pred)

    def _append_log(self, x, y):
        y_pred = self._get_prediction(x)
        for i, metric in enumerate(self._metrics):
            self._logs[i].append(metric(y, y_pred))

    def _print_metric_logs(self, x, y, show_loss):
        print()
        print("-" * 30)
        for i, name in enumerate(self._metric_names):
            print("{:<16s}: {:12.8}".format(name, self._logs[i][-1]))
        if show_loss:
            print("{:<16s}: {:12.8}".format("Loss", self._layers[-1].calculate(y, self.predict(x)) / self._data_size))
        print("-" * 30)

    # API

    def feed(self, x, y):
        self._feed_data(x, y)

    def add(self, layer, *args):
        if isinstance(layer, str):
            self._add_layer(layer, *args)
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
                    if isinstance(layer, SubLayer):
                        if _next != self._current_dimension:
                            raise BuildLayerError("Invalid SubLayer provided (shape[1] should be {}, {} found)".format(
                                self._current_dimension, _next
                            ))
                    elif _current != self._current_dimension:
                        raise BuildLayerError("Invalid Layer provided (shape[0] should be {}, {} found)".format(
                            self._current_dimension, _current
                        ))
                    self._add_layer(layer, _current, _next)

                elif len(layer.shape) == 1:
                    _next = layer.shape[0]
                    layer.shape = (self._current_dimension, _next)
                    self._add_layer(layer, self._current_dimension, _next)
                else:
                    raise LayerError("Invalid Layer provided (invalid shape '{}' found)".format(layer.shape))

    def build(self, units="build"):
        if isinstance(units, str):
            if units == "build":
                for name, shape in zip(self._layer_names, self._layer_shapes):
                    try:
                        self.add(self._available_root_layers[name](shape))
                    except KeyError:
                        self.add(name)
                self._add_cost_layer()
            elif units == "build from load":
                self.add(self._cost_layer)
            else:
                raise NotImplementedError("Invalid param '{}' provided to 'build' method".format(units))
        else:
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
                self.add(Sigmoid((unit_num, )))
            self._add_cost_layer()

    def preview(self, add_cost=True):
        if not self._layers:
            rs = "None"
        else:
            if add_cost:
                self._add_cost_layer()
            rs = (
                "Input  :  {:<10s} - {}\n".format("Dimension", self._layers[0].shape[0]) +
                "\n".join(["Layer  :  {:<10s} - {}".format(
                    _layer.name, _layer.shape[1]
                ) for _layer in self._layers[:-1]]) +
                "\nCost   :  {:<10s}\n".format(self._cost_layer)
            )
        print("=" * 30 + "\n" + "Structure\n" + "-" * 30 + "\n" + rs + "\n" + "-" * 30 + "\n")

    @staticmethod
    def split_data(x, y, train_only, training_scale=TRAINING_SCALE, cv_scale=CV_SCALE):
        if train_only:
            train_len = len(x)
            x_train, y_train = np.array(x[:train_len]), np.array(y[:train_len])
            x_cv, y_cv, x_test, y_test = x_train, y_train, x_train, y_train
        else:
            shuffle_suffix = np.random.permutation(len(x))
            x, y = x[shuffle_suffix], y[shuffle_suffix]
            train_len = int(len(x) * training_scale)
            cv_len = train_len + int(len(x) * cv_scale)
            x_train, y_train = np.array(x[:train_len]), np.array(y[:train_len])
            x_cv, y_cv = np.array(x[train_len:cv_len]), np.array(y[train_len:cv_len])
            x_test, y_test = np.array(x[cv_len:]), np.array(y[cv_len:])

        if BOOST_LESS_SAMPLES:
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

        return (x_train, x_cv, x_test), (y_train, y_cv, y_test)

    def fit(self,
            x=None, y=None,
            lr=LEARNING_RATE, lb=0.01, apply_bias=True,
            epoch=EPOCH, batch_size=BATCH_SIZE,
            train_only=TRAIN_ONLY, record_period=RECORD_PERIOD,
            metrics=None, do_log=True, print_log=False, show_loss=False, debug=False,
            visualize=False, visualize_setting=None):

        x, y = self._feed_data(x, y)

        if not self._layers:
            raise BuildNetworkError("Please provide layers before fitting data")
        self._add_cost_layer()

        if y.shape[1] != self._current_dimension:
            raise BuildNetworkError("Output layer's shape should be {}, {} found".format(
                self._current_dimension, y.shape[1]))

        (x_train, x_cv, x_test), (y_train, y_cv, y_test) = NN.split_data(x, y, train_only)
        train_len = len(x_train)
        do_random_batch = train_len >= batch_size
        train_repeat = int(train_len / batch_size) + 1
        self._feed_data(x_train, y_train)

        self._metrics = ["acc"] if metrics is None else metrics
        for i, metric in enumerate(self._metrics):
            if isinstance(metric, str):
                if metric not in self._available_metrics:
                    raise BuildNetworkError("Metric '{}' is not implemented".format(metric))
                self._metrics[i] = self._available_metrics[metric]
        self._metric_names = [str(_m).split()[1].split(".")[-1] for _m in self._metrics]

        self._logs = [[] for _ in range(len(self._metrics))]

        layer_width = len(self._layers)
        self._whether_apply_bias = apply_bias

        regularization_param = 1 - lb * lr / self._data_size

        bar = ProgressBar(min_value=0, max_value=max(1, epoch // record_period))
        bar.start()

        for counter in range(epoch):
            for _ in range(train_repeat):

                if do_random_batch:
                    batch = np.random.randint(train_len, size=batch_size)
                    x_batch, y_batch = x_train[batch], y_train[batch]
                else:
                    x_batch, y_batch = x_train, y_train

                _activations = self._get_activations(x_batch)

                _deltas = [self._layers[-1].bp_first(y_batch, _activations[-1])]
                for i in range(-1, -len(_activations), -1):
                    _deltas.append(self._layers[i - 1].bp(_activations[i - 1], self._weights[i], _deltas[-1]))

                for i in range(layer_width - 1, 0, -1):
                    if not isinstance(self._layers[i], SubLayer):
                        _delta = _deltas[layer_width - i - 1]
                        self._weights[i] *= regularization_param
                        self._weights[i] += _activations[i - 1].T.dot(_delta) * lr
                        if apply_bias:
                            self._bias[i] += np.sum(_delta, axis=0, keepdims=True) * lr
                _delta = _deltas[-1]
                self._weights[0] += x_batch.T.dot(_delta) * lr
                if apply_bias:
                    self._bias[0] += np.sum(_delta, axis=0, keepdims=True) * lr

                if debug:
                    pass

            if do_log:
                self._append_log(x_cv, y_cv)

            if (counter + 1) % record_period == 0:
                if do_log and print_log:
                    self._print_metric_logs(x_cv, y_cv, show_loss)
                if visualize:
                    if visualize_setting is None:
                        self.do_visualization(x_cv, y_cv)
                    else:
                        self.do_visualization(x_cv, y_cv, *visualize_setting)
                bar.update(counter // record_period + 1)

        if do_log:
            self._append_log(x_test, y_test)

        return self._logs

    def save(self, path=None, name=None, overwrite=True):

        path = "Models" if path is None else path
        name = "Model.nn" if name is None else name
        if not os.path.exists(path):
            os.mkdir(path)
        slash = "\\" if platform.system() == "Windows" else "/"

        _dir = path + slash + name
        if not overwrite and os.path.isfile(_dir):
            _count = 1
            _new_dir = _dir + "({})".format(_count)
            while os.path.isfile(_new_dir):
                _count += 1
                _new_dir = _dir + "({})".format(_count)
            _dir = _new_dir

        with open(_dir, "wb") as file:
            pickle.dump({
                "_logs": self._logs,
                "_metric_names": self._metric_names,
                "_layers": self._layers[:-1],
                "_layer_names": self.layer_names,
                "_layer_shapes": self.layer_shapes,
                "_cost_layer": self._layers[-1].name,
                "_weights": self._weights,
                "_next_dimension": self._current_dimension
            }, file)

    def load(self, path):
        self.initialize()
        try:
            with open(path, "rb") as file:
                _dic = pickle.load(file)
                for key, value in _dic.items():
                    setattr(self, key, value)
                self.build("build from load")
                for i in range(len(self._metric_names) - 1, -1, -1):
                    name = self._metric_names[i]
                    if name not in self._available_metrics:
                        self._metric_names.pop(i)
                    else:
                        self._metrics.insert(0, self._available_metrics[name])
                return _dic
        except Exception as err:
            raise BuildNetworkError("Failed to load Network ({}), structure initialized".format(err))

    def predict(self, x):
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape((1, len(x)))
        return self._get_prediction(x)

    def predict_classes(self, x, flatten=True):
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape((1, len(x)))
        if flatten:
            return np.argmax(self._get_prediction(x), axis=1)
        return np.argmax([self._get_prediction(x)], axis=2).T

    def evaluate(self, x, y, metrics=None):
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
        logs, y_pred = [], self._get_prediction(x)
        for metric in metrics:
            logs.append(metric(y, y_pred))
        return logs

    def do_visualization(self, x=None, y=None, plot_scale=2, plot_precision=10 ** -2):

        x = self._x if x is None else x
        y = self._y if y is None else y

        plot_num = int(1 / plot_precision)

        xf = np.linspace(self._x_min * plot_scale, self._x_max * plot_scale, plot_num)
        yf = np.linspace(self._x_min * plot_scale, self._x_max * plot_scale, plot_num)
        input_x, input_y = np.meshgrid(xf, yf)
        input_xs = np.c_[input_x.ravel(), input_y.ravel()]

        if self._x.shape[1] != 2:
            output_ys_2d = np.argmax(self.predict(np.c_[
                input_xs, self._x[:, 2:][0]
            ]), axis=1).reshape((len(xf), len(yf)))
            output_ys_3d = self.predict(np.c_[
                input_xs, self._x[:, 2:][0]
            ])[:, 0].reshape((len(xf), len(yf)))
        else:
            output_ys_2d = np.argmax(self.predict(input_xs), axis=1).reshape((len(xf), len(yf)))
            output_ys_3d = self.predict(input_xs)[:, 0].reshape((len(xf), len(yf)))

        xf, yf = np.meshgrid(xf, yf, sparse=True)

        plt.contourf(input_x, input_y, output_ys_2d, cmap=cm.Spectral)
        plt.scatter(x[:, 0], x[:, 1], c=np.argmax(y, axis=1), s=40, cmap=cm.Spectral)
        plt.axis("off")
        plt.show()

        if self._y.shape[1] == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(xf, yf, output_ys_3d, cmap=cm.coolwarm,)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.show()
