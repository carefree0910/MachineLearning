# encoding: utf8

import numpy as np
from abc import ABCMeta, abstractmethod

from Errors import *
from Util import Timing


# Abstract Layers

class Layer(metaclass=ABCMeta):

    LayerTiming = Timing()

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

    @classmethod
    def feed_timing(cls, timing):
        if isinstance(timing, Timing):
            cls.LayerTiming = timing

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

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        if bias is None:
            return self._activate(x.dot(w), predict)
        return self._activate(x.dot(w) + bias, predict)

    @abstractmethod
    def _activate(self, x, predict):
        raise NotImplementedError("Please implement activation function for {}".format(self.name))

    @abstractmethod
    def derivative(self, x):
        raise NotImplementedError("Please implement derivative function for {}".format(self.name))

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def bp(self, x, w, prev_delta):
        if isinstance(self._last_sub_layer, CostLayer):
            return prev_delta.dot(w.T)
        return prev_delta.dot(w.T) * self.derivative(x)

    # Util

    @staticmethod
    @LayerTiming.timeit(level=2, prefix="[Core Util] ")
    def safe_exp(y):
        return np.exp(y - np.max(y, axis=1, keepdims=True))

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Please provide a name for your layer")

    def __repr__(self):
        return str(self)


class SubLayer(Layer):

    def __init__(self, shape, parent):

        Layer.__init__(self, shape)
        self.parent = parent
        parent.child = self
        self._root = None
        self.description = ""

    @property
    def root(self):
        _parent = self.parent
        while _parent.parent:
            _parent = _parent.parent
        return _parent

    @root.setter
    def root(self, value):
        self._root = value

    def _activate(self, x, predict):
        raise NotImplementedError("Please implement activation function for a SubLayer")

    def derivative(self, x):
        raise NotImplementedError("Please implement derivative function for a SubLayer")

    def __str__(self):
        raise NotImplementedError("Please provide a name for your layer")


# Activation Layers

class Tanh(Layer):

    def _activate(self, x, predict):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - x ** 2

    def __str__(self):
        return "Tanh"


class Sigmoid(Layer):

    def _activate(self, x, predict):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)

    def __str__(self):
        return "Sigmoid"


class ELU(Layer):

    def _activate(self, x, predict):
        _rs, _rs0 = x.copy(), x < 0
        _rs[_rs0] = np.exp(_rs[_rs0]) - 1
        return _rs

    def derivative(self, x):
        _rs, _arg0 = np.zeros(x.shape), x < 0
        _rs[_arg0], _rs[~_arg0] = x[_arg0] + 1, 1
        return _rs

    def __str__(self):
        return "ELU"


class ReLU(Layer):

    def _activate(self, x, predict):
        return np.maximum(0, x)

    def derivative(self, x):
        return x > 0

    def __str__(self):
        return "ReLU"


class Softplus(Layer):

    def _activate(self, x, predict):
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        return 1 / (1 + 1 / (np.exp(x) - 1))

    def __str__(self):
        return "Softplus"


class Softmax(Layer):

    def _activate(self, x, predict):
        exp_y = Layer.safe_exp(x)
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    def derivative(self, x):
        return x * (1 - x)

    def __str__(self):
        return "Softmax"

    __repr__ = __str__


class Identical(Layer):

    def _activate(self, x, predict):
        return x

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
        self.description = "(Drop prob: {})".format(prob)

    def _activate(self, x, predict):
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

    def _activate(self, x, predict):
        return x

    def derivative(self, x):
        raise LayerError("derivative function should not be called in CostLayer")

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
            y_pred = y_pred.copy()
            y_pred[cls._batch_range, y_arg_max] -= 1
            return y_pred
        return np.sum(-np.log(y_pred[range(len(y_pred)), y_arg_max])) / len(y)

    def __str__(self):
        return self._cost_function_name
