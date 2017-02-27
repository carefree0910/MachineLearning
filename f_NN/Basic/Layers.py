import numpy as np

from Util.Timing import Timing


# Abstract Layer

class Layer:
    LayerTiming = Timing()

    def __init__(self, shape):
        """
        :param shape: shape[0] = units of previous layer
                      shape[1] = units of current layer (self)
        """
        self.shape = shape
        self.child = None

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    # Core

    def _activate(self, x, predict):
        pass

    def derivative(self, y):
        pass

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias, predict=False):
        if not isinstance(self, CostLayer):
            return self._activate(x.dot(w) + bias, predict)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def bp(self, y, w, prev_delta):
        if self.child is not None:
            return prev_delta
        return prev_delta.dot(w.T) * self.derivative(y)


# Activation Layers

class Sigmoid(Layer):
    def _activate(self, x, predict):
        return 1 / (1 + np.exp(-x))

    def derivative(self, y):
        return y * (1 - y)


class Tanh(Layer):
    def _activate(self, x, predict):
        return np.tanh(x)

    def derivative(self, y):
        return 1 - y ** 2


class ReLU(Layer):
    def _activate(self, x, predict):
        return np.maximum(0, x)

    def derivative(self, y):
        return y != 0


class ELU(Layer):
    def _activate(self, x, predict):
        _rs, _rs0 = x.copy(), x < 0
        _rs[_rs0] = np.exp(_rs[_rs0]) - 1
        return _rs

    def derivative(self, y):
        _rs, _indices = np.ones(y.shape), y < 0
        _rs[_indices] = y[_indices] + 1
        return _rs


class Softplus(Layer):
    def _activate(self, x, predict):
        return np.log(1 + np.exp(x))

    def derivative(self, y):
        return 1 - 1 / np.exp(y)


class Identical(Layer):
    def _activate(self, x, predict):
        return x

    def derivative(self, y):
        return 1


class Softmax(Layer):
    @staticmethod
    def safe_exp(y):
        return np.exp(y - np.max(y, axis=1, keepdims=True))

    def _activate(self, x, predict):
        exp_y = Softmax.safe_exp(x)
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    def derivative(self, y):
        return y * (1 - y)


# Cost Layer

class CostLayer(Layer):
    CostLayerTiming = Timing()

    def __init__(self, parent, shape, cost_function="Log Likelihood"):
        Layer.__init__(self, shape)
        self._parent = parent
        self._available_cost_functions = {
            "MSE": CostLayer._mse,
            "Cross Entropy": CostLayer._cross_entropy,
            "Log Likelihood": CostLayer._log_likelihood
        }
        self._cost_function_name = cost_function
        self._cost_function = self._available_cost_functions[cost_function]

    def __str__(self):
        return self._cost_function_name

    def _activate(self, x, predict):
        raise ValueError("activate function should not be called in CostLayer")

    def derivative(self, y):
        raise ValueError("derivative function should not be called in CostLayer")

    @CostLayerTiming.timeit(level=1, prefix="[Core] ")
    def bp_first(self, y, y_pred):
        if self._parent.name == "Sigmoid" and self._cost_function_name == "Cross Entropy":
            return y - y_pred
        if self._parent.name == "Softmax" and self._cost_function_name == "Log Likelihood":
            return -self._cost_function(y, y_pred) / 4
        return -self._cost_function(y, y_pred) * self._parent.derivative(y_pred)

    @property
    def calculate(self):
        return lambda y, y_pred: self._cost_function(y, y_pred, False)

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
        # noinspection PyTypeChecker
        return np.average(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

    @classmethod
    def _log_likelihood(cls, y, y_pred, diff=True, eps=1e-8):
        y_arg_max = np.argmax(y, axis=1)
        if diff:
            y_pred[range(len(y_pred)), y_arg_max] -= 1
            return y_pred
        return np.sum(-np.log(y_pred[range(len(y_pred)), y_arg_max] + eps)) / len(y)
