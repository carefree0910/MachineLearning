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


# Cost Layer

class CostLayer(Layer):

    # Optimization
    _batch_range = None

    def __init__(self, shape, cost_function="MSE", transform=None):
        Layer.__init__(self, shape)
        self._available_cost_functions = {
            "MSE": CostLayer._mse,
            "SVM": CostLayer._svm,
            "CrossEntropy": CostLayer._cross_entropy
        }
        self._available_transform_functions = {
            "Softmax": CostLayer._softmax,
            "Sigmoid": CostLayer._sigmoid
        }
        self._cost_function_name = cost_function
        if transform is None and cost_function == "CrossEntropy":
            self._transform = "Softmax"
            self._transform_function = CostLayer._softmax
        else:
            self._transform = transform
            self._transform_function = self._available_transform_functions.get(transform, None)
        self._cost_function = self._available_cost_functions[cost_function]

    def _activate(self, x, predict):
        if self._transform_function is None:
            return x
        return self._transform_function(x)

    def _derivative(self, y, delta=None):
        pass

    def bp_first(self, y, y_pred):
        if self._cost_function_name == "CrossEntropy" and (
                self._transform == "Softmax" or self._transform == "Sigmoid"):
            return y - y_pred
        # TODO: Support bp with transform function (define derivative for transform function)
        return -self._cost_function(y, y_pred)

    @property
    def calculate(self):
        return lambda y, y_pred: self._cost_function(y, y_pred, False)

    @property
    def cost_function(self):
        return self._cost_function_name

    @cost_function.setter
    def cost_function(self, value):
        self._cost_function_name = value
        self._cost_function = self._available_cost_functions[value]

    def set_cost_function_derivative(self, func, name=None):
        name = "Custom Cost Function" if name is None else name
        self._cost_function_name = name
        self._cost_function = func

    # Transform Functions

    @staticmethod
    def safe_exp(x):
        return np.exp(x - np.max(x, axis=1, keepdims=True))

    @staticmethod
    def _softmax(x):
        exp_x = CostLayer.safe_exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Cost Functions

    @staticmethod
    def _mse(y, y_pred, diff=True):
        if diff:
            return -y + y_pred
        assert_string = "y or y_pred should be np.ndarray in cost function"
        assert isinstance(y, np.ndarray) or isinstance(y_pred, np.ndarray), assert_string
        return 0.5 * np.average((y - y_pred) ** 2)

    @staticmethod
    def _svm(y, y_pred, diff=True):
        n, y = y_pred.shape[0], np.argmax(y, axis=1)
        correct_class_scores = y_pred[np.arange(n), y]
        margins = np.maximum(0, y_pred - correct_class_scores[:, None] + 1.0)
        margins[np.arange(n), y] = 0
        loss = np.sum(margins) / n
        num_pos = np.sum(margins > 0, axis=1)
        if not diff:
            return loss
        dx = np.zeros_like(y_pred)
        dx[margins > 0] = 1
        dx[np.arange(n), y] -= num_pos
        dx /= n
        return dx

    @staticmethod
    def _cross_entropy(y, y_pred, diff=True):
        if diff:
            return -y / y_pred + (1 - y) / (1 - y_pred)
        # noinspection PyTypeChecker
        return np.average(-y * np.log(np.maximum(y_pred, 1e-12)) - (1 - y) * np.log(np.maximum(1 - y_pred, 1e-12)))

    def __str__(self):
        return self._cost_function_name
