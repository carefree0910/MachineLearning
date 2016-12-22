from Zhihu.NN.Errors import *
from Zhihu.NN._extra.Optimizers import *
from Zhihu.NN.Util import Timing


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
        self.is_fc = False
        self.is_fc_base = False
        self.is_last_root = False
        self.is_sub_layer = False
        self._last_sub_layer = None

    def feed_timing(self, timing):
        if isinstance(timing, Timing):
            self.LayerTiming = timing

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
    def params(self):
        return self._shape,

    @property
    def special_params(self):
        return

    def set_special_params(self, dic):
        for key, value in dic.items():
            setattr(self, key, value)

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

    def derivative(self, y, delta=None):
        return self._derivative(y, delta)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        if isinstance(self, CostLayer):
            return self._activate(x + bias, predict)
        return self._activate(x.dot(w) + bias, predict)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def bp(self, y, w, prev_delta):
        if self.child is not None:
            return prev_delta
        if isinstance(self, CostLayer):
            return self._derivative(y, prev_delta.dot(w.T) * self._parent.derivative(y))
        return prev_delta.dot(w.T) * self._derivative(y)

    @abstractmethod
    def _activate(self, x, predict):
        pass

    @abstractmethod
    def _derivative(self, y, delta=None):
        pass

    # Util

    @staticmethod
    @LayerTiming.timeit(level=2, prefix="[Core Util] ")
    def safe_exp(y):
        return np.exp(y - np.max(y, axis=1, keepdims=True))

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


# Activation Layers

class Tanh(Layer):

    def _activate(self, x, predict):
        return np.tanh(x)

    def _derivative(self, y, delta=None):
        return 1 - y ** 2


class Sigmoid(Layer):

    def _activate(self, x, predict):
        return 1 / (1 + np.exp(-x))

    def _derivative(self, y, delta=None):
        return y * (1 - y)


class ELU(Layer):

    def _activate(self, x, predict):
        _rs, _rs0 = x.copy(), x < 0
        _rs[_rs0] = np.exp(_rs[_rs0]) - 1
        return _rs

    def _derivative(self, y, delta=None):
        _rs, _arg0 = np.zeros(y.shape), y < 0
        _rs[_arg0], _rs[~_arg0] = y[_arg0] + 1, 1
        return _rs


class ReLU(Layer):

    def _activate(self, x, predict):
        return np.maximum(0, x)

    def _derivative(self, y, delta=None):
        return y > 0


class Softplus(Layer):

    def _activate(self, x, predict):
        return np.log(1 + np.exp(x))

    def _derivative(self, y, delta=None):
        return 1 / (1 + 1 / (np.exp(y) - 1))


class Identical(Layer):

    def _activate(self, x, predict):
        return x

    def _derivative(self, y, delta=None):
        return 1


class Softmax(Layer):

    def _activate(self, x, predict):
        exp_y = Layer.safe_exp(x)
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    def _derivative(self, y, delta=None):
        return y * (1 - y)


# Cost Layer

class CostLayer(Layer):

    # Optimization
    _batch_range = None

    def __init__(self, parent, shape, cost_function="LogLikelihood"):

        Layer.__init__(self, shape)
        self._parent = parent
        self._available_cost_functions = {
            "MSE": CostLayer._mse,
            "SVM": CostLayer._svm,
            "CrossEntropy": CostLayer._cross_entropy,
            "LogLikelihood": CostLayer._log_likelihood
        }

        if cost_function not in self._available_cost_functions:
            raise LayerError("Cost function '{}' not implemented".format(cost_function))
        self._cost_function_name = cost_function
        self._cost_function = self._available_cost_functions[cost_function]

    def _activate(self, x, predict):
        return x

    def _derivative(self, y, delta=None):
        raise LayerError("derivative function should not be called in CostLayer")

    def bp_first(self, y, y_pred):
        if self._parent.name == "Sigmoid" and self.cost_function == "Cross Entropy":
            return y * (1 - y_pred) - (1 - y) * y_pred
        if self.cost_function == "Log Likelihood":
            return -self._cost_function(y, y_pred) / 4
        return -self._cost_function(y, y_pred) * self._parent.derivative(y_pred)

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
        assert_string = "y or y_pred should be np.ndarray in cost function"
        assert isinstance(y, np.ndarray) or isinstance(y_pred, np.ndarray), assert_string
        # noinspection PyTypeChecker
        return np.average(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

    @classmethod
    def _log_likelihood(cls, y, y_pred, diff=True, eps=1e-8):
        if cls._batch_range is None:
            cls._batch_range = np.arange(len(y_pred))
        y_arg_max = np.argmax(y, axis=1)
        if diff:
            y_pred = y_pred.copy()
            y_pred[cls._batch_range, y_arg_max] -= 1
            return y_pred
        return np.sum(-np.log(y_pred[range(len(y_pred)), y_arg_max] + eps)) / len(y)

    def __str__(self):
        return self._cost_function_name

    
# Factory

class LayerFactory:
    available_root_layers = {
        "Tanh": Tanh, "Sigmoid": Sigmoid,
        "ELU": ELU, "ReLU": ReLU, "Softplus": Softplus,
        "Softmax": Softmax,
        "Identical": Identical
    }
    available_sub_layers = {
        "Dropout", "Normalize", "ConvNorm", "ConvDrop",
        "MSE", "SVM", "Cross Entropy", "Log Likelihood"
    }
    available_cost_functions = {
        "MSE", "SVM", "Cross Entropy", "Log Likelihood"
    }

    def handle_str_main_layers(self, name, *args, **kwargs):
        if name not in self.available_sub_layers:
            if name in self.available_root_layers:
                name = self.available_root_layers[name]
            else:
                raise BuildNetworkError("Undefined layer '{}' found".format(name))
            return name(*args, **kwargs)
        return None
    
    def get_layer_by_name(self, name, parent, current_dimension, *args, **kwargs):
        _layer = self.handle_str_main_layers(name, *args, **kwargs)
        if _layer:
            return _layer, None
        _current, _next = parent.shape[1], current_dimension
        _layer = CostLayer(parent, (_current, _next), name)
        return _layer, (_current, _next)
